import argparse
import psycopg
import pgvector.psycopg

from time import perf_counter
from mdutils.mdutils import MdUtils

import common


def build_arg_parse():
    parser = argparse.ArgumentParser(description="Vectorchord Test Suite")
    common.build_arg_parse(parser)

    # Add specific arguments for Vectorchord suite
    # None

    return parser


class TestSuite(common.TestSuite):

    metric_dict = {
        "l2": "vector_l2_ops",
        "euclidean": "vector_l2_ops",
        "cos": "vector_cosine_ops",
        "ip": "vector_ip_ops",
        "dot": "vector_ip_ops"
    }

    @staticmethod
    def process_batch(args):
        import time

        # Unpack and use nprob, epsilon, etc.
        test, answer, top, metric_ops, url, table_name, nprob, epsilon = args

        conn = psycopg.connect(url)

        conn.execute("SET jit=false")
        conn.execute(f"SET vchordrq.probes=\"{nprob}\"")
        conn.execute(f"SET vchordrq.epsilon={epsilon}")

        hits = 0
        results = []

        for query, ground_truth in zip(test, answer):
            start = time.perf_counter()
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT id FROM {table_name} ORDER BY embedding {metric_ops} %s::vector LIMIT {top}",
                    (query.tolist() if hasattr(query, "tolist") else list(query),),
                )
                result = cursor.fetchall()
            end = time.perf_counter()

            result_ids = set([p[0] for p in result[:top]])
            ground_truth_ids = set(
                ground_truth[:top].tolist() if hasattr(ground_truth[:top], "tolist") else ground_truth[:top])
            hit = len(result_ids & ground_truth_ids)
            hits += hit

            results.append((hit, (start, end)))

        conn.close()
        return results


    def make_batch_args(self, test, answer, top, metric, table_name, benchmark):
        # Pass extra args for vectorchord

        # Determine metric operations based on dataset metric
        if metric in {"l2", "euclidean"}:
            metric_ops = "<->"
        elif metric in {"cos", "angular"}:
            metric_ops = "<=>"
        elif metric in {"dot", "ip"}:
            metric_ops = "<#>"
        else:
            raise ValueError("unsupported metric type")

        return (
            test,
            answer,
            top,
            metric_ops,
            self.url,
            table_name,
            benchmark["nprob"],
            benchmark["epsilon"]
        )

    def create_connection(self):
        conn = super().create_connection()
        pgvector.psycopg.register_vector(conn)

        return conn

    def init_ext(self, suite_name: str = None):
        conn = super().create_connection()
        conn.execute("CREATE EXTENSION IF NOT EXISTS vchord cascade")
        conn.close()
        self.debug_log("Extensions initialized successfully.")

    def prewarm_index(self, table_name: str):
        conn = self.create_connection()
        print("Prewarming the index...", end="", flush=True)
        try:
            conn.execute(
                f"SELECT vchordrq_prewarm('{table_name}_embedding_idx'::regclass)")
            print(" done!")
        except Exception:
            print(" failed!")
            pass
        finally:
            conn.close()

    def create_index(self, suite_name: str, table_name: str, dataset: dict):
        event, os_monitor_thread, index_monitor_thread = super().create_index(suite_name, table_name, dataset)

        workers = self.config[suite_name]["workers"]
        lists = self.config[suite_name]["lists"]
        build_threads=self.config[suite_name].get("build_threads",1) # default to 1 if not specified
        kmeans_hierarchical = self.config[suite_name]["kmeans_hierarchical"]
        kmeans_dimension = self.config[suite_name].get("kmeans_dimension", dataset["dim"])
        sampling_factor = self.config[suite_name]["samplingFactor"]
        residual_quantization = self.config[suite_name]["residual_quantization"]
        metric = dataset["metric"]
        self.debug_log(f"workers: {workers}, lists: {lists}, kmeans_hierarchical: {kmeans_hierarchical}, kmeans_dimension: {kmeans_dimension}, sampling_factor: {sampling_factor}, metric: {metric}, residual_quantization: {residual_quantization}, build_threads: {build_threads}")
        self.results[suite_name]["lists"] = lists
        self.results[suite_name]["build_threads"] = build_threads
        metric_ops = self.metric_dict[metric]
        kmeans_hierarchical_string = "kmeans_algorithm.hierarchical = {}" if kmeans_hierarchical else "kmeans_algorithm.lloyd = {}"
        residual_quantization_string = "true" if residual_quantization else "false"

        config = f"""
        residual_quantization = {residual_quantization_string}
        build.pin = 2
        """

        # external
        if self.centroids:
            # Case when centroids file is provided
            external_centroids_cfg = f"""
            [build.external]
            table = 'public.centroids'
            """
            ivf_config = "\n".join(
                [config, external_centroids_cfg])
            self.debug_log(f"external_centroids_cfg: {ivf_config}"
                f"Index build options with centroids file: metric: {metric_ops}, ivf_config: {ivf_config}")
        elif self.centroids_table:
            # Case when centroids table is provided
            external_centroids_cfg = f"""
            [build.external]
            table = '{self.centroids_table}'
            """
            ivf_config = "\n".join(
                [config, external_centroids_cfg])
            self.debug_log(
                f"Index build options with centroids table: metric: {metric_ops}, ivf_config: {ivf_config}")
        # internal
        else:
            if metric == "l2" or metric == "euclidean":
                internal_centroids_cfg = f"""
                [build.internal]
                lists = {lists}
                build_threads = {build_threads}
                spherical_centroids = false
                kmeans_dimension = {kmeans_dimension}
                sampling_factor = {sampling_factor}
                {kmeans_hierarchical_string}
                """
            elif metric == "cos" or metric == "dot" or metric == "ip":
                internal_centroids_cfg = f"""
                [build.internal]
                lists = {lists}
                build_threads = {build_threads}
                spherical_centroids = true
                kmeans_dimension = {kmeans_dimension}
                sampling_factor = {sampling_factor}
                {kmeans_hierarchical_string}
                """
            else:
                raise ValueError

            ivf_config = "\n".join(
                [config, internal_centroids_cfg])
            self.debug_log(
                f"Index build options: metric: {metric_ops}, ivf_config: {ivf_config}")

        conn = self.create_connection()

        start_time = perf_counter()
        conn.execute(f"SET max_parallel_maintenance_workers TO {workers}")
        conn.execute(f"SET max_parallel_workers TO {workers}")
        conn.execute(
            f"CREATE INDEX {table_name}_embedding_idx ON {table_name} USING vchordrq (embedding {metric_ops}) WITH (options = $${ivf_config}$$)"
        )

        self.results[suite_name]["index_build_time"] = int(round(perf_counter() - start_time))
        print(f'Index build time: {self.results[suite_name]["index_build_time"]}s')

        conn.execute("CHECKPOINT")
        conn.close()

        event.set()
        index_monitor_thread.join()
        os_monitor_thread.join()

    def sequential_bench(
            self,
            name: str,
            table_name: str,
            conn: psycopg.Connection,
            metric: str,
            top: int,
            benchmark: dict,
            dataset: dict
    ) -> tuple[list[tuple[int, float]], str]:

        conn.execute("SET jit=false")
        conn.execute(f"SET vchordrq.probes=\"{benchmark['nprob']}\"")
        conn.execute(f"SET vchordrq.epsilon={benchmark['epsilon']}")

        self.prewarm_index(table_name)

        if metric in {"l2", "euclidean"}:
            metric_ops = "<->"
        elif metric in {"cos", "angular"}:
            metric_ops = "<=>"
        elif metric in {"dot", "ip"}:
            metric_ops = "<#>"
        else:
            raise ValueError("unsupported metric type")

        self.debug_log(f"metric: {dataset['metric']}, nprob: {benchmark['nprob']}, epsilon: {benchmark['epsilon']}, metric_ops: {metric_ops}")

        return super().sequential_bench(name, table_name, conn, metric_ops, top, benchmark, dataset)

    def generate_markdown_result(self):
        self.debug_log(self.results)
        md_file = MdUtils(
            file_name="./results/benchmark_results", title="Benchmark Results",
        )

        list_of_strings = [
            "test_name",
            "dataset",
            "workers",
            "metric",
            "num_processes",
            "lists",
            "sampling_factor",
            "nprob",
            "epsilon",
            "top",
            "load_time",
            "build_threads",
            "index_build_time",
            "index_size",
            "recall",
            "qps",
            "p50_latency",
            "p99_latency",
        ]
        columns = len(list_of_strings)

        rows = 1
        for suite_name, suite in self.config.items():
            for name, benchmark in suite["benchmarks"].items():
                rows += 1
                list_of_strings += [
                    suite_name,
                    self.config[suite_name]["dataset"],
                    self.config[suite_name]["workers"],
                    self.config[suite_name]["metric"],
                    self.num_processes,
                    self.results[suite_name].get("lists", "N/A"),
                    self.config[suite_name]["samplingFactor"],
                    benchmark["nprob"],
                    benchmark["epsilon"],
                    self.config[suite_name]["top"],
                    self.results[suite_name].get("load_time", "N/A"),
                    self.results[suite_name].get("build_threads", "N/A"),
                    self.results[suite_name].get("index_build_time", "N/A"),
                    self.results[suite_name].get("index_size", "N/A"),
                    f'{self.results[suite_name][name]["recall"]:.4f}',
                    f'{self.results[suite_name][name]["qps"]:.4f}',
                    f'{self.results[suite_name][name]["p50_latency"]:.4f}',
                    f'{self.results[suite_name][name]["p99_latency"]:.4f}',
                ]

        md_file.new_table(
            columns=columns,
            rows=rows,
            text=list_of_strings,
            text_align="right",
        )

        md_file.create_md_file()


if __name__ == "__main__":
    parser = build_arg_parse()
    args = parser.parse_args()
    print(args)

    test_suite = TestSuite(
        args.suite, args.url, args.devices,
        args.chunk_size, args.skip_add_embeddings,
        args.centroids_file, args.centroids_table,
        args.skip_index_creation, args.num_processes,
        args.max_load_threads, args.debug, args.overwrite_table
    )

    test_suite.run()

    print("Test suite completed.")
