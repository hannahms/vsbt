import argparse
from time import perf_counter
import re
from operator import truediv

import psycopg
import pgvector.psycopg

from mdutils.mdutils import MdUtils

import common


def build_arg_parse():
    parser = argparse.ArgumentParser(description="AIDB Test Suite")
    common.build_arg_parse(parser)

    # Add specific arguments for AIDB suite
    # None

    return parser


class TestSuite(common.TestSuite):
    @staticmethod
    def process_batch(args):

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
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgpu")
        conn.close()
        self.debug_log("Extensions initialized successfully.")

    def prewarm_index(self, table_name: str):
        conn = self.create_connection()
        print("Prewarming the index...", end="", flush=True)
        try:
            conn.execute(
                f"SELECT vchordrq_prewarm('{table_name}_pgpu_ext'::regclass)")
            print(" done!")
        except Exception:
            print(" failed!")
            pass
        finally:
            conn.close()

    def make_handler(self, suite_name: str):
        # Updated pattern to match: "💾 Training complete (123.45s). Building VectorChord Index..."
        pattern = re.compile(
            r"Training complete\s*\((.*?)\)\.\s*Building VectorChord Index",
            re.IGNORECASE
        )

        def handler(n):
            # n is psycopg.Notice
            print(n.message_primary)
            m = pattern.search(n.message_primary)
            if m:
                self.results[suite_name]["clustering_time"] = m.group(1)

        return handler

    def create_index(self, suite_name: str, table_name: str, dataset: dict = None):
        event, os_monitor_thread, index_monitor_thread = super(
        ).create_index(suite_name, table_name, dataset)

        workers = self.config[suite_name]["workers"]
        lists = self.config[suite_name]["lists"]
        sampling_factor = self.config[suite_name]["samplingFactor"]
        residual_quantization = self.config[suite_name]["residual_quantization"]
        batch_size = self.config[suite_name]["batchSize"]
        distance = self.config[suite_name]["metric"]
        spherical_bool = "true" if distance in ("dot", "cos", "l2") else "false"
        kmeans_nredo = self.config[suite_name].get("kmeans_n_redo", 3)
        kmeans_n_iter = self.config[suite_name].get("kmeans_n_iter", 30)
        random_sampling = self.config[suite_name].get("random_sampling", "true")


        # "ip" is more commonly used for "inner product"; this is what pgpu accepts right now
        distance = "ip" if distance == "dot" else distance
        # TODO: Consider setting n_redo and n_iter higher for hierarchical indexing
        self.debug_log(
            f"workers: {workers}, lists: {lists}, sampling_factor: {sampling_factor}, batch_size: {batch_size} distance: {distance}, spherical_bool: {spherical_bool}, residual_quantization: {residual_quantization}")

        self.results[suite_name]["lists"] = lists

        conn = self.create_connection()
        conn.add_notice_handler(self.make_handler(suite_name))
        start_time = perf_counter()

        conn.execute(f"SET max_parallel_maintenance_workers TO {workers}")
        conn.execute(f"SET max_parallel_workers TO {workers}")

        conn.execute(
            f'''
            select pgpu.create_vector_index_on_gpu(
              table_name => 'public.{table_name}',
              column_name => 'embedding',
              lists => ARRAY{lists},
              sampling_factor => {sampling_factor},
              batch_size => {batch_size},
              kmeans_nredo => {kmeans_nredo},
              kmeans_iterations=> {kmeans_n_iter},
              distance_operator=> '{distance}',
              spherical_centroids => '{spherical_bool}',
              residual_quantization => {residual_quantization},
              random_sampling => {random_sampling}
            );
            '''
        )

        self.results[suite_name]["index_build_time"] = int(round(perf_counter() - start_time))
        print(f'Index build time: {self.results[suite_name]["index_build_time"]}s')


        conn.execute("CHECKPOINT")
        conn.close()
        print(f"Index built successfully.")

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

        if metric in {"l2", "euclidean"}:
            metric_ops = "<->"
        elif metric in {"cos", "angular"}:
            metric_ops = "<=>"
        elif metric in {"dot", "ip"}:
            metric_ops = "<#>"
        else:
            raise ValueError("unsupported metric type")

        self.debug_log(
            f"nprob: {benchmark['nprob']}, epsilon: {benchmark['epsilon']}, metric: {metric}, metric_ops: {metric_ops}")

        self.prewarm_index(table_name)

        return super().sequential_bench(name, table_name, conn, metric_ops, top, benchmark, dataset)

    def generate_markdown_result(self):
        print(self.results)
        md_file = MdUtils(
            file_name="./results/benchmark_results", title="Benchmark Results",
        )

        list_of_strings = [
            "test_name",
            "dataset",
            "workers",
            "metric",
            "random_sampling",
            "lists",
            "sampling_factor",
            "nprob",
            "epsilon",
            "top",
            "load_time",
            "index_build_time",
            "clustering_time",
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
                    self.config[suite_name].get("random_sampling", "N/A"),
                    self.results[suite_name].get("lists", "N/A"),
                    self.config[suite_name]["samplingFactor"],
                    benchmark["nprob"],
                    benchmark["epsilon"],
                    self.config[suite_name]["top"],
                    self.results[suite_name].get("load_time", "N/A"),
                    self.results[suite_name].get("index_build_time", "N/A"),
                    self.results[suite_name].get("clustering_time", "N/A"),
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
