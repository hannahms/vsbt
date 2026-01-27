import argparse
import psycopg
import pgvector.psycopg

from time import perf_counter
from mdutils.mdutils import MdUtils

import common


def build_arg_parse():
    parser = argparse.ArgumentParser(description="PGVector Test Suite")
    common.build_arg_parse(parser)

    # Add specific arguments for PGVector suite
    # None

    return parser


class TestSuite(common.TestSuite):

    @staticmethod
    def process_batch(args):
        import time

        # Unpack and use ef_search
        test, answer, top, metric_ops, url, table_name, ef_search = args

        conn = psycopg.connect(url)

        conn.execute(f"SET hnsw.ef_search={ef_search}")

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
        # Pass extra args for pgvector

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
            benchmark["efSearch"]
        )

    def create_connection(self):
        conn = super().create_connection()
        pgvector.psycopg.register_vector(conn)

        return conn

    def init_ext(self, suite_name: str = None):
        conn = super().create_connection()
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        conn.close()
        self.debug_log("Extensions initialized successfully.")

    def prewarm_index(self, table_name: str):
        conn = self.create_connection()
        print("Prewarming the index...", end="", flush=True)
        try:
            conn.execute(
                f"SELECT pg_prewarm('{table_name}_embedding_idx')")
            print(" done!")
        except Exception:
            print(" failed!")
            pass
        finally:
            conn.close()

    def create_index(self, suite_name: str, table_name: str, dataset: dict):
        event, os_monitor_thread, index_monitor_thread = super().create_index(suite_name, table_name, dataset)

        workers = self.config[suite_name]["workers"]
        m = self.config[suite_name]["m"]
        ef_construction = self.config[suite_name]["efConstruction"]
        metric = dataset["metric"]

        if metric == "l2" or metric == "euclidean":
            metric_func = "vector_l2_ops"
        elif metric == "cos":
            metric_func = "vector_cosine_ops"
        elif metric == "ip" or metric == "dot":
            metric_func = "vector_ip_ops"
        else:
            raise ValueError

        self.debug_log(
            f"Index config: metric_func={metric_func}, workers={workers}, m={m}, ef_construction={ef_construction}"
        )

        conn = self.create_connection()
        start_time = perf_counter()
        conn.execute(f"SET max_parallel_maintenance_workers TO {workers}")
        conn.execute(f"SET max_parallel_workers TO {workers}")
        conn.execute(
            f"CREATE INDEX {table_name}_embedding_idx ON {table_name} USING hnsw (embedding {metric_func}) WITH (m = {m}, ef_construction = {ef_construction})"
        )


        self.results[suite_name]["index_build_time"] = int(round(perf_counter() - start_time))
        print(f'Index build time: {self.results[suite_name]["index_build_time"]}s')

        conn.close()
        print("Index built successfully.")

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
    ) -> list[tuple[int, float]]:
        conn.execute(f"SET hnsw.ef_search={benchmark['efSearch']}")
        self.debug_log(f"ef_search: {benchmark['efSearch']}")

        self.prewarm_index(table_name)

        if metric in {"l2", "euclidean"}:
            metric_ops = "<->"
        elif metric in {"cos", "angular"}:
            metric_ops = "<=>"
        elif metric in {"dot", "ip"}:
            metric_ops = "<#>"
        else:
            raise ValueError("unsupported metric type")

        self.debug_log(f"Benchmark config: metric_ops={metric_ops}, dataset_metric={dataset['metric']}")

        return super().sequential_bench(name, table_name, conn, metric_ops, top, benchmark,dataset)

    def generate_markdown_result(self):
        self.debug_log(f"Results: {self.results}")
        md_file = MdUtils(
            file_name="./results/benchmark_results", title="Benchmark Results",
        )

        list_of_strings = [
            "test_name",
            "dataset",
            "workers",
            "metric",
            "query_clients",
            "m",
            "ef_construction",
            "ef_search",
            "top",
            "load_time",
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
                    self.query_clients,
                    self.config[suite_name]["m"],
                    self.config[suite_name]["efConstruction"],
                    benchmark["efSearch"],
                    self.config[suite_name]["top"],
                    self.results[suite_name].get("load_time", "N/A"),
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

    test_suite = TestSuite(
        args.suite, args.url, args.devices,
        args.chunk_size, args.skip_add_embeddings,
        args.centroids_file, args.centroids_table,
        args.skip_index_creation, args.query_clients,
        args.max_load_threads, args.debug, args.overwrite_table
    )
    test_suite.run()

    print("Test suite completed.")
