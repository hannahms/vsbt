# vector-bench

A benchmarking suite for evaluating vector database extensions (such as [pgvector](https://github.com/pgvector/pgvector), [vectorchord](https://docs.vectorchord.ai/vectorchord/getting-started/installation.html), [aidb](https://github.com/EnterpriseDB/aidb) ) on large-scale datasets. Supports flexible configuration, parallel execution, and detailed result reporting.

## Features

- Benchmarking for multiple vector database extensions
- Support for various distance metrics (L2, cosine, dot)
- Configurable test suites via YAML files
- Parallel index building and query execution
- System resource monitoring (e.g., NVMe devices)
- Results output in Markdown and CSV

## Usage

Run the benchmark tool with:

```
python3 <suite_name>_suite.py -s <suite_yaml>.yaml --url <postgres_url> [other options]
```

Example:

```
python3 pgvector_suite.py -s pgvector_suite.yaml --url postgresql://postgres@localhost:5432/postgres
```

## Arguments

| Argument                | Required | Description                                                              |
|-------------------------|----------|--------------------------------------------------------------------------|
| `-s`, `--suite`         | Yes      | Path to the YAML file describing the test suite                          |
| `--url`                 | Yes      | Database URL (e.g., `postgresql://postgres@localhost:5432/postgres`)     |
| `--devices`             | No       | Block devices to monitor (e.g., `dm-0`, `nvme0n1`)                       |
| `--chunk_size`          | No       | Chunk size for loading the dataset from the hdf5 file (default: 1000000) |
| `-c`, `--centroids-file`| No       | Path to the centroids file (VC only)                                     |
| `-ct`, `--centroids_table` | No    | Centroids table name (VC only)                                           |
| `--skip_add_embeddings` | No       | Skip adding embeddings (flag)                                            |
| `--skip_index_creation` | No       | Skip index creation step (flag)                                          |
| `--num_processes`       | No       | Number of processes for parallel benchmark (default: 1)                  |
| `--debug`               | No       | Enable debug logging (flag)                                              |

See all options with:

```
python3 pgvector_suite.py -h
```

## Test Suite YAML Configuration

The YAML file defines dataset-specific and benchmark parameters only.  
**Do not** specify extension, PostgreSQL parameters, devices, or suite name in the YAML.

Example YAML (`vectorchord_suite.yaml`):

```yaml
vc-laion-5m-8192-test-ip:
  dataset: laion-5m-test-ip
  datasetType: hdf5
  metric: dot
  nlists: 8192
  samplingFactor: 256
  batchSize: 1000000
  workers: 63
  top: 10
  benchmarks:
    20-1.0:
      nprob: 20
      epsilon: 1.0
    30-1.0:
      nprob: 30
      epsilon: 1.0
    # ... more benchmarks ...
```

- Each top-level key is a suite name.
- Each suite defines dataset, type, metric, and a set of benchmarks.


## Dependencies

- Python 3.x
- [psutil](https://pypi.org/project/psutil/)
- [nvme-cli](https://github.com/linux-nvme/nvme-cli) (for device monitoring)
- Other dependencies in `requirements.txt`

Install dependencies:

```
pip3 install -r requirements.txt
sudo apt-get install nvme-cli
```

## Results

Results are saved in Markdown format in the specified results directory (e.g., `./results/benchmark_results.md`).  
Each row includes suite and dataset info, index build time, query performance metrics, and system resource usage.

### Example (Markdown Table)

| dataset         | workers | metric | num_processes | lists | sampling_factor | nprob | epsilon | top | index_build_time | clustering_time | index_size | recall | qps   | p50_latency | p99_latency |
|-----------------|---------|--------|---------------|-------|-----------------|-------|---------|-----|------------------|-----------------|------------|--------|-------|-------------|-------------|
| laion-5m-test-ip| 63      | dot    | 1             | 8192  | 256             | 20    | 1.0     | 10  | 1200             | 300             | 2.1GB      | 0.9234 | 120.5 | 8.2ms       | 15.7ms      |
| laion-5m-test-ip| 63      | dot    | 1             | 8192  | 256             | 30    | 1.0     | 10  | 1200             | 300             | 2.1GB      | 0.9451 | 110.2 | 9.1ms       | 17.3ms      |

The actual file may contain more rows and columns depending on your configuration and benchmarks.

## Contributing

Contributions and bug reports are welcome! Please open an issue or submit a pull request.

---

For more details, see the code and example YAML files in the repository.
```