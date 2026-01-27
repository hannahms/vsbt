# Usage

```
$ sudo python3 benchmark/bench.py -h
usage: bench.py [-h] [-e {pgvector,vectorchord}] [-m {l2,cos,dot}] -n NAME -i INPUT --url URL -d DIM [-w WORKERS] [--chunks CHUNKS] [--results_dir RESULTS_DIR] --devices DEVICES
                [DEVICES ...] [--top {10,100}]

Index Benchmarking Tool

options:
  -h, --help            show this help message and exit
  -e {pgvector,vectorchord}, --extension {pgvector,vectorchord}
                        Extension to use
  -m {l2,cos,dot}, --metric {l2,cos,dot}
                        Distance metric
  -n NAME, --name NAME  Dataset name, like: sift
  -i INPUT, --input INPUT
                        Input filepath
  --url URL             url, like `postgresql://postgres:123@localhost:5432/postgres`
  -d DIM, --dim DIM     Dimension
  -w WORKERS, --workers WORKERS
                        Workers to build index
  --chunks CHUNKS       chunks for in-memory mode. If OOM, increase it
  --results_dir RESULTS_DIR
                        Folder to store the results
  --devices DEVICES [DEVICES ...]
                        Block devices to be monitored
  --top {10,100}        Top K for recall calculation
```

# Dependency

```
# Need nvme cli
sudo apt-get install nvme-cli

# Install Python dependency
pip3 install -r requirements.txt
```

# Example

```
sudo python3 benchmark/bench.py -e pgvector -n train -i laion-1m-test-ip.hdf5 -m l2 -d 768 --url postgresql://postgres:123@localhost:5432/postgres --results_dir ./results --devices dm-0
```