import os
import random
from time import perf_counter
import argparse
from pathlib import Path
from tqdm import tqdm
from numpy import linalg as LA

from fastkmeans import FastKMeans
import numpy as np
import h5py
from multiprocessing import Pool, cpu_count
import itertools

DEFAULT_LISTS = 160000
N_ITER = 10
SEED = 42
MAX_POINTS_PER_CLUSTER = 256


def build_arg_parse():
    parser = argparse.ArgumentParser(description="Train K-means centroids")
    parser.add_argument("-o", "--output", help="output filepath",
                        required=False, default='centroids.npy')
    parser.add_argument("-t", "--total", type=int,
                        required=False, default=400_000_000)
    parser.add_argument(
        "--lists",
        help="Number of centroids",
        type=int,
        required=False,
        default=DEFAULT_LISTS,
    )
    parser.add_argument(
        "--niter", help="number of iterations", type=int, default=N_ITER
    )
    parser.add_argument(
        "-m", "--metric", choices=["l2", "cos", "dot"], default="cos")
    parser.add_argument(
        "-g", "--gpu", help="enable GPU for KMeans", action="store_true"
    )
    parser.add_argument("-i", "--input", help="input filepath")
    return parser


def sample_vectors(iterator, sample_size, total_size, chunk_size=10000000):
    probability = sample_size / total_size

    result = []
    chunk_count = 0
    file_paths = []

    with tqdm(total=total_size, desc="Processing vectors", mininterval=1) as pbar:
        for vector in iterator:
            if random.random() < probability:
                result.append(vector)
                if len(result) >= sample_size:
                    break

                if len(result) >= chunk_size:
                    chunk_file = f"sampled_chunk_{chunk_count}.npy"
                    np.save(chunk_file, np.stack(result))
                    file_paths.append(chunk_file)
                    result = []
                    chunk_count += 1

            pbar.update(1)

    if result:
        chunk_file = f"sampled_chunk_{chunk_count}.npy"
        np.save(chunk_file, np.stack(result))
        file_paths.append(chunk_file)

    return file_paths


def _slice_chunk(args: tuple[int, str, np.ndarray]):
    k, file_path, chunk, start_idx = args
    dataset = h5py.File(Path(file_path), "r")
    data = dataset["train"]
    start, end = min(chunk), max(chunk)
    indexes = [c - start for c in chunk]
    source = data[start: end + 1]
    select = source[indexes]
    delta, dim = select.shape

    output = np.memmap("index.mmap", dtype=np.float32,
                       mode="r+", shape=(k, dim))
    output[start_idx: start_idx + delta, :] = select
    output.flush()


def reservoir_sampling_np(data, file_path, k: int, chunks: int):
    """Reservoir sampling in memory by numpy."""
    index = np.random.permutation(len(data))[:k]
    indices = np.sort(index)
    num_processes = cpu_count()
    # Split indices into chunks for parallel processing
    index_chunks = np.array_split(indices, chunks)
    _, dim = data.shape
    np.memmap("index.mmap", dtype=np.float32, mode="w+", shape=(k, dim))
    # Create arguments for each chunk
    start_idx_acu = [0]
    start_idx_acu.extend(
        list(itertools.accumulate([len(c) for c in index_chunks[:-1]]))
    )
    chunk_args = [
        (k, file_path, chunk, start_idx_acu[i]) for i, chunk in enumerate(index_chunks)
    ]
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        list(pool.map(_slice_chunk, chunk_args))


def kmeans_cluster(
    data,
    file_path,
    dim,
    k,
    niter,
    metric,
    gpu=False,
):
    # dim = 512
    # sample_vectors(data_iter, MAX_POINTS_PER_CLUSTER * args.lists, size)
    reservoir_sampling_np(
        data, file_path, MAX_POINTS_PER_CLUSTER * args.lists, 10
    )
    train = np.array(
        np.memmap(
            "index.mmap",
            dtype=np.float32,
            mode="r",
            shape=(MAX_POINTS_PER_CLUSTER * k, dim),
        )
    )

    # files = sorted([f for f in os.listdir(
    #     '.') if f.startswith('sampled_chunk')])
    # train = []
    # for file in tqdm(files):
    #     chunk = np.load(file)
    #     train.append(chunk)

    train = np.vstack(train).reshape(-1, dim)
    print(train.shape)

    if metric == "cos":
        train = train / LA.norm(train, axis=1, keepdims=True)

    # TODO: FastKMeans is lack of spherical_centroid support now, see https://github.com/AnswerDotAI/fastkmeans/issues/5
    kmeans = FastKMeans(
        d=dim, k=k, gpu=gpu, verbose=True, niter=niter, seed=SEED, use_triton=True
    )
    kmeans.train(train)
    return kmeans.centroids


def read_npy_files(file_pattern):
    files = sorted([f for f in os.listdir('data') if f.endswith(
        '.npy') and f.startswith(file_pattern)])

    for file in files:
        data = np.load(os.path.join('data', file))
        for row in data:
            yield row.reshape(1, -1)
            del row


if __name__ == "__main__":
    parser = build_arg_parse()
    args = parser.parse_args()
    print(args)

    # file_pattern = 'img_emb'
    # iterator = read_npy_files(file_pattern)
    dataset = h5py.File(Path(args.input), "r")
    n, dim = dataset["train"].shape

    start_time = perf_counter()
    centroids = kmeans_cluster(
        dataset["train"],
        args.input,
        dim,
        args.lists,
        args.niter,
        args.metric,
        args.gpu
    )
    print(
        f"K-means (k=({args.lists})): {perf_counter() - start_time:.2f}s"
    )

    np.save(Path(args.output), centroids, allow_pickle=False)
