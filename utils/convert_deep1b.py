import os
import struct
import shutil
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
FILES = [
    {
        "input": "base.1B.fbin",
        "output": "deep1b_base.npy",
        "dtype_str": "<f4",       # Explicit Little-Endian Float32
        "type": "vectors"
    },
    {
        "input": "query.public.10K.fbin",
        "output": "deep1b_queries.npy",
        "dtype_str": "<f4",       # Explicit Little-Endian Float32
        "type": "vectors"
    },
    {
        "input": "groundtruth.public.10K.ibin",
        "output": "deep1b_groundtruth.npy",
        "dtype_str": "<i4",       # Explicit Little-Endian Int32
        "type": "groundtruth"
    }
]

def create_npy_header_bytes(shape, dtype_str):
    """
    Manually creates a valid NPY 1.0 header byte sequence.
    Reference: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
    """
    # 1. Create the dictionary string safely
    # Example: {'descr': '<f4', 'fortran_order': False, 'shape': (1000,), }
    header_dict = f"{{'descr': '{dtype_str}', 'fortran_order': False, 'shape': {shape}, }}"

    # 2. Calculate Padding (Alignment to 64 bytes)
    # Total Header Size = Magic(6) + Version(2) + Length(2) + HeaderString(L)
    # We need (10 + L) % 64 == 0
    current_len = len(header_dict) + 1  # +1 for the required newline \n
    remainder = (10 + current_len) % 64
    padding = (64 - remainder) if remainder > 0 else 0

    # 3. Construct Final Header String
    header_str = header_dict + (" " * padding) + "\n"
    header_bytes = header_str.encode('latin1')

    # 4. Construct Prefix
    magic = b"\x93NUMPY"
    version = b"\x01\x00"  # Version 1.0
    header_len_bytes = struct.pack("<H", len(header_bytes))  # 2-byte unsigned short (Little Endian)

    return magic + version + header_len_bytes + header_bytes

def convert_manual(task):
    input_path = task["input"]
    output_path = task["output"]

    if not os.path.exists(input_path):
        print(f"Skipping {input_path} (Source not found)")
        return

    print(f"Converting {input_path} -> {output_path} (Manual Write)...")

    # 1. Read Source Header (8 bytes) to get Shape
    with open(input_path, "rb") as f_src:
        header_src = f_src.read(8)
        n_vectors, dim = struct.unpack("ii", header_src)

    print(f"   -> Shape: ({n_vectors}, {dim}) | Type: {task['dtype_str']}")

    # 2. Generate Pristine NPY Header
    npy_header = create_npy_header_bytes((n_vectors, dim), task["dtype_str"])

    # 3. Write File
    with open(output_path, "wb") as f_dst:
        # A. Write NPY Header
        f_dst.write(npy_header)

        # B. Stream Data from Source
        with open(input_path, "rb") as f_src:
            # Skip the 8-byte fbin/ibin header
            f_src.seek(8)

            # Efficient Stream Copy (100MB chunks)
            chunk_size = 100 * 1024 * 1024
            total_bytes = os.path.getsize(input_path) - 8

            with tqdm(total=total_bytes, unit='B', unit_scale=True, desc="   -> Copying") as pbar:
                while True:
                    chunk = f_src.read(chunk_size)
                    if not chunk:
                        break
                    f_dst.write(chunk)
                    pbar.update(len(chunk))

    # 4. Verification
    print(f"   -> Verifying {output_path}...")
    try:
        # This uses standard np.load which should now work perfectly
        # because we wrote a clean string header manually.
        chk = np.load(output_path, mmap_mode='r')
        print(f"   ✅ Verification Passed! Shape: {chk.shape}, Dtype: {chk.dtype}")
    except Exception as e:
        print(f"   ❌ Verification Failed! Error: {e}")

if __name__ == "__main__":
    for task in FILES:
        convert_manual(task)