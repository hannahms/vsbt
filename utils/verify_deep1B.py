import os
import struct

# Expected configurations for Deep1B
FILES = [
    {
        "name": "base.1B.fbin",
        "expected_n": 1_000_000_000,
        "expected_dim": 96,
        "dtype_size": 4  # float32 = 4 bytes
    },
    {
        "name": "query.public.10K.fbin",
        "expected_n": 10_000,
        "expected_dim": 96,
        "dtype_size": 4  # float32 = 4 bytes
    },
    {
        "name": "groundtruth.public.10K.ibin",
        "expected_n": 10_000,
        # Ground truth dimension (K) is usually 100 for Deep1B
        "expected_dim": None,
        "dtype_size": 4  # int32 = 4 bytes
    }
]

def check_file(file_info):
    fname = file_info["name"]

    if not os.path.exists(fname):
        print(f"❌ MISSING: {fname}")
        return

    actual_size = os.path.getsize(fname)

    try:
        with open(fname, "rb") as f:
            # Read 8 bytes header (2 integers)
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                print(f"❌ CORRUPT: {fname} (File too short for header)")
                return

            # Unpack header: num_vectors (int32), dimension (int32)
            num_vectors, dim = struct.unpack('ii', header_bytes)

            print(f"Checking {fname}...")
            print(f"  -> Header says: N={num_vectors:,}, D={dim}")

            # Check logic
            if file_info["expected_n"] and num_vectors != file_info["expected_n"]:
                print(f"  ⚠️ WARNING: Expected {file_info['expected_n']} vectors, header says {num_vectors}")

            if file_info["expected_dim"] and dim != file_info["expected_dim"]:
                 print(f"  ⚠️ WARNING: Expected dim {file_info['expected_dim']}, header says {dim}")

            # Calculate precise expected size
            # Size = Header (8 bytes) + (N * D * 4 bytes)
            expected_size = 8 + (num_vectors * dim * file_info["dtype_size"])

            print(f"  -> Expected Size: {expected_size:,} bytes")
            print(f"  -> Actual Size:   {actual_size:,} bytes")

            if actual_size == expected_size:
                print(f"  ✅ INTEGRITY PASS: File size matches header perfectly.")
            elif actual_size < expected_size:
                diff = expected_size - actual_size
                print(f"  ❌ TRUNCATED: File is missing {diff:,} bytes!")
            else:
                diff = actual_size - expected_size
                print(f"  ⚠️ OVERSIZED: File has {diff:,} extra bytes (garbage at end?).")

    except Exception as e:
        print(f"❌ ERROR reading {fname}: {e}")
    print("-" * 40)

if __name__ == "__main__":
    print("Verifying Deep1B Source Files...\n" + "-"*40)
    for f in FILES:
        check_file(f)