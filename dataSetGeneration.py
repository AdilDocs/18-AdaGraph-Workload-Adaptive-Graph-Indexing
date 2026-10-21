"""
Download the datasets needed for AdaGraph Phases 1-2.

Run on your own machine:   python download_datasets.py
Requires:                   pip install requests numpy

Creates ./data/ with:
  sift/sift-128-euclidean.hdf5   SIFT1M (base, query, groundtruth)    ~500 MB
  t2i/t2i_base_1M.fbin       Text-to-Image base, first 1M vectors      ~800 MB
  t2i/t2i_query_10K.fbin     Text-to-Image OOD test queries (10K)      ~8 MB
  t2i/t2i_learn_1M.fbin      Text-to-Image training/history queries    ~800 MB

Only the first N vectors are fetched from the 1-billion-scale Text-to-Image
files, using HTTP range requests, so you never download the full ~800 GB.
"""

import os
import struct
import time

import requests

DATA = "data"

SIFT_HDF5_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
T2I_ROOT = "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I"
T2I_FILES = {
    # remote file name           -> (local file name,          number of vectors)
    "base.1B.fbin":              ("t2i_base_1M.fbin",          1_000_000),
    "query.public.100K.fbin":    ("t2i_query_10K.fbin",        10_000),
    "query.learn.50M.fbin":      ("t2i_learn_1M.fbin",         1_000_000),
}


def download_sift():
    """SIFT1M from the ANN-Benchmarks HTTP mirror (HDF5: train=base, test=queries, neighbors=ground truth)."""
    out_dir = os.path.join(DATA, "sift")
    out_path = os.path.join(out_dir, "sift-128-euclidean.hdf5")
    if os.path.exists(out_path):
        print("SIFT1M already present, skipping.")
        return
    os.makedirs(out_dir, exist_ok=True)
    print("Downloading SIFT1M (ANN-Benchmarks mirror) ...")
    try:
        with requests.get(SIFT_HDF5_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"SIFT1M done ({os.path.getsize(out_path) / 1e6:.0f} MB) -> {out_path}")
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        print(f"SIFT1M download failed: {e}\n"
              f"Download it manually in your browser from:\n  {SIFT_HDF5_URL}\n"
              f"and save it as {out_path}")


PIECE = 32 * 1024 * 1024   # download in 32 MB pieces
MAX_RETRIES = 10


def _fetch_range(url, start, end):
    """Fetch bytes [start, end] with retries; returns the bytes."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
            r.raise_for_status()
            data = r.content
            if len(data) == end - start + 1:
                return data
            raise IOError(f"short read: got {len(data)} of {end - start + 1} bytes")
        except Exception as e:
            wait = min(60, 5 * attempt)
            print(f"  piece {start}-{end} failed ({e}); retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Giving up on bytes {start}-{end} after {MAX_RETRIES} retries")


def download_fbin_prefix(remote_name, local_name, n_vectors):
    """Fetch the first n_vectors of a .fbin file (int32 n, int32 d, float32 data).
    Resumable: a partial download is kept as <name>.part and continued on rerun."""
    out_dir = os.path.join(DATA, "t2i")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, local_name)
    part_path = out_path + ".part"

    url = f"{T2I_ROOT}/{remote_name}"
    total_n, dim = struct.unpack("<ii", _fetch_range(url, 0, 7))
    n = min(n_vectors, total_n)
    data_bytes = n * dim * 4
    total_bytes = 8 + data_bytes

    if os.path.exists(out_path):
        if os.path.getsize(out_path) == total_bytes:
            print(f"{local_name} already present, skipping.")
            return
        print(f"{local_name} is incomplete (from an earlier failed run); removing it.")
        os.remove(out_path)

    # Start a new .part file with the cropped header, or resume an existing one.
    if not os.path.exists(part_path):
        with open(part_path, "wb") as f:
            f.write(struct.pack("<ii", n, dim))
    done = os.path.getsize(part_path)  # bytes already on disk (header included)
    print(f"Downloading {n:,} x {dim} vectors from {remote_name} "
          f"({done / 1e6:.0f}/{total_bytes / 1e6:.0f} MB already done) ...")

    with open(part_path, "ab") as f:
        while done < total_bytes:
            # local offset `done` corresponds to the same remote offset (header is 8 bytes in both)
            end = min(done + PIECE, total_bytes) - 1
            f.write(_fetch_range(url, done, end))
            f.flush()
            done = end + 1
            print(f"  {done / 1e6:.0f}/{total_bytes / 1e6:.0f} MB")

    os.replace(part_path, out_path)
    print(f"{local_name} done ({os.path.getsize(out_path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    download_sift()
    for remote, (local, n) in T2I_FILES.items():
        download_fbin_prefix(remote, local, n)
    print("\nAll datasets downloaded to ./data")