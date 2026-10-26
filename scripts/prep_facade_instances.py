"""UnderOneFacade voxelized scenes -> pointcept folders with pseudo-instances for interactive segmentation.

The benchmark has semantic labels only; each 3D-connected component of a facade element class (window, door,
balcony, ...) at the voxel grid becomes one instance. Wall, ground, terrain, roof, interior and other stay unlabelled.
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

ELEMENTS = (1, 2, 3, 4, 5, 6, 7, 8, 12)  # window, door, balcony, molding, decoration, column, arch, stairs, blinds
A = {}


def work(f):
    a = A["args"]
    coord, color, label = torch.load(f, weights_only=False)
    label = np.asarray(label).astype(np.int64)
    ins = np.full(len(coord), -1, np.int32)
    n = 0
    for c in ELEMENTS:
        idx = np.nonzero(label == c)[0]
        if len(idx) < a.min_voxels:
            continue
        pairs = cKDTree(coord[idx]).query_pairs(a.radius, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
        k, comp = connected_components(g, directed=False)
        size = np.bincount(comp, minlength=k)
        keep = np.nonzero(size >= a.min_voxels)[0]
        remap = np.full(k, -1)
        remap[keep] = np.arange(n, n + len(keep))
        ins[idx] = remap[comp]
        n += len(keep)
    d = os.path.join(a.dst, os.path.basename(f)[:-4])
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/coord.npy", coord.astype(np.float32))
    if not np.isscalar(color):
        np.save(f"{d}/color.npy", ((np.asarray(color) + 1) * 127.5).astype(np.float32))
    np.save(f"{d}/segment.npy", np.where(label == 255, -1, label).astype(np.int16))
    np.save(f"{d}/instance.npy", ins)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--src", default=os.path.join(paths.DOWNLOADS, "data", "underonefacade_5cm"))
    ap.add_argument("--dst", default=os.path.join(paths.DOWNLOADS, "data", "pointcept", "underonefacade"))
    ap.add_argument("--radius", type=float, default=0.09)
    ap.add_argument("--min_voxels", type=int, default=100)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    a.dst = os.path.join(a.dst, a.split)
    files = sorted(glob.glob(os.path.join(a.src, a.split, "*.pth")))
    A["args"] = a
    with Pool(a.procs) as p:
        n = p.map(work, files, chunksize=1)
    print(len(files), "scenes", sum(n), "instances", flush=True)


if __name__ == "__main__":
    main()
