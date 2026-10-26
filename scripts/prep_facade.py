"""UnderOneFacade (TLS/MLS facades, UK/Germany/Singapore) -> voxelized semseg .pth (OpenScene format) + per-voxel label histograms.

Each voxel keeps the count of every LoFG3 label inside it (<name>.hist.npy), so eval_semseg scores predictions on the
full-resolution points (every point takes its voxel's prediction). Colour (mean per voxel) is kept for the UK/Singapore
.las files only, for the RGB-trained click baselines; the German .xyz files store the label colour in their rgb columns,
so their colour is zero. OpenSNAP itself never reads colour.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

NUM = 15  # LoFG3 classes; raw ids 1..15, 0 = VOID
REGIONS = {"DEBY": "Germany", "KoonSeng": "Singapore", "Geylang": "Singapore", "JooChait": "Singapore", "Temple": "Singapore",
           "SouthBridge": "Singapore"}
A = {}


def region(name):
    return next((r for k, r in REGIONS.items() if name.startswith(k)), "UK")


def read(path):
    if path.endswith(".las"):
        import laspy
        las = laspy.read(path)
        rgb = np.stack([las.red, las.green, las.blue], 1).astype(np.float64)
        return np.stack([las.x, las.y, las.z], 1), np.asarray(las.classification, np.int64), rgb / (257.0 if rgb.max() > 255 else 1.0)
    import pandas as pd
    df = pd.read_csv(path, sep=" ", skiprows=1, header=None, usecols=[0, 1, 2, 3], dtype=np.float64, engine="c")
    arr = df.to_numpy()
    return arr[:, :3], arr[:, 3].astype(np.int64), np.zeros((len(arr), 3))


def work(path):
    a = A["args"]
    name = os.path.splitext(os.path.basename(path))[0]
    xyz, raw, rgb = read(path)
    ok = np.isfinite(xyz).all(1)
    xyz, raw, rgb = xyz[ok], raw[ok], rgb[ok]
    lab = np.where(raw >= 1, np.minimum(raw, NUM) - 1, NUM)  # NUM = void column; raw 16 (Singapore) -> other, as the authors' loader
    xyz = xyz - np.r_[np.median(xyz[:, :2], 0), xyz[:, 2].min()]
    grid = a.grid
    while True:  # coarsen very large facades so the segmenter fits in GPU memory
        g = np.floor(xyz / grid).astype(np.int64)
        g -= g.min(0)
        span = g.max(0) + 1
        _, inv, cnt = np.unique((g[:, 0] * span[1] + g[:, 1]) * span[2] + g[:, 2], return_inverse=True, return_counts=True)
        if len(cnt) <= a.max_voxels:
            break
        grid *= 1.25
    inv = inv.reshape(-1)
    V = len(cnt)
    coord = np.stack([np.bincount(inv, xyz[:, i], V) for i in range(3)], 1) / cnt[:, None]
    color = np.stack([np.bincount(inv, rgb[:, i], V) for i in range(3)], 1) / cnt[:, None]
    hist = np.bincount(inv * (NUM + 1) + lab, minlength=V * (NUM + 1)).reshape(V, NUM + 1).astype(np.int32)
    label = np.where(hist[:, :NUM].sum(1) > 0, hist[:, :NUM].argmax(1), 255).astype(np.uint8)
    torch.save((coord.astype(np.float32), (color / 127.5 - 1).astype(np.float32), label), os.path.join(a.dst, f"{name}.pth"))
    np.save(os.path.join(a.dst, f"{name}.hist.npy"), hist[:, :NUM])
    return dict(scene=name, region=region(name), points=int(len(xyz)), voxels=int(V), grid=grid, void=int(hist[:, NUM].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(paths.DOWNLOADS, "data", "underonefacade"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--grid", type=float, default=0.05)
    ap.add_argument("--dst", default=os.path.join(paths.DOWNLOADS, "data", "underonefacade_5cm"))
    ap.add_argument("--max_voxels", type=int, default=2_500_000)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    a.dst = os.path.join(a.dst, a.split)
    os.makedirs(a.dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, a.split, "*.las")) + glob.glob(os.path.join(a.src, a.split, "*.xyz")))
    print(len(files), "files", flush=True)
    A["args"] = a
    with Pool(a.procs) as p:
        meta = p.map(work, files, chunksize=1)
    for m in meta:
        print(m, flush=True)
    json.dump(meta, open(os.path.join(a.dst, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
