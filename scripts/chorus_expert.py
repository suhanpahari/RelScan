"""Round 3 of results/NOVELTY.md: Chorus (arXiv 2512.17817) as a fourth per-object expert, added to v3 evidence dumps.

Runs in its own env (env_chorus: torch 2.5.1 / spconv / flash-attn, downloads/competitors/Chorus installed as a package).
For every record of --dumps it encodes the scene's raw points (coord, color, normal) with the point-cloud checkpoint
`chorus_pts` (pretrained on ScanNet++ v2 only), takes the SigLIP2-aligned `lang` head, and pools it over exactly the
stored proposal masks (obj_points / obj_offsets / inverse written by dump_evidence.py --masks 1):
  chorus      cos(mean-pooled lang feature, SigLIP2 text(w))                        (N, |L|)
  chorus_p90  90th percentile over the object's points of cos(point lang feature, text(w))
Text: google/siglip2-so400m-patch16-512 (the Chorus language teacher), raw class names as in views.ImageExpert.
--voxel_input (used for S3DIS, whose rooms are 0.4-3.8M raw points and OOM the shared slice): encode one point per
dump voxel (mean coord/color, normalized mean normal); rooms above --max_points voxels are encoded on a random
subset and the features copied to every voxel from its nearest encoded voxel. Pooling is then over voxels.

  env_chorus/bin/python scripts/chorus_expert.py --dataset scannet --dumps EV/scannet_dev --gpu MIG-...
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import paths  # noqa: E402

SPLITS = {"s3dis": ["Area_1", "Area_5"], "scannet": ["train", "val"], "replica": ["all"]}
ROOT = {"s3dis": "pointcept/s3dis", "scannet": "pointcept/scannet", "replica": "pointcept/replica"}


def scene_dir(dataset, scene):
    for s in SPLITS[dataset]:
        d = os.path.join(paths.DOWNLOADS, "data", ROOT[dataset], s, scene)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(scene)


@torch.no_grad()
def text_bank(labels, name, cache_dir):
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(name, cache_dir=cache_dir).eval().half().cuda()
    proc = AutoProcessor.from_pretrained(name, cache_dir=cache_dir)
    t = proc.tokenizer(list(labels), padding="max_length", max_length=64, return_tensors="pt").to("cuda")
    out = model.get_text_features(**t)
    out = out if torch.is_tensor(out) else out.pooler_output
    del model
    torch.cuda.empty_cache()
    return F.normalize(out.float(), dim=-1)


def encode(enc, arrays, scene):
    lang = torch.as_tensor(np.asarray(enc.encode(arrays, scene_name=scene, save=False).features["lang"])).float().cuda()
    return F.normalize(lang, dim=-1)


def voxelize(arrays, inv, max_points, seed=0):
    """One point per dump voxel; above max_points, a random subset plus each voxel's nearest subset index."""
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n)[:, None].astype(np.float32)
    out = {k: np.stack([np.bincount(inv, v[:, j], minlength=n) for j in range(3)], 1).astype(np.float32) / cnt
           for k, v in arrays.items()}
    out["normal"] /= np.maximum(np.linalg.norm(out["normal"], axis=1, keepdims=True), 1e-6)
    if n <= max_points:
        return out, None
    from scipy.spatial import cKDTree
    keep = np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))
    nn = cKDTree(out["coord"][keep]).query(out["coord"], workers=2)[1]
    return {k: v[keep] for k, v in out.items()}, (np.arange(n), nn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPLITS))
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--checkpoint", default="chorus_pts")
    ap.add_argument("--text", default="google/siglip2-so400m-patch16-512")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--voxel_input", type=int, default=0)
    ap.add_argument("--max_points", type=int, default=250000)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    torch.set_num_threads(2)
    import chorus
    enc = chorus.load(a.checkpoint, outputs=("lang",))
    T, labels = None, None
    files = sorted(glob.glob(os.path.join(a.dumps, "*.pt")))
    for n, f in enumerate(files):
        rec = torch.load(f, weights_only=False)
        if "chorus" in rec:
            continue
        if T is None or rec["labels"] != labels:
            labels = rec["labels"]
            T = text_bank(labels, a.text, os.path.join(paths.DOWNLOADS, "hf", "hub"))
        d = scene_dir(a.dataset, rec["scene"])
        arrays = {k: np.load(os.path.join(d, f"{k}.npy")).astype(np.float32) for k in ("coord", "color", "normal")}
        inv = rec["inverse"].long().cuda()
        sub = None
        if a.voxel_input:
            arrays, sub = voxelize(arrays, rec["inverse"].long().numpy(), a.max_points)
            inv = torch.arange(len(arrays["coord"]) if sub is None else len(sub[0]), device="cuda")
        for attempt in range(int(os.environ.get("OOM_RETRIES", 10))):
            try:
                lang = encode(enc, arrays, rec["scene"])
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"OOM on {rec['scene']}, retry {attempt + 1} in 180 s", flush=True)
                time.sleep(180)
        else:
            raise RuntimeError(f"persistent OOM on {rec['scene']}")
        if sub is not None:
            lang = lang[torch.as_tensor(sub[1], device="cuda")]
        pts, off = rec["obj_points"].long().cuda(), rec["obj_offsets"].tolist()
        n_vox = int(inv.max()) + 1
        mean, p90 = [], []
        for o in range(len(off) - 1):
            vm = torch.zeros(n_vox, dtype=torch.bool, device="cuda")
            vm[pts[off[o]:off[o + 1]]] = True
            x = lang[vm[inv]]
            if not len(x):
                mean.append(torch.zeros(len(labels), device="cuda"))
                p90.append(torch.full((len(labels),), -1.0, device="cuda"))
                continue
            mean.append(F.normalize(x.mean(0), dim=-1) @ T.T)
            S = x @ T.T
            p90.append(S.kthvalue(max(1, int(np.ceil(0.9 * len(S)))), 0).values)
        rec["chorus"] = torch.stack(mean).half().cpu()
        rec["chorus_p90"] = torch.stack(p90).half().cpu()
        tmp = f + ".tmp"
        torch.save(rec, tmp)
        os.replace(tmp, f)
        print(f"[{n + 1}/{len(files)}] {rec['scene']} objs={rec['n_objects']} points={len(lang)}", flush=True)
    print("CHORUS_DONE", flush=True)


if __name__ == "__main__":
    main()
