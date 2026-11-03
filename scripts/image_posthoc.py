"""Round 3 of results/NOVELTY.md: image-evidence variants added to v3 dumps from the stored proposal masks.

Variants (cosines with the encoder's text(w), raw class names), stored as image_<enc>_<variant> (N, |L|):
  ms   the current multi-scale crops (views.object_embeddings), i.e. an encoder swap with the crops held fixed
  lm   SceneSplat's crop with background + masked crop without background, summed (views.object_embeddings_3crop);
       the full three-crop fusion is not used: on scene0000_00 its masked crop alone is near chance at ScanNet's
       320x240 frames and the fusion loses to the plain crops (NOVELTY.md round 3, crop diagnostic)
ScanNet and Replica only (S3DIS has no posed frames here).

  python scripts/image_posthoc.py --dataset scannet --dumps EV/scannet_dev \
      --jobs google/siglip2-so400m-patch16-512:ms,lm google/siglip-so400m-patch14-384:lm
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import paths  # noqa: E402
from opensnap.views import ImageExpert, PosedFrames, ScanNetFrames, object_embeddings, object_embeddings_3crop  # noqa: E402
import eval_queries as Q  # noqa: E402

NICE = os.path.join(paths.DOWNLOADS, "replica_nice", "Replica")
SCANNET_2D = os.path.join(paths.DOWNLOADS, "data", "scannet_2d")


def raw_masks(rec):
    inv, off, pts = rec["inverse"].long().numpy(), rec["obj_offsets"].tolist(), rec["obj_points"].long().numpy()
    pm = np.zeros((len(off) - 1, len(inv)), bool)
    vm = np.zeros(int(inv.max()) + 1, bool)
    for o in range(len(off) - 1):
        vm[:] = False
        vm[pts[off[o]:off[o + 1]]] = True
        pm[o] = vm[inv]
    return pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["scannet", "replica"])
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--jobs", nargs="+", default=["google/siglip2-so400m-patch16-512:ms,lm"], help="encoder:variants")
    ap.add_argument("--frame_step", type=int, default=10)
    ap.add_argument("--reverse", action="store_true", help="walk the records backwards: a second process can share a directory")
    a = ap.parse_args()
    torch.set_num_threads(2)
    jobs = []
    for j in a.jobs:
        name, vs = j.rsplit(":", 1)
        short = "siglip2" if "siglip2" in name else "so400m" if "so400m" in name else name.split("/")[-1]
        jobs.append((short, ImageExpert(name, cache_dir=os.path.join(paths.DOWNLOADS, "hf", "hub")), vs.split(",")))
    root = Q.DATA[a.dataset][0]
    splits = ["train", "val"] if a.dataset == "scannet" else Q.DATA[a.dataset][1]  # dumps include train scenes
    intr = json.load(open(os.path.join(NICE, "cam_params.json")))["camera"] if a.dataset == "replica" else None
    files = sorted(glob.glob(os.path.join(a.dumps, "*.pt")), reverse=a.reverse)
    for n, f in enumerate(files):
        rec = torch.load(f, weights_only=False)
        if all(f"image_{short}_{v}" in rec for short, _, vs in jobs for v in vs):
            continue
        scene = rec["scene"]
        d = next(os.path.join(paths.DOWNLOADS, "data", root, s, scene) for s in splits
                 if os.path.isdir(os.path.join(paths.DOWNLOADS, "data", root, s, scene)))
        coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
        frames = PosedFrames(os.path.join(NICE, scene), intr, step=a.frame_step) if a.dataset == "replica" \
            else ScanNetFrames(os.path.join(SCANNET_2D, scene))
        pm = raw_masks(rec)
        for short, ex, vs in jobs:
            T = F.normalize(ex.text(rec["labels"]).float(), dim=-1)
            if "ms" in vs:
                e, _ = object_embeddings(frames, ex, coord, pm)
                rec[f"image_{short}_ms"] = (e.float() @ T.T).half().cpu()
            if "lm" in vs:
                parts, _ = object_embeddings_3crop(frames, ex, coord, pm, parts=True)
                rec[f"image_{short}_lm"] = (parts["local_plus_masked"].float() @ T.T).half().cpu()
        tmp = f + ".tmp"
        torch.save(rec, tmp)
        os.replace(tmp, f)
        print(f"[{n + 1}/{len(files)}] {scene} objs={rec['n_objects']}", flush=True)
    print("IMAGE_POSTHOC_DONE", flush=True)


if __name__ == "__main__":
    main()
