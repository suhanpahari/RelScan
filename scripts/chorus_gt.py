"""Round 4 (M1) training data: Chorus trunk features pooled over ScanNet GT instances.

Runs in env_chorus. Scenes are --meta's `scenes`: one scan per ScanNet train room, minus every room of the head's
training scenes (train[5::10][:60]) and of the dev scenes (train[::40]), so neither M1 nor the head sees a room twice.
Rooms above --max_points points are encoded on a random subset, each point taking its nearest subset point's features
(scene0497_00, 518k points, OOMs otherwise; 300k still OOMs). Per GT instance (majority segment200 label, >= 20 points, wall/floor/ceiling
kept) it stores the mean `backbone_upcast` and the mean frozen `lang` output, to --out/<scene>.pt: classes (list), mean (K, 1232), lang (K, 1152), n (K,).

  env_chorus/bin/python scripts/chorus_gt.py --meta TR/gt/meta.json --out TR/gt --shard 0 --shards 2 --gpu MIG-...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chorus_expert import scene_dir, voxelize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--checkpoint", default="chorus_pts")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_points", type=int, default=250000, help="larger rooms: encode a random subset, copy to nearest")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    torch.set_num_threads(2)
    meta = json.load(open(a.meta))
    classes, scenes = meta["classes"], meta["scenes"][a.shard::a.shards]
    import chorus
    enc = chorus.load(a.checkpoint, outputs=("lang", "backbone_upcast"))
    for n, scene in enumerate(scenes):
        dst = os.path.join(a.out, f"{scene}.pt")
        if os.path.exists(dst):
            continue
        d = scene_dir("scannet", scene)
        arrays = {k: np.load(os.path.join(d, f"{k}.npy")).astype(np.float32) for k in ("coord", "color", "normal")}
        ins = np.load(os.path.join(d, "instance.npy")).reshape(-1).astype(np.int64)
        seg = np.load(os.path.join(d, "segment200.npy")).reshape(-1).astype(np.int64)
        arrays, sub = voxelize(arrays, np.arange(len(ins)), a.max_points)
        for attempt in range(int(os.environ.get("OOM_RETRIES", 10))):
            try:
                out = enc.encode(arrays, scene_name=scene, save=False).features
                trunk = torch.as_tensor(np.asarray(out["backbone_upcast"])).float()
                lang = torch.as_tensor(np.asarray(out["lang"])).float()
                if sub is not None:
                    nn = torch.as_tensor(sub[1])
                    trunk, lang = trunk[nn], lang[nn]
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"OOM on {scene}, retry {attempt + 1} in 180 s", flush=True)
                time.sleep(180)
        else:
            raise RuntimeError(f"persistent OOM on {scene}")
        cls, mean, lmean, cnt = [], [], [], []
        for i in np.unique(ins):
            pts = ins == i
            if i < 0 or pts.sum() < 20 or not (seg[pts] >= 0).any():
                continue
            cls.append(classes[np.bincount(seg[pts][seg[pts] >= 0], minlength=len(classes)).argmax()])
            p = torch.from_numpy(pts)
            mean.append(trunk[p].mean(0)), lmean.append(lang[p].mean(0)), cnt.append(int(pts.sum()))
        tmp = dst + ".tmp"
        torch.save(dict(scene=scene, classes=cls, mean=torch.stack(mean).half(), lang=torch.stack(lmean).half(),
                        n=torch.tensor(cnt)), tmp)
        os.replace(tmp, dst)
        print(f"[{n + 1}/{len(scenes)}] {scene} instances={len(cls)}", flush=True)
        del trunk, lang
        torch.cuda.empty_cache()
    print("GT_DONE", flush=True)


if __name__ == "__main__":
    main()
