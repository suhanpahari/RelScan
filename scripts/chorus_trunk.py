"""Round 4 step 0 of results/NOVELTY.md: Chorus trunk features pooled over the stored proposal masks, for a modified
(set-trained) language head (M1) and proposal grouping (M2).

Runs in env_chorus. For every record of --dumps it encodes the scene as chorus_expert.py does (raw points, or one
point per dump voxel with --voxel_input), takes `backbone_upcast` (the 1232-d input of Chorus' teacher projectors)
and writes, per stored object, to --out/<scene>.pt (the records themselves are not touched):
  mean    (N, 1232) mean trunk feature over the object's points
  lang    (N, 1152) mean of Chorus' own per-point lang output (the frozen projector), for sanity checks
  n       (N,)      points per object
--out/projector_lang.pt holds the frozen lang projector's state_dict, to initialize M1 from.

  env_chorus/bin/python scripts/chorus_trunk.py --dataset scannet --dumps EV/scannet_train --out TR/scannet_train --gpu MIG-...
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chorus_expert import SPLITS, scene_dir, voxelize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPLITS))
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="chorus_pts")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--voxel_input", type=int, default=0)
    ap.add_argument("--max_points", type=int, default=250000)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    torch.set_num_threads(2)
    os.makedirs(a.out, exist_ok=True)
    import chorus
    enc = chorus.load(a.checkpoint, outputs=("lang", "backbone_upcast"))
    proj = os.path.join(a.out, "projector_lang.pt")
    if not os.path.exists(proj):
        torch.save({k: v.cpu() for k, v in enc.model.projectors["lang"].state_dict().items()}, proj)
    files = sorted(glob.glob(os.path.join(a.dumps, "*.pt")))
    for n, f in enumerate(files):
        dst = os.path.join(a.out, os.path.basename(f))
        if os.path.exists(dst):
            continue
        rec = torch.load(f, weights_only=False)
        d = scene_dir(a.dataset, rec["scene"])
        arrays = {k: np.load(os.path.join(d, f"{k}.npy")).astype(np.float32) for k in ("coord", "color", "normal")}
        inv = rec["inverse"].long().cuda()
        sub = None
        if a.voxel_input:
            arrays, sub = voxelize(arrays, rec["inverse"].long().numpy(), a.max_points)
            inv = torch.arange(len(arrays["coord"]) if sub is None else len(sub[0]), device="cuda")
        for attempt in range(int(os.environ.get("OOM_RETRIES", 10))):
            try:
                out = enc.encode(arrays, scene_name=rec["scene"], save=False).features
                trunk = torch.as_tensor(np.asarray(out["backbone_upcast"])).float().cuda()
                lang = torch.as_tensor(np.asarray(out["lang"])).float().cuda()
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"OOM on {rec['scene']}, retry {attempt + 1} in 180 s", flush=True)
                time.sleep(180)
        else:
            raise RuntimeError(f"persistent OOM on {rec['scene']}")
        if sub is not None:
            nn = torch.as_tensor(sub[1], device="cuda")
            trunk, lang = trunk[nn], lang[nn]
        pts, off = rec["obj_points"].long().cuda(), rec["obj_offsets"].tolist()
        n_vox = int(inv.max()) + 1
        mean, lmean, cnt = [], [], []
        for o in range(len(off) - 1):
            vm = torch.zeros(n_vox, dtype=torch.bool, device="cuda")
            vm[pts[off[o]:off[o + 1]]] = True
            sel = vm[inv]
            k = int(sel.sum())
            cnt.append(k)
            mean.append(trunk[sel].mean(0) if k else torch.zeros(trunk.shape[1], device="cuda"))
            lmean.append(lang[sel].mean(0) if k else torch.zeros(lang.shape[1], device="cuda"))
        tmp = dst + ".tmp"
        torch.save(dict(scene=rec["scene"], mean=torch.stack(mean).half().cpu(), lang=torch.stack(lmean).half().cpu(),
                        n=torch.tensor(cnt)), tmp)
        os.replace(tmp, dst)
        print(f"[{n + 1}/{len(files)}] {rec['scene']} objs={len(cnt)} trunk={tuple(trunk.shape)}", flush=True)
        del trunk, lang
        torch.cuda.empty_cache()
    print("TRUNK_DONE", flush=True)


if __name__ == "__main__":
    main()
