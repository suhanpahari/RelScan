"""Score OVO-SLAM's Replica outputs (official code, run_eval.py --segment) with this repo's OVI-MAP protocol code.

OVO and OVI-MAP share the OpenNeRF Replica GT (51 reduced classes, the NICE-SLAM mesh vertices), so its per-vertex
labels (<exp>/replica/<scene>.txt, reduced ids) and instances (<exp>/instance_pred/<scene>.txt, RLE masks + reduced id +
confidence) go straight into the same mIoU/mAcc and instance AP as eval_standard.py --bench replica.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import bench  # noqa: E402
import eval_standard as E  # noqa: E402


def rle_decode(rle):
    counts = np.array(rle["counts"].split(), np.int64)
    mask = np.zeros(rle["length"], bool)
    for start, n in zip(counts[::2] - 1, counts[1::2]):
        mask[start:start + n] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="../downloads/competitors/OVO/data/output/Replica/ovo_full")
    ap.add_argument("--out", default="results/compare/ovo_replica.json")
    ap.add_argument("--scenes", nargs="*", default=[])
    a = ap.parse_args()
    labels, ids, scenes, inst = E.benchmark("replica")
    scenes = a.scenes or scenes
    valid = inst["48"]
    C = len(ids) + 1
    conf = np.zeros((C, C), np.int64)
    matched, per_scene = [], {}
    for s in scenes:
        _, verts, sem, gt, _, _ = E.load_replica(s, mesh_only=True)
        pv = np.loadtxt(os.path.join(a.exp, "replica", f"{s}.txt"), dtype=np.int64) + 1  # reduced id -> ours, -1 -> 0
        assert len(pv) == len(sem), (s, len(pv), len(sem))
        ok = sem > 0
        conf += np.bincount(sem[ok] * C + pv[ok], minlength=C * C).reshape(C, C)
        preds = []
        for line in open(os.path.join(a.exp, "instance_pred", f"{s}.txt")).read().split("\n"):
            if not line.strip():
                continue
            f, lab, p = line.split()
            mk = rle_decode(json.load(open(os.path.join(a.exp, "instance_pred", f))))
            assert len(mk) == len(gt)
            preds.append((mk, int(lab) + 1, float(p)))
        matched.append(bench.match_scene(gt, preds, valid))
        per_scene[s] = dict(n_instances=len(preds))
    tp = np.diag(conf)[1:].astype(float)
    gtc, prc = conf[1:].sum(1).astype(float), conf[:, 1:].sum(0).astype(float)
    pres = gtc > 0
    res = dict(mIoU=float((tp / np.maximum(gtc + prc - tp, 1))[pres].mean()), mAcc=float((tp / np.maximum(gtc, 1))[pres].mean()))
    apr = bench.summarize_ap(bench.evaluate(matched, valid), [dict(zip(ids, labels))[i] for i in valid])
    res.update({k: apr[k] for k in ("AP", "AP50", "AP25")})
    print(" ".join(f"{k} {100 * v:.1f}" for k, v in res.items()))
    json.dump(dict(exp=a.exp, results=res, scenes=per_scene, per_class_AP=apr["per_class_AP"]), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
