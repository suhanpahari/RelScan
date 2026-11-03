"""Error budget of the trained membership head (round 3 config) on dev, to choose what a Chorus modification must fix.

GT side, per present query c and GT instance g:
  noprop   no stored object has IoU >= 0.5 with g (proposal ceiling)
  hit      an object with IoU >= 0.5 with g is selected (p > t)
  synonym  covered, not selected, and the best-covering object gives a higher p to a class in hood(c) (near-synonym)
  other    covered, not selected, a higher p goes to a class outside hood(c)
  weak     covered, not selected, and no queried class beats c on that object (low everywhere)
FP side, per selected object that matches no GT of c at IoU >= 0.5:
  partial  IoU in [0.25, 0.5) with a GT of c (bad mask of the right object)
  dup      IoU >= 0.5 with a GT of c that another selected object already hits better
  othercls IoU >= 0.5 with a GT of another queried class (wrong name; split syn / non-syn)
  bg       none of the above (fragment / unannotated stuff)

  python head_errors.py --train scannet:EV/scannet_train --dev scannet:EV/scannet_dev [--image] [--extra ...]
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

SCRIPTS = "/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/soham/opensnap/opensnap/scripts"
sys.path.insert(0, SCRIPTS)
import head_lab as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--t", type=float, default=0.2)
    a = ap.parse_args()
    lab = H.load(a.train)
    X, y, _ = H.rows(lab, H.Feats(lab, a.image, a.extra))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(max_iter=2000, C=1.0).fit((X - mu) / sd, y)
    for spec in a.dev:
        lab = H.load(spec)
        Z, _, meta = H.rows(lab, H.Feats(lab, a.image, a.extra))
        P = m.predict_proba((Z - mu) / sd)[:, 1]
        by_scene = defaultdict(dict)  # scene -> class -> (p, io, n_gt)
        for qi, (s, c, n_gt, io) in enumerate(lab.queries):
            by_scene[s][c] = (P[meta == qi], io, n_gt)
        gt, fp, fp_abs, conf = Counter(), Counter(), Counter(), Counter()
        for s, Q in by_scene.items():
            cls = list(Q)
            Pm = np.stack([Q[c][0] for c in cls], 1)  # (N objects, queried classes)
            for j, c in enumerate(cls):
                p, io, n_gt = Q[c]
                hood = set(lab.hood(c, "text:clip_l:0.92")) - {c}
                sel = np.nonzero(p > a.t)[0]
                if n_gt:
                    for g in range(n_gt):
                        cov = io[:, g]
                        if cov.max() < 0.5:
                            gt["noprop"] += 1
                        elif (cov[sel] >= 0.5).any() if len(sel) else False:
                            gt["hit"] += 1
                        else:
                            o = int(cov.argmax())
                            better = [cls[k] for k in np.nonzero(Pm[o] > p[o])[0]]
                            if not better:
                                gt["weak"] += 1
                            else:
                                top = cls[int(Pm[o].argmax())]
                                kind = "synonym" if top in hood else "other"
                                gt[kind] += 1
                                conf[(c, top)] += 1
                hit_best = {}
                for o in sel:
                    if n_gt and io[o].max() >= 0.5:
                        g = int(io[o].argmax())
                        if g in hit_best:
                            fp["dup"] += 1
                        hit_best[g] = o
                        continue
                    C = fp if n_gt else fp_abs
                    if n_gt and io[o].max() >= 0.25:
                        C["partial"] += 1
                        continue
                    owner = [c2 for c2 in cls if c2 != c and Q[c2][2] and Q[c2][1][o].max() >= 0.5]
                    if owner:
                        C["othercls_syn" if owner[0] in hood else "othercls_non"] += 1
                    else:
                        C["bg"] += 1
        n = sum(gt.values())
        print(f"== {spec} t {a.t} extra {a.extra} image {a.image}")
        print("GT  " + "  ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in gt.most_common()))
        for name, C in (("FP present", fp), ("FP absent", fp_abs)):
            n2 = sum(C.values()) or 1
            print(f"{name}  " + "  ".join(f"{k} {v} ({100 * v / n2:.0f}%)" for k, v in C.most_common()))
        print("top confusions (query -> winning class): " + ", ".join(f"{c}->{w} {k}" for (c, w), k in conf.most_common(15)))


if __name__ == "__main__":
    main()
