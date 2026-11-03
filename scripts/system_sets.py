"""Whole systems as answer sets on the OVI-MAP Replica benchmark (8 scenes, the 48 instance classes, NICE-SLAM mesh GT).

Every scene is asked IS(c) for every instance class, present or absent; a system answers with the instances it outputs
for label c. Two operating points per system:
  native  OpenSNAP: argmax label with p > 0.5; OVO-SLAM: every instance it outputs (its own decision)
  loso    the confidence threshold maximizing cgF1 on the other 7 scenes (an upper bound, not a deployable setting)
Metrics are those of scripts/answer_sets.py (cgF1 = 100 pmF1 IL_MCC; one-to-one matching at IoU >= 0.5).

  python scripts/system_sets.py --out results/compare/system_sets_replica.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import answer_sets as A  # noqa: E402
import eval_standard as E  # noqa: E402
import loso_image as LI  # noqa: E402
import score_ovo as SO  # noqa: E402
import tune_dev as T  # noqa: E402

VOCAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opensnap", "vocab")
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "downloads")


def opensnap_preds(scene, w, mix="S", t_img=1.0):
    """The headline OpenSNAP+image Replica setting (replica_joint_loso_image grid (0, 0.6, 'S', 1.0, 0.0, 1.0)); w = 0: 3D only."""
    d = torch.load(os.path.join(R, "runs", "dump_replica_ul4", f"{scene}.pt"), weights_only=False)
    img = {k: x.float() for k, x in torch.load(os.path.join(R, "runs", "dump_replica_ul4_img", f"{scene}.pt"), weights_only=False).items()}
    base = tuple(json.load(open("results/compare/tuned.json"))["chosen"][:4])
    pm = LI.probs(d, img, base, w, mix, t_img).cpu()
    p, c = pm.max(1)
    ids = np.asarray(d["ids"])
    conf = [x for _, x in T.preds(d, pm, base[3])]
    return [(mk, int(ids[int(c[k])]), float(p[k]), conf[k]) for k, mk in enumerate(T.masks_of(d))]


def ovo_preds(exp, scene):
    out = []
    for line in open(os.path.join(exp, "instance_pred", f"{scene}.txt")).read().split("\n"):
        if line.strip():
            f, lab, p = line.split()
            out.append((SO.rle_decode(json.load(open(os.path.join(exp, "instance_pred", f)))), int(lab) + 1, None, float(p)))
    return out


def queries(gt, preds, valid):
    """[(class, n_gt, iou (P_c, n_gt), conf (P_c,), p (P_c,) or None)] for every valid class."""
    out = []
    for c in valid:
        gids = [g for g in np.unique(gt) if g and g // 1000 == c]
        P = [k for k, pr in enumerate(preds) if pr[1] == c and pr[0].sum() >= 100]  # bench.match_scene's min_region
        io = np.zeros((len(P), len(gids)))
        for j, g in enumerate(gids):
            gm = gt == g
            for i, k in enumerate(P):
                inter = (preds[k][0] & gm).sum()
                io[i, j] = inter / max(preds[k][0].sum() + gm.sum() - inter, 1)
        conf = np.array([preds[k][3] for k in P])
        p = np.array([preds[k][2] for k in P]) if P and preds[P[0]][2] is not None else None
        out.append((c, len(gids), io, conf, p))
    return out


def sets(qs, t, use_p=False):
    return [(n, io, np.nonzero(((p if p is not None else np.zeros(0)) if use_p else conf) > t)[0], None) for _, n, io, conf, p in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/compare/system_sets_replica.json")
    a = ap.parse_args()
    v = json.load(open(os.path.join(VOCAB, "replica51.json")))
    valid = [v["labels"].index(l) + 1 for l in v["instance_labels"]]
    _, _, scenes, _ = E.benchmark("replica")
    systems = {"OpenSNAP + image (so400m)": lambda s: opensnap_preds(s, 0.6),
               "OpenSNAP 3D only": lambda s: opensnap_preds(s, 0.0),
               "OVO-SLAM (default)": lambda s: ovo_preds(os.path.join(R, "competitors", "OVO", "data", "output", "Replica", "ovo_full"), s),
               "OVO-SLAM (TextRegion + PE-Core)": lambda s: ovo_preds(os.path.join(R, "competitors", "OVO_tr", "data", "output", "Replica", "ovo_textregion"), s)}
    res = {}
    for name, fn in systems.items():
        Qs = {}
        for s in scenes:
            gt = E.load_replica(s, mesh_only=True)[3]
            Qs[s] = queries(gt, fn(s), valid)
        allq = [q for s in scenes for q in Qs[s]]
        r = {}
        has_p = any(q[4] is not None for q in allq)
        native = sets(allq, 0.5, True) if has_p else sets(allq, -np.inf)
        r["native"] = A.evaluate(native)
        confs = np.concatenate([q[3] for q in allq if len(q[3])])
        grid = np.unique(np.quantile(confs, np.linspace(0, 0.99, 50)))
        held = []
        for s in scenes:
            rest = [q for x in scenes if x != s for q in Qs[x]]
            t = max(grid, key=lambda t: A.evaluate(sets(rest, t))["cgF1"])
            held += sets(Qs[s], t)
        r["loso"] = A.evaluate(held)
        r["n_queries"], r["n_absent"] = len(allq), sum(q[1] == 0 for q in allq)
        res[name] = r
        print(f"{name:34s} native cgF1 {r['native']['cgF1']:5.1f} IL_MCC {r['native']['IL_MCC']:.3f} pmF1 {r['native']['pmF1']:.3f} "
              f"absFP {r['native']['absFP']:.3f} | loso cgF1 {r['loso']['cgF1']:5.1f} absFP {r['loso']['absFP']:.3f}  "
              f"({r['n_queries']} queries, {r['n_absent']} absent)", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
