"""Score open-vocabulary 3D object queries as answer SETS, not rankings (input: eval_relational --absent 1 --full 1).

AP only ranks the objects of a concept that is known to be present. A query engine has to return a set, and must
return nothing for a concept that is absent from the scan. Following the SAM 3 protocol (cgF1 = pmF1 x IL_MCC),
adapted to 3D instances:
  pmF1    micro F1 of the returned objects on queries whose concept is present (one-to-one matching at IoU >= 0.5)
  IL_MCC  Matthews correlation of "the answer is non-empty" against "the concept is present", over all queries
  absFP   fraction of absent-concept queries that return at least one object
  cntMAE  |returned| vs number of GT instances; E[cnt]MAE uses sum(p) for probabilistic engines
Thresholds: OVQL is used as is (p > 0.5, no tuning). Cosine scores need a threshold, chosen leave-one-scene-out on
the other scenes (an advantage over OVQL); SMS is the standardized-maximum-similarity filter of Details Matter
(per-query z-score over the scene's objects, top-1 label per object), with its threshold chosen the same way.
"""
import argparse
import json
import os

import numpy as np


def match(ious, sel):
    """Greedy one-to-one matching of the selected objects to GT instances at IoU >= 0.5 -> (tp, fp)."""
    if not len(sel):
        return 0, 0
    m = np.asarray(ious, float).reshape(len(ious), -1)[sel]
    if m.shape[1] == 0:
        return 0, len(sel)
    used, tp = set(), 0
    for k in np.argsort(-m.max(1)):
        cand = [jj for jj in np.argsort(-m[k]) if m[k, jj] >= 0.5 and jj not in used]
        if cand:
            used.add(cand[0])
            tp += 1
    return tp, len(sel) - tp


def mcc(tp, fp, fn, tn):
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / d if d else 0.0


def evaluate(queries):
    """queries: list of (n_gt, ious (N x n_gt), selected index array, expected count or None)."""
    TP = FP = FN = 0
    c = dict(tp=0, fp=0, fn=0, tn=0)
    abs_fp, n_abs, cnt, ecnt = 0, 0, [], []
    for n_gt, ious, sel, ec in queries:
        pres, ans = n_gt > 0, len(sel) > 0
        c["tp" if pres and ans else "fn" if pres else "fp" if ans else "tn"] += 1
        if pres:
            tp, fp = match(ious, sel)
            TP, FP, FN = TP + tp, FP + fp, FN + n_gt - tp
        else:
            n_abs += 1
            abs_fp += ans
        cnt.append(abs(len(sel) - n_gt))
        if ec is not None:
            ecnt.append(abs(ec - n_gt))
    pmf1 = 2 * TP / max(2 * TP + FP + FN, 1)
    il = mcc(c["tp"], c["fp"], c["fn"], c["tn"])
    out = dict(cgF1=100 * pmf1 * il, pmF1=pmf1, IL_MCC=il, absFP=abs_fp / max(n_abs, 1), cntMAE=float(np.mean(cnt)))
    if ecnt:
        out["EcntMAE"] = float(np.mean(ecnt))
    return out


def per_scene(recs, v):
    """{scene: [(concept, n_gt, ious aligned to object index, scores aligned to object index)]} for IS queries."""
    out = {}
    for r in recs:
        rows = []
        for q in r["results"].get(v, []):
            if q["kind"] != "IS":
                continue
            n = r["n_objects"]
            s = np.zeros(n)
            io = np.zeros((n, q["n_gt"]))
            s[q["idx"]] = q["scores"]
            if q["n_gt"]:
                io[q["idx"]] = np.asarray(q["ious"]).reshape(len(q["idx"]), -1)
            rows.append((q["a"], q["n_gt"], io, s))
        out[r["scene"]] = rows
    return out


def thresholded(data, scenes, t, prob):
    return [(n_gt, io, np.nonzero(s > t)[0], float(s.sum()) if prob else None) for sc in scenes for _, n_gt, io, s in data[sc]]


def sms(data, scenes, tau):
    qs = []
    for sc in scenes:
        rows = data[sc]
        if not rows:
            continue
        L = np.stack([s for *_, s in rows], 1)  # objects x queried concepts
        z = (L - L.mean(0)) / np.maximum(L.std(0), 1e-6)
        top = L.argmax(1)
        zt = z[np.arange(len(L)), top]
        for c, (_, n_gt, io, _) in enumerate(rows):
            qs.append((n_gt, io, np.nonzero((top == c) & (zt > tau))[0], None))
    return qs


def loso(data, build, grid):
    """Leave-one-scene-out: each scene's queries use the threshold that maximizes cgF1 on the other scenes."""
    scenes, qs, chosen = sorted(data), [], []
    for sc in scenes:
        rest = [s for s in scenes if s != sc]
        t = max(grid, key=lambda g: evaluate(build(data, rest, g))["cgF1"])
        chosen.append(float(t))
        qs += build(data, [sc], t)
    return qs, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {}
    for f in a.inputs:
        recs = json.load(open(f))["records"]
        ds = os.path.basename(f).replace(".json", "")
        names = list(dict.fromkeys(v for r in recs for v in r["results"]))
        R = res.setdefault(ds, {})
        for v in names:
            data = per_scene(recs, v)
            if not any(data.values()):
                continue
            if v.startswith("ovql"):
                R[f"{v} (p>0.5)"] = evaluate(thresholded(data, sorted(data), 0.5, True))
                qs, t = loso(data, lambda d, s, g: thresholded(d, s, g, True), np.linspace(0.05, 0.95, 19))
                R[f"{v} (tuned t)"] = dict(evaluate(qs), t=float(np.median(t)))
            elif v.startswith("cosraw"):
                lo, hi = (np.percentile(np.concatenate([s for rows in data.values() for *_, s in rows]), p) for p in (50, 99.9))
                qs, t = loso(data, lambda d, s, g: thresholded(d, s, g, False), np.linspace(lo, hi, 40))
                R[f"{v} (tuned t)"] = dict(evaluate(qs), t=float(np.median(t)))
                qs, t = loso(data, sms, np.linspace(-1, 6, 29))
                R[f"SMS/{v.split('/')[1]} (tuned tau)"] = dict(evaluate(qs), t=float(np.median(t)))
            elif v.startswith("label_sql_closed"):
                R[v] = evaluate(thresholded(data, sorted(data), 0.0, False))
        n_q = sum(len(rows) for rows in per_scene(recs, names[0]).values())
        n_abs = sum(q["n_gt"] == 0 for r in recs for q in r["results"][names[0]] if q["kind"] == "IS")
        print(f"== {ds}: {len(recs)} scenes, {n_q} IS queries ({n_abs} absent)")
        for k, m in R.items():
            print(f"{k:34s} " + " ".join(f"{x} {m[x]:.3f}" if x != "cgF1" else f"cgF1 {m[x]:.1f}" for x in m))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
