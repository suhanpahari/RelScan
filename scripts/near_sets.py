"""Relational answer sets NEAR(a, b) from evidence dumps, under the protocol in results/NOVELTY.md.

Queries per scene (fixed before any result): every ordered pair of present classes (a != b), so "both present but
not near" is a hard negative, plus, for every present class, one absent partner in each direction (a present / b
absent and a absent / b present), drawn with a per-scene seed. The answer to NEAR(a, b) is the set of GT objects of
class a whose bbox gap to some object of class b is <= 0.5 m (eval_queries.relational_gt).

The dumps do not store GT geometry, so GT objects are rebuilt from the raw scan with the SNAP grid (CenterShift +
0.02 m cells, >= 20 cells, majority label), and checked against rec["gt_classes"].

Score of object o: s(o) = p_a(o) * R_b(o), where R_b is "some object near o is b":
  max      max_{o' != o} near(o, o') p_b(o')                       (eval_relational's form)
  noisyor  1 - prod_{o' != o} (1 - near(o, o') p_b(o'))            (the probability, if the objects are independent)

  python scripts/near_sets.py --dataset scannet --dev EV/scannet_dev [--test EV/scannet_val]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402
import eval_queries as Q  # noqa: E402
import evidence as V  # noqa: E402
import sets_lab as S  # noqa: E402

SPLITS = {"s3dis": ["Area_1", "Area_5"], "scannet": ["train", "val"], "replica": ["all"]}


def scene_dir(dataset, scene):
    root = Q.DATA[dataset][0]
    for s in SPLITS[dataset]:
        d = os.path.join(paths.DOWNLOADS, "data", root, s, scene)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(scene)


def gt_boxes(dataset, scene, classes):
    """[(class, (lo, hi))] in the order of eval_queries.gt_objects on the SNAP grid."""
    d = scene_dir(dataset, scene)
    coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
    ins = np.load(os.path.join(d, "instance.npy")).reshape(-1).astype(np.int64)
    seg = np.load(os.path.join(d, f"{Q.DATA[dataset][2]}.npy")).reshape(-1).astype(np.int64)
    shift = np.array([(coord[:, 0].min() + coord[:, 0].max()) / 2, (coord[:, 1].min() + coord[:, 1].max()) / 2, coord[:, 2].min()])
    cell = np.floor((coord - shift) / 0.02).astype(np.int64)
    out = []
    for i in np.unique(ins):
        if i < 0:
            continue
        pts = ins == i
        lab = np.bincount(seg[pts][seg[pts] >= 0], minlength=len(classes)).argmax() if (seg[pts] >= 0).any() else -1
        if lab < 0 or classes[lab] in Q.SKIP[dataset]:
            continue
        if len(np.unique(cell[pts], axis=0)) >= 20:
            c = coord[pts]
            out.append((classes[lab], (c.min(0), c.max(0))))
    return out


class NearLab(S.Lab):
    def __init__(self, dataset, recs):
        super().__init__(dataset, recs)
        classes = Q.classes_for(dataset)
        self.rq, bad = [], 0
        for s, r in recs.items():
            objs = gt_boxes(dataset, s, classes)
            if [c for c, _ in objs] != list(r["gt_classes"]):
                bad += 1
                continue  # GT rebuild disagrees with the dump: leave the scene out rather than guess
            objs3 = [(c, None, b) for c, b in objs]
            present = sorted({c for c, _ in objs})
            absent = [c for c in self.classes if c not in r["seen"]]
            rng = np.random.default_rng(abs(hash(s)) % 2 ** 31)
            pairs = [(a, b) for a in present for b in present if a != b]
            if absent:
                pairs += [(a, absent[rng.integers(len(absent))]) for a in present]
                pairs += [(absent[rng.integers(len(absent))], b) for b in present]
            for a, b in pairs:
                g = Q.relational_gt(objs3, a, b, 0.5) if a in present and b in present else []
                self.rq.append((s, a, b, len(g), r["iou"][:, g].float().numpy()))
        self.bad = bad
        self.near = {s: r["near"].float().fill_diagonal_(0) for s, r in recs.items()}

    def rel_score(self, name, s, a, b, agg="max", **kw):
        key = ("rel", name, s, a, b, agg, tuple(sorted(kw.items())))
        if key not in self.cache:
            pa = torch.from_numpy(self.score(name, s, a, **kw)).float()
            pb = torch.from_numpy(self.score(name, s, b, **kw)).float()
            if name in ("cos", "sms"):
                pa, pb = pa.clamp(min=0), pb.clamp(min=0)
            nb = self.near[s] * pb[None]
            R = nb.max(1).values if agg in ("max", "min") else 1 - torch.exp(torch.log1p(-nb.clamp(0, 1 - 1e-6)).sum(1))
            self.cache[key] = (torch.minimum(pa, R) if agg.startswith("min") else pa * R).numpy()
        return self.cache[key]

    def rsets(self, name, t, kw, agg="max"):
        out = []
        for s, a, b, n_gt, io in self.rq:
            sc = self.rel_score(name, s, a, b, agg, **kw)
            out.append((s, n_gt, io, np.nonzero(sc > t)[0], float(sc.sum()) if name in S.PROB else None))
        return out

    def rap(self, name, kw, agg="max"):
        per = {}
        for s, a, b, n_gt, io in self.rq:
            if n_gt:
                per.setdefault((a, b), []).append((self.rel_score(name, s, a, b, agg, **kw), io, n_gt))
        return float(np.nanmean([np.nanmean([Q.average_precision(it, t) for t in np.arange(0.5, 0.96, 0.05)]) for it in per.values()]))


def rgrid(lab, name, kw, agg):
    if name in S.PROB:
        return [0.002, 0.005, 0.01, 0.02, 0.03] + list(np.linspace(0.05, 0.95, 19))
    v = np.concatenate([lab.rel_score(name, s, a, b, agg, **kw) for s, a, b, _, _ in lab.rq[:400]])
    v = v[v > 0]
    return list(np.quantile(v, np.linspace(0.5, 0.999, 40)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--candidates", default="", help="json list of [name, kw, agg]")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = NearLab(a.dataset, V.load(a.dev))
    iw = [0.75] if a.image else [0.0]
    baselines = [("cos", dict(feat=f), "max") for f in (["image", "dense"] if a.image else ["dense"])] + \
                [("label", dict(image_w=w), "max") for w in iw] + [("ovql", dict(image_w=w), g) for w in iw for g in ("max", "noisyor")]
    hood = dict(hood="text:clip_l:0.92", mode="excl")
    img = dict(image_w=4.0, fuse="bayes") if a.image else dict(image_w=0.0)
    cands = [tuple(c) for c in json.loads(a.candidates)] if a.candidates else [("ovql", dict(hood, **img), g) for g in ("max", "noisyor", "min", "min_noisyor")]
    if not a.candidates:  # calibrate once on dev IS objects, then compose with no relational tuning
        A, B = S.fit_platt(dev, dict(hood, **img))
        print(f"platt on dev IS: A={A:.3f} B={B:.3f}", flush=True)
        cands += [("platt", dict(hood, **img, A=round(A, 4), B=round(B, 4)), g) for g in ("max", "noisyor", "min")]
    res = dict(dataset=a.dataset, image=a.image, dev={}, test={})
    print(f"dev: {len(dev.recs)} scenes ({dev.bad} GT mismatches dropped), {len(dev.rq)} NEAR queries "
          f"({sum(q[3] == 0 for q in dev.rq)} empty)", flush=True)
    for name, kw, agg in baselines:
        t = max(rgrid(dev, name, kw, agg), key=lambda t: S.metrics(dev.rsets(name, t, kw, agg))["cgF1"])
        m = S.metrics(dev.rsets(name, t, kw, agg))
        res["dev"][json.dumps([name, kw, agg])] = dict(m, t=float(t), role="baseline")
        print(f"  B {name:6s} {json.dumps(kw):44s} {agg:8s} t={t:.3f} cgF1 {m['cgF1']:.1f} IL_MCC {m['IL_MCC']:.3f} absFP {m['absFP']:.3f}", flush=True)
    for name, kw, agg in cands:
        m = S.metrics(dev.rsets(name, 0.5, kw, agg))
        res["dev"][json.dumps([name, kw, agg])] = dict(m, t=0.5, role="candidate")
        print(f"  C {name:6s} {json.dumps(kw):44s} {agg:8s} p>0.5   cgF1 {m['cgF1']:.1f} IL_MCC {m['IL_MCC']:.3f} absFP {m['absFP']:.3f}", flush=True)
    best_b = max((k for k, v in res["dev"].items() if v["role"] == "baseline"), key=lambda k: res["dev"][k]["cgF1"])
    best_c = max((k for k, v in res["dev"].items() if v["role"] == "candidate"), key=lambda k: res["dev"][k]["cgF1"])
    print("chosen on dev: candidate", best_c, "vs baseline", best_b, flush=True)
    res["chosen"], res["bar"] = best_c, best_b
    if a.test:
        test = NearLab(a.dataset, V.load(a.test))
        print(f"test: {len(test.recs)} scenes ({test.bad} GT mismatches dropped), {len(test.rq)} NEAR queries "
              f"({sum(q[3] == 0 for q in test.rq)} empty)", flush=True)
        qs = {}
        for k, v in res["dev"].items():
            name, kw, agg = json.loads(k)
            qs[k] = test.rsets(name, v["t"], kw, agg)
            res["test"][k] = dict(S.metrics(qs[k]), t=v["t"], role=v["role"])
        for k in (best_c, best_b):
            name, kw, agg = json.loads(k)
            res["test"][k]["AP"] = test.rap(name, kw, agg)
        d, lo, hi = S.bootstrap(qs[best_c], qs[best_b])
        res["delta"] = dict(cgF1=d, lo=lo, hi=hi, AP=res["test"][best_c]["AP"] - res["test"][best_b]["AP"])
        for k in sorted(res["test"], key=lambda k: -res["test"][k]["cgF1"]):
            m = res["test"][k]
            print(f"  {m['role'][0].upper()} {k:80s} cgF1 {m['cgF1']:.1f} IL_MCC {m['IL_MCC']:.3f} pmF1 {m['pmF1']:.3f} absFP {m['absFP']:.3f}")
        print(f"TEST chosen - bar: cgF1 {d:+.1f} [{lo:+.1f}, {hi:+.1f}]  AP {res['delta']['AP'] * 100:+.1f}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
