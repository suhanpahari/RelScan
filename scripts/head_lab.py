"""Direction 1 of results/NOVELTY.md round 2: a trained, class-agnostic membership head ("is stored object o a c?").

SAM 3 learns match x presence with hard negatives; here the same is done offline over the stored evidence: every
(object, query class) pair of a training scene is a row, labelled 1 iff the object matches a GT instance of c at
IoU >= 0.5, with every absent class of the scene as negatives. Features never name the class (cosines, ranks within
the scene and within the label set, OVQL probabilities, marginalized mass, scene presence, object size, containment),
so the head transfers to unseen class names (S3DIS, Replica).

  python scripts/head_lab.py --train scannet:EV/scannet_train --dev scannet:EV/scannet_dev s3dis:EV/s3dis_dev [--image]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evidence as V  # noqa: E402
import sets_lab as S  # noqa: E402

HOOD = dict(hood="text:clip_l:0.92", mode="excl")


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def rankfrac(M, col):
    """Fraction of label columns scoring above column `col`, per row (0 = c is the object's top label)."""
    return (M > M[:, col:col + 1]).float().mean(1)


class Feats:
    def __init__(self, lab, image, extra=()):
        self.lab, self.image, self.extra, self.cache = lab, image, tuple(extra), {}

    def scene(self, s):
        if s not in self.cache:
            r = self.lab.recs[s]
            d = dict(dense=r["dense"].float() / r["dense_norm"].clamp(min=1e-6)[:, None], snap=r["snap"].float())
            if self.image and "image" in r:
                d["image"] = r["image"].float() / r["image_norm"].clamp(min=1e-6)[:, None]
            c = r.get("contain")
            if c is not None:
                c = c.float().clone()
                c.fill_diagonal_(0)
            d["contain"] = c
            for key in self.extra:  # round 3: further per-object cosine matrices (N, |L|) stored in the record
                d[key] = r[key].float() if key in r else None
            self.cache = {s: d}  # one scene at a time keeps memory flat
        return self.cache[s]

    def __call__(self, s, c):
        lab, r = self.lab, self.lab.recs[s]
        k = lab.idx[c]
        d = self.scene(s)
        iw = dict(image_w=4.0, fuse="bayes") if self.image else dict(image_w=0.0)
        p_w = torch.from_numpy(lab.score("ovql", s, c, **iw)).float()
        p_m = torch.from_numpy(lab.score("ovql", s, c, **HOOD, **iw)).float()
        hood = [lab.idx[w] for w in lab.hood(c, "text:clip_l:0.92") if w != c and w in lab.idx]
        f = [p_w, p_m, torch.logit(p_w.clamp(1e-6, 1 - 1e-6)), torch.logit(p_m.clamp(1e-6, 1 - 1e-6)),
             zs(p_m), (p_m[None] > p_m[:, None]).float().mean(1),  # rank of o within the scene for c
             torch.full_like(p_m, float(p_m.max())), torch.full_like(p_m, float(p_m.topk(min(2, len(p_m))).values[-1])),
             torch.full_like(p_m, float(len(hood))), torch.log(r["n_points"].float().clamp(min=1)),
             torch.full_like(p_m, float(np.log(len(p_m))))]
        for key in ("dense", "snap", "image"):
            if key == "image" and not self.image:
                continue
            M = d.get(key)
            if M is None:  # an image-setting scene without image evidence
                f += [torch.zeros_like(p_m)] * 5
                continue
            x = M[:, k]
            hm = M[:, hood].max(1).values if hood else torch.full_like(x, -1.0)
            other = M.clone()
            other[:, [k] + hood] = -9
            f += [x, zs(x), rankfrac(M, k), x - other.max(1).values, hm]
        if self.image:
            f.append(r["image_valid"].float() if "image_valid" in r else torch.zeros_like(p_m))
        C = d["contain"]
        if C is not None:  # fragment cues: is o inside another object, and how much does that object look like c
            inside = C.max(1).values
            host = (C > 0.8).float() * p_m[None]
            f += [inside, host.max(1).values, (C.T > 0.8).float().sum(1)]
        for key in self.extra:
            f += expert_feats(d[key], k, hood, p_m)
        return torch.stack(f, 1).numpy()


def expert_feats(M, k, hood, ref):
    """The 5 per-expert features: cosine to c, scene z-score, rank among all labels, margin over non-hood labels, best
    hood cosine (zeros when the record lacks this expert)."""
    if M is None:
        return [torch.zeros_like(ref)] * 5
    x = M[:, k]
    hm = M[:, hood].max(1).values if hood else torch.full_like(x, -1.0)
    other = M.clone()
    other[:, [k] + hood] = -9
    return [x, zs(x), rankfrac(M, k), x - other.max(1).values, hm]


def groups(image, contain, extra=()):
    """Column ranges of the feature groups in Feats' layout, for the ablation (--ablate)."""
    g = dict(prob=range(0, 4), rank=range(4, 6), presence=range(6, 8), context=range(8, 11), dense=range(11, 16), snap=range(16, 21))
    n = 21
    if image:
        g["image"], n = range(21, 27), 27
    if contain:
        g["contain"], n = range(n, n + 3), n + 3
    for key in extra:
        g[key], n = range(n, n + 5), n + 5
    return g


def rows(lab, feats):
    X, y, meta = [], [], []
    for qi, (s, c, n_gt, io) in enumerate(lab.queries):
        x = feats(s, c)
        X.append(x)
        y.append((io.max(1) >= 0.5) if n_gt else np.zeros(len(x), bool))
        meta.append(np.full(len(x), qi))
    return np.concatenate(X), np.concatenate(y), np.concatenate(meta)


def query_sets(lab, P, meta, rule, post=None):
    out = []
    for qi, (s, c, n_gt, io) in enumerate(lab.queries):
        p = P[meta == qi]
        sel = S.ef1_select(p) if rule == "ef1" else np.nonzero(p > rule)[0]
        if post:
            sel = S.fragments(sel, lab.recs[s].get("contain"), post)
        out.append((s, n_gt, io, sel, float(p.sum())))
    return out


def load(spec):
    ds, d = spec.split(":", 1)
    return S.Lab(ds, V.load(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--freeze", default="", help="json {model, rule, post}: the dev choice applied to --test")
    ap.add_argument("--bar", default="", help="json [name, kw, t]: the best dev baseline (sets_lab), for the bootstrap and AP guard")
    ap.add_argument("--extra", nargs="*", default=[], help="round 3: further expert keys in the records (e.g. dense_p90 image3c_siglip2 chorus)")
    ap.add_argument("--ablate", nargs="*", default=[], help="feature groups (see groups()) zeroed at train and dev/test")
    ap.add_argument("--ref_train", nargs="*", default=[], help="round 3: train specs of a reference head (round-2 features, no --extra), "
                    "applied to --test with the same frozen rule; paired bootstrap of head - reference")
    ap.add_argument("--class_split", default="", help="round 4: json {seen, held}; held-out classes leave head training, and "
                    "dev cgF1 is also reported on seen / held / novel (not a ScanNet class) query names")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    split = json.load(open(a.class_split)) if a.class_split else None
    off = []
    Xs, ys = [], []
    for spec in a.train:
        lab = load(spec)
        if split:
            lab.queries = [q for q in lab.queries if q[1] not in set(split["held"])]
        X, y, _ = rows(lab, Feats(lab, a.image, a.extra))
        Xs.append(X), ys.append(y)
        g = groups(a.image, "contain" in next(iter(lab.recs.values())), a.extra)
        off = [i for k in a.ablate for i in g[k]]
    X, y = np.concatenate(Xs), np.concatenate(ys)
    X[:, off] = 0
    print(f"train rows {len(y)}, positives {int(y.sum())}, features {X.shape[1]}", flush=True)
    models = {"logreg": LogisticRegression(max_iter=2000, C=1.0),
              "gbm": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)}
    mu, sd = X.mean(0), X.std(0) + 1e-6
    for name, m in models.items():
        m.fit((X - mu) / sd if name == "logreg" else X, y)
    pred = lambda name, Z: models[name].predict_proba((Z - mu) / sd if name == "logreg" else Z)[:, 1]  # noqa: E731
    ref = None
    if a.ref_train:
        R = [rows(lab_, Feats(lab_, a.image, []))[:2] for lab_ in map(load, a.ref_train)]
        Xr, yr = np.concatenate([r[0] for r in R]), np.concatenate([r[1] for r in R])
        mur, sdr = Xr.mean(0), Xr.std(0) + 1e-6
        ref = (LogisticRegression(max_iter=2000, C=1.0).fit((Xr - mur) / sdr, yr), mur, sdr)
        print(f"reference train rows {len(yr)}, positives {int(yr.sum())}, features {Xr.shape[1]}", flush=True)
    res = dict(image=a.image, train=a.train, extra=a.extra, ablate=a.ablate, dev={}, test={})
    rules = [0.5, "ef1", 0.3, 0.2, 0.1]
    seen, held = (set(split["seen"]), set(split["held"])) if split else (set(), set())
    for spec in a.dev:
        lab = load(spec)
        Z, yz, meta = rows(lab, Feats(lab, a.image, a.extra))
        Z[:, off] = 0
        for name in models:
            P = pred(name, Z)
            for rule, post in [(r, None) for r in rules] + ([(r, q) for r in ("ef1", 0.2) for q in ("drop", "host")]
                                                          if "contain" in next(iter(lab.recs.values())) else []):
                qs = query_sets(lab, P, meta, rule, post)
                m = S.metrics(qs)
                res["dev"][f"{spec}|{name}|{rule}|{post}"] = m
                extra = ""
                if split and post is None:
                    for part, keep in (("seen", lambda c: c in seen), ("held", lambda c: c in held), ("novel", lambda c: c not in seen | held)):
                        sub = [q for q, (_, c, _, _) in zip(qs, lab.queries) if keep(c)]
                        if sub:
                            ms = S.metrics(sub)
                            res["dev"][f"{spec}|{name}|{rule}|{post}|{part}"] = ms
                            extra += f" | {part} {ms['cgF1']:5.1f} (n {len(sub)})"
                print(f"DEV {spec.split(':')[0]:8s} {name:6s} {str(rule):4s} {str(post):5s} cgF1 {m['cgF1']:5.1f} IL_MCC {m['IL_MCC']:.3f} "
                      f"pmF1 {m['pmF1']:.3f} absFP {m['absFP']:.3f}{extra}", flush=True)
    if a.test:
        fr = json.loads(a.freeze)
        for spec in a.test:
            lab = load(spec)
            Z, yz, meta = rows(lab, Feats(lab, a.image, a.extra))
            Z[:, off] = 0
            P = pred(fr["model"], Z)
            qs = query_sets(lab, P, meta, fr["rule"], fr.get("post"))
            m = S.metrics(qs)
            res["test"][spec] = m
            print(f"TEST {spec} {fr} cgF1 {m['cgF1']:.1f} IL_MCC {m['IL_MCC']:.3f} pmF1 {m['pmF1']:.3f} absFP {m['absFP']:.3f}", flush=True)
            if a.bar:
                bname, bkw, bt = json.loads(a.bar)
                qb = lab.sets(bname, bt, bkw)
                mb = S.metrics(qb)
                d, lo, hi = S.bootstrap(qs, qb)
                per = {}
                for qi, (s_, c, n_gt, io) in enumerate(lab.queries):
                    if n_gt:
                        per.setdefault(c, []).append((P[meta == qi], io, n_gt))
                ap_head = float(np.nanmean([np.nanmean([S.Q.average_precision(it, t) for t in np.arange(0.5, 0.96, 0.05)]) for it in per.values()]))
                ap_bar = lab.ap(bname, bkw)
                res["test_vs_bar"] = dict(bar=json.loads(a.bar), bar_metrics=mb, delta=d, lo=lo, hi=hi, AP_head=ap_head, AP_bar=ap_bar)
                print(f"TEST bar {a.bar} cgF1 {mb['cgF1']:.1f} absFP {mb['absFP']:.3f} | head - bar: cgF1 {d:+.1f} [{lo:+.1f}, {hi:+.1f}]  "
                      f"AP {100 * ap_head:.1f} vs {100 * ap_bar:.1f}", flush=True)
            if ref is not None:
                Zr, _, meta_r = rows(lab, Feats(lab, a.image, []))
                qr = query_sets(lab, ref[0].predict_proba((Zr - ref[1]) / ref[2])[:, 1], meta_r, fr["rule"], fr.get("post"))
                mr = S.metrics(qr)
                d, lo, hi = S.bootstrap(qs, qr)
                res["test_vs_ref"] = dict(ref_train=a.ref_train, ref_metrics=mr, delta=d, lo=lo, hi=hi)
                print(f"TEST ref {a.ref_train} cgF1 {mr['cgF1']:.1f} absFP {mr['absFP']:.3f} | head - ref: cgF1 {d:+.1f} [{lo:+.1f}, {hi:+.1f}]",
                      flush=True)
            res.setdefault("test_sets", {})[spec] = [(q[0], q[1], q[3].tolist()) for q in qs]
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
