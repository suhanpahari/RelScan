"""Answer-set lab: every candidate query semantics and every baseline, scored offline from evidence dumps under the
protocol in results/NOVELTY.md (choices on dev, untuned p > 0.5 for probabilistic methods, dev-tuned thresholds for
baselines, paired scene bootstrap against the best baseline, AP guard on present concepts).

  python scripts/sets_lab.py --dataset s3dis --dev EV/s3dis_dev --test EV/s3dis_a5 [--image]

A method is f(scene evidence, query class) -> score per object; its answer set is {o : score > threshold}.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import answer_sets as A  # noqa: E402
import eval_queries as Q  # noqa: E402
import evidence as V  # noqa: E402

torch.set_num_threads(2)  # the box is shared; default threading oversubscribes it ~100x
EV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "downloads", "runs", "evidence")


class Lab:
    def __init__(self, dataset, recs):
        self.ds, self.recs = dataset, recs
        txt = torch.load(os.path.join(EV, "text", f"{dataset}.pt"), weights_only=False)
        self.L = txt["labels"]
        assert all(r["labels"] == self.L for r in recs.values())
        self.idx = {w: k for k, w in enumerate(self.L)}
        self.sim = {k: v.float() for k, v in txt["sim"].items()}
        self.sig = (txt["siglip_scale"], txt["siglip_bias"])
        self.wn = json.load(open(os.path.join(EV, "text", "wordnet.json")))
        self.vocab = self.L[:234]
        self.classes = [c for c in Q.classes_for(dataset) if c not in Q.SKIP[dataset]]
        self.up = {}  # ancestors of c: labels h whose WordNet subtree contains c
        for h, below in self.wn["hyp"].items():
            for c in below:
                if c != h:
                    self.up.setdefault(c, set()).add(h)
        self.cache = {}
        self.queries = []  # (scene, class, n_gt, iou (N, n_gt))
        for s, r in recs.items():
            for c in self.classes:
                gcols = [j for j, g in enumerate(r["gt_classes"]) if g == c]
                if not gcols and c in r["seen"]:
                    continue  # annotated points but no instance large enough: ambiguous, not asked
                self.queries.append((s, c, len(gcols), r["iou"][:, gcols].float().numpy()))

    # ---- neighbourhoods N(c) over label columns -------------------------------------------------------------------
    def hood(self, c, kind):
        if kind == "word":
            return {c}
        if kind.startswith("text:"):  # text:<space>:<radius>
            _, space, d = kind.split(":")
            row = self.sim[space][self.idx[c]]
            return {self.L[k] for k in torch.nonzero(row >= float(d)).flatten().tolist()} | {c}
        if kind in ("wn_syn", "wn_hyp"):
            return set(self.wn[kind[3:]].get(c, [c])) | {c}
        if kind.startswith("wn_hyp+text:"):
            return self.hood(c, "wn_hyp") | self.hood(c, "text:" + kind.split(":", 1)[1])
        raise ValueError(kind)

    def cols(self, c, exclude=()):
        return [self.idx[c]] + [self.idx[b] for b in self.vocab if b != c and b not in exclude]

    # ---- methods --------------------------------------------------------------------------------------------------
    def score(self, name, s, c, **kw):
        key = (name, s, c, tuple(sorted(kw.items())))
        if key not in self.cache:
            self.cache[key] = getattr(self, "m_" + name)(self.recs[s], c, **kw).numpy()
        return self.cache[key]

    def m_ovql(self, r, c, hood="word", mode="sum", image_w=0.0, image_tau=0.02, fuse="geo"):
        """OVQL over [c] + B234. mode 'sum': p = mass of N(c); 'excl': drop N(c) \\ {c} from the contrast set;
        'hier': also drop c's WordNet ancestors ("chair" is not an alternative to "office chair", only unspecific)."""
        N = self.hood(c, hood)
        if mode == "hier":
            N = N | self.up.get(c, set())
        D = V.dist if fuse == "geo" else V.dist_bayes
        if mode in ("excl", "hier"):
            return D(r, self.cols(c, exclude=N), image_w, image_tau)[:, 0]
        cols = self.cols(c)
        P = D(r, cols, image_w, image_tau)
        inN = torch.tensor([self.L[k] in N for k in cols])
        return P[:, inN].sum(1)

    def m_presence(self, r, c, gate="siglip_max", combine="mul", **kw):
        """SAM 3-style factorization: score(o) = presence(c, scene) x p(o is c | c present).
        gate 'siglip_max': max over objects of SigLIP's own binary sigmoid (does any crop show c?);
        'noisy_or': 1 - prod_o (1 - p(o)); 'max': max_o p(o). combine 'mul' (product) or 'hard' (p if presence > 0.5)."""
        p = self.m_ovql(r, c, **kw)
        if gate == "siglip_max":
            g = self.m_siglip_sig(r, c).max() if "image" in r else torch.tensor(1.0)
        elif gate == "noisy_or":
            g = 1 - (1 - p.clamp(0, 1 - 1e-6)).log().sum().exp()
        else:
            g = p.max()
        return p * g if combine == "mul" else torch.where(g > 0.5, p, torch.zeros_like(p))

    def m_platt(self, r, c, A=1.0, B=0.0, **kw):
        """Platt-calibrated OVQL: sigmoid(A logit(p) + B), with (A, B) fitted once on dev IS objects (fit_platt)."""
        p = self.m_ovql(r, c, **kw).clamp(1e-6, 1 - 1e-6)
        return torch.sigmoid(A * torch.logit(p) + B)

    def m_siglip_sig(self, r, c):
        """SigLIP's own calibrated binary probability: sigmoid(scale * cos + bias) (no vocabulary competition)."""
        if "image" not in r:
            return torch.zeros(r["n_objects"])
        p = torch.sigmoid(self.sig[0] * V.cosine(r, self.idx[c]) + self.sig[1])
        return torch.where(r["image_valid"], p, torch.zeros_like(p))

    def m_fuse_sig(self, r, c, hood="text:clip_l:0.92", w=0.5):
        """Binary product of experts: marginalized 3D membership and SigLIP's sigmoid, in logit space."""
        p3 = self.m_ovql(r, c, hood=hood).clamp(1e-6, 1 - 1e-6)
        if "image" not in r:
            return p3
        z = (1 - w) * torch.logit(p3) + w * (self.sig[0] * V.cosine(r, self.idx[c]) + self.sig[1])
        return torch.where(r["image_valid"], torch.sigmoid(z), p3)

    def m_cos(self, r, c, feat="image"):
        if feat not in ("image", "dense") and feat not in r:
            return torch.full((r["n_objects"],), -1.0)
        if feat == "image" and "image" not in r:
            return torch.full((r["n_objects"],), -1.0)
        return V.cosine(r, self.idx[c], feat)

    def m_sms(self, r, c, feat="image"):
        """Details Matter: per-query z-score over the scene's objects; each object keeps its top-1 class; returns z."""
        if feat == "image" and "image" not in r:
            return torch.full((r["n_objects"],), -9.0)
        Lm = torch.stack([V.cosine(r, self.idx[q], feat) for q in self.classes], 1)
        z = (Lm - Lm.mean(0)) / Lm.std(0).clamp(min=1e-6)
        top = Lm.argmax(1)
        k = self.classes.index(c)
        return torch.where(top == k, z[:, k], torch.full_like(z[:, k], -9.0))

    def m_label(self, r, c, image_w=0.0, image_tau=0.02):
        """Closed-set label-then-filter: argmax of OVQL's distribution over the dataset classes; score = its probability."""
        cols = [self.idx[q] for q in self.classes]
        P = V.dist(r, cols, image_w, image_tau)
        p, top = P.max(1)
        return torch.where(top == self.classes.index(c), p, torch.zeros_like(p))

    # ---- evaluation -----------------------------------------------------------------------------------------------
    def sets(self, name, t, kw, scenes=None, post=None):
        out = []
        for s, c, n_gt, io in self.queries:
            if scenes is not None and s not in scenes:
                continue
            sc = self.score(name, s, c, **kw)
            sel = ef1_select(sc) if t == "ef1" else np.nonzero(sc > t)[0]
            if post:
                sel = fragments(sel, self.recs[s].get("contain"), post)
            out.append((s, n_gt, io, sel, float(sc.sum()) if name in PROB else None))
        return out

    def ap(self, name, kw):
        per = {}
        for s, c, n_gt, io in self.queries:
            if n_gt:
                per.setdefault(c, []).append((self.score(name, s, c, **kw), io, n_gt))
        return float(np.nanmean([np.nanmean([Q.average_precision(it, t) for t in np.arange(0.5, 0.96, 0.05)]) for it in per.values()]))


PROB = {"ovql", "platt", "presence", "siglip_sig", "fuse_sig", "label"}


def fragments(sel, contain, mode, tau=0.8):
    """Direction 3: proposals that lie inside another proposal (|o ∩ o'| / |o| > tau) are fragments.
    'drop': remove a selected fragment when its host is also selected; 'host': answer with the host instead."""
    if contain is None or len(sel) == 0:
        return sel
    C = contain.float().numpy().copy()
    np.fill_diagonal(C, 0)
    keep = set(int(o) for o in sel)
    for o in sorted(keep, key=lambda o: C[o].max()):
        host = int(C[o].argmax())
        if C[o, host] <= tau:
            continue
        if mode == "drop" and host in keep:
            keep.discard(o)
        elif mode == "host":
            keep.discard(o)
            keep.add(host)
    return np.array(sorted(keep), int)


def ef1_select(p, n=256, seed=0):
    """The answer set that maximizes expected F1 under independent Bernoulli(p) memberships (calibrated p).
    The candidates are the top-k sets (optimal for EF1 under independence); F1(empty, empty) = 1, so the empty answer
    scores P(nothing is c) and presence is decided by the same rule, without a threshold. Monte Carlo, fixed seed."""
    p = np.asarray(p, np.float64)
    if not len(p):
        return np.zeros(0, int)
    order = np.argsort(-p)
    ps = p[order][:64]  # sets larger than 64 never win at these probabilities; bounds the cost
    y = np.random.default_rng(seed).random((n, len(ps))) < ps
    rest = (np.random.default_rng(seed + 1).random((n, len(p) - len(ps))) < p[order][64:]).sum(1) if len(p) > 64 else 0
    tp = np.cumsum(y, 1)  # (n, k)
    Y = y.sum(1) + rest
    k = np.arange(1, len(ps) + 1)
    f1 = 2 * tp / (k[None] + Y[:, None])
    e = np.concatenate([[np.mean(Y == 0)], f1.mean(0)])
    return order[:int(e.argmax())]


def fit_platt(lab, kw):
    """(A, B) of sigmoid(A logit(p) + B) by logistic NLL on dev IS objects; label = IoU >= 0.5 with a GT of the query class."""
    z, y = [], []
    for s, c, n_gt, io in lab.queries:
        z.append(torch.logit(torch.from_numpy(lab.score("ovql", s, c, **kw)).float().clamp(1e-6, 1 - 1e-6)))
        y.append(torch.from_numpy((io.max(1) >= 0.5) if n_gt else np.zeros(len(z[-1]), bool)).float())
    z, y = torch.cat(z), torch.cat(y)
    w = torch.zeros(2, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits((1 + w[0]) * z + w[1], y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(1 + w[0]), float(w[1])


def metrics(qs):
    return A.evaluate([(n_gt, io, sel, ec) for _, n_gt, io, sel, ec in qs])


def bootstrap(qa, qb, n=1000, seed=0):
    """Paired scene bootstrap of cgF1(a) - cgF1(b); returns (delta, lo, hi)."""
    scenes = sorted({q[0] for q in qa})
    by = lambda qs: {s: [q for q in qs if q[0] == s] for s in scenes}  # noqa: E731
    A_, B_ = by(qa), by(qb)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        pick = rng.choice(scenes, len(scenes))
        d.append(metrics([q for s in pick for q in A_[s]])["cgF1"] - metrics([q for s in pick for q in B_[s]])["cgF1"])
    return metrics(qa)["cgF1"] - metrics(qb)["cgF1"], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def tune(lab, name, kw, grid):
    return max(grid, key=lambda t: metrics(lab.sets(name, t, kw))["cgF1"])


def grid_for(lab, name, kw):
    if name in PROB:
        return list(np.linspace(0.05, 0.95, 19))
    v = np.concatenate([lab.score(name, s, c, **kw) for s, c, _, _ in lab.queries[:400]])
    v = v[v > -5]
    return list(np.quantile(v, np.linspace(0.5, 0.999, 40)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--image", action="store_true", help="the image setting (SigLIP evidence available)")
    ap.add_argument("--candidates", default="", help="json list of [name, kw] to try on dev (default: built-in list)")
    ap.add_argument("--extra", nargs="*", default=[], help="round 3: further expert keys, added as cos/sms baselines")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = Lab(a.dataset, V.load(a.dev))
    iw = [0.5, 0.75] if a.image else [0.0]
    baselines = [("cos", dict(feat=f)) for f in (["image", "dense"] if a.image else ["dense"])] + \
                [("sms", dict(feat=f)) for f in (["image", "dense"] if a.image else ["dense"])] + \
                [("label", dict(image_w=w)) for w in iw] + [("ovql", dict(image_w=w)) for w in iw] + \
                [(m, dict(feat=f)) for f in a.extra for m in ("cos", "sms")]
    cands = json.loads(a.candidates) if a.candidates else \
        [("ovql", dict(hood=h, mode=m, image_w=w)) for h in ("text:clip_l:0.95", "text:clip_l:0.92", "text:clip_l:0.9", "text:siglip:0.9",
                                                            "text:siglip:0.85", "wn_syn", "wn_hyp", "wn_hyp+text:clip_l:0.92")
         for m in ("sum", "excl", "hier") for w in iw] + \
        ([("siglip_sig", {})] + [("fuse_sig", dict(hood=h, w=w)) for h in ("text:clip_l:0.92", "wn_hyp") for w in (0.3, 0.5, 0.7)]
         if a.image else [])
    res = dict(dataset=a.dataset, image=a.image, dev={}, test={})
    print(f"dev: {len(dev.recs)} scenes, {len(dev.queries)} queries ({sum(q[2] == 0 for q in dev.queries)} absent)", flush=True)
    for name, kw in baselines:
        t = tune(dev, name, kw, grid_for(dev, name, kw))
        m = metrics(dev.sets(name, t, kw))
        res["dev"][json.dumps([name, kw])] = dict(m, t=float(t), role="baseline")
        print(f"  B {name:10s} {json.dumps(kw):40s} t={t:.3f}  cgF1 {m['cgF1']:.1f} IL_MCC {m['IL_MCC']:.3f} absFP {m['absFP']:.3f}", flush=True)
    for name, kw in cands:
        m = metrics(dev.sets(name, 0.5, kw))
        res["dev"][json.dumps([name, kw])] = dict(m, t=0.5, role="candidate")
        print(f"  C {name:10s} {json.dumps(kw):40s} p>0.5    cgF1 {m['cgF1']:.1f} IL_MCC {m['IL_MCC']:.3f} absFP {m['absFP']:.3f}", flush=True)
    best_b = max((k for k, v in res["dev"].items() if v["role"] == "baseline"), key=lambda k: res["dev"][k]["cgF1"])
    best_c = max((k for k, v in res["dev"].items() if v["role"] == "candidate"), key=lambda k: res["dev"][k]["cgF1"], default=None)
    if best_c is None:  # baselines only (e.g. the bar for a trained head)
        print("best baseline on dev:", best_b, flush=True)
        res["bar"] = best_b
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
        return
    print("chosen on dev: candidate", best_c, "vs baseline", best_b, flush=True)
    res["chosen"], res["bar"] = best_c, best_b
    if a.test:
        test = Lab(a.dataset, V.load(a.test))
        print(f"test: {len(test.recs)} scenes, {len(test.queries)} queries ({sum(q[2] == 0 for q in test.queries)} absent)", flush=True)
        qs = {}
        for k, v in res["dev"].items():
            name, kw = json.loads(k)
            qs[k] = test.sets(name, v["t"], kw)
            res["test"][k] = dict(metrics(qs[k]), t=v["t"], role=v["role"])
        for k in (best_c, best_b):
            name, kw = json.loads(k)
            res["test"][k]["AP"] = test.ap(name, kw)
        d, lo, hi = bootstrap(qs[best_c], qs[best_b])
        res["delta"] = dict(cgF1=d, lo=lo, hi=hi, AP=res["test"][best_c]["AP"] - res["test"][best_b]["AP"])
        for k in sorted(res["test"], key=lambda k: -res["test"][k]["cgF1"]):
            m = res["test"][k]
            print(f"  {m['role'][0].upper()} {k:72s} cgF1 {m['cgF1']:.1f} IL_MCC {m['IL_MCC']:.3f} pmF1 {m['pmF1']:.3f} absFP {m['absFP']:.3f}")
        print(f"TEST chosen - bar: cgF1 {d:+.1f} [{lo:+.1f}, {hi:+.1f}]  AP {res['delta']['AP'] * 100:+.1f}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
