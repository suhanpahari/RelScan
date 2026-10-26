"""Comparative search benchmark on UnderOneFacade: concept (IS) and relational (IS AND NEAR) queries, instance AP.

Ground truth = pseudo-instances (3D-connected components of facade-element labels, >= 100 voxels). Every system answers
the same queries with a ranked list of 3D objects, scored at point level with eval_queries' AP:
  snap_native / opensnap / opensnap_refine   OpenSNAP's OVQL engine over its stored object relation (as eval_queries)
  openscene_cluster   OpenScene 3D point labels (argmax over the facade classes) grouped into connected components
  openshape           OpenShape (3D shape-text) class probabilities on the same segmenter proposals
  gsam / sam_siglip   render-then-lift: 2D masks lifted through the z-buffer and merged across views into 3D instances
Baselines answer NEAR by composition: an IS(a) object within `near` metres (bbox gap) of an IS(b) object, scored by the
product of the two scores. Baseline instances come from facade_openshape.py / facade_render_lift.py --inst_out.
"""
import argparse
import copy
import dataclasses
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402
from opensnap.data import LABELSETS  # noqa: E402
from eval_queries import VARIANTS, average_precision, gap  # noqa: E402

LABELS = LABELSETS["underonefacade"]
STUFF = {"wall", "ground surface", "terrain", "roof", "interior", "other"}  # no instances in the pseudo-GT
REGION = lambda s: "Germany" if s.startswith("DEBY") else ("UK" if s.startswith("Lenton") else "Singapore")


def gt_objects(ins, seg, min_points=100):
    out = []
    for i in np.unique(ins[ins >= 0]):
        idx = np.nonzero(ins == i)[0]
        if len(idx) >= min_points:
            out.append((LABELS[np.bincount(seg[idx][seg[idx] >= 0], minlength=len(LABELS)).argmax()], idx))
    return out


def box(coord, idx):
    return coord[idx].min(0), coord[idx].max(0)


def components(coord, idx, radius=0.09, min_points=100):
    if len(idx) < min_points:
        return []
    pairs = cKDTree(coord[idx]).query_pairs(radius, output_type="ndarray")
    k, comp = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx),) * 2),
                                   directed=False)
    return [idx[comp == c] for c in range(k) if (comp == c).sum() >= min_points]


def compose(objs_a, objs_b, coord, near):
    """NEAR by composition: (points, score) of a-objects within `near` of some b-object."""
    boxes_b = [(box(coord, p), s) for p, s in objs_b]
    out = []
    for p, s in objs_a:
        ba = box(coord, p)
        best = max((sb for bb, sb in boxes_b if gap(ba, bb) <= near and not np.array_equal(bb[0], ba[0])), default=0.0)
        if best > 0:
            out.append((p, s * best))
    return out


def load_instances(path, kind):
    d = np.load(path)
    splits = np.split(d["points"], np.cumsum(d["sizes"])[:-1]) if len(d["sizes"]) else []
    if kind == "probs":  # every mask can answer every class, ranked by its probability
        return lambda a: [(p, float(d["probs"][i, a])) for i, p in enumerate(splits)]
    return lambda a: [(p, float(s)) for p, l, s in zip(splits, d["labels"], d["scores"]) if l == a]


def score(objs_pred, gt_idx, gt, n):
    """(scores, ious[P, G]) at point level."""
    ious = np.zeros((len(objs_pred), len(gt_idx)))
    if objs_pred and gt_idx:
        gid = np.full(n, -1)
        for j, i in enumerate(gt_idx):
            gid[gt[i][1]] = j
        size = np.array([len(gt[i][1]) for i in gt_idx])
        for k, (p, _) in enumerate(objs_pred):
            g = gid[p]
            inter = np.bincount(g[g >= 0], minlength=len(gt_idx))
            ious[k] = inter / (len(p) + size - inter)
    return [s for _, s in objs_pred], ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/nuscenes_openseg.pth.tar")
    ap.add_argument("--near", type=float, default=0.5)
    ap.add_argument("--rel_per_scene", type=int, default=5)
    ap.add_argument("--proposals", default="fps", choices=["fps", "c2f"])
    ap.add_argument("--inst_root", default="../downloads/runs/facade_instances")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated scene names")
    ap.add_argument("--out", default="results/facade_search.json")
    args = ap.parse_args()
    from opensnap.core import Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import materialize
    from opensnap.query.engine import Engine
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    clip_snap = TextEncoder("ViT-B/32")
    dense = OpenSceneModel(args.openscene_ckpt, "openseg", voxel_size=0.05)
    m = OpenSNAP(SnapModel(args.snap_ckpt, domain="Outdoor"), clip_snap, dense, TextEncoder(dense.clip_name), Config(),
                 anchors=snap_train_vocab())
    files = sorted(glob.glob(os.path.join(paths.DOWNLOADS, "data", "pointcept", "underonefacade", args.split, "*")))
    files = [f for f in files if not args.only or os.path.basename(f) in args.only.split(",")]
    files = files[:args.limit] if args.limit else files
    baselines = {name: (os.path.join(args.inst_root, name), kind) for name, kind in
                 (("openshape", "probs"), ("gsam", "labels"), ("sam_siglip", "labels"))}
    recs = []
    for f in files:
        name = os.path.basename(f)
        coord = np.load(os.path.join(f, "coord.npy")).astype(np.float32)
        ins = np.load(os.path.join(f, "instance.npy")).astype(np.int64)
        seg = np.load(os.path.join(f, "segment.npy")).astype(np.int64)
        gt = gt_objects(ins, seg)
        if not gt:
            continue
        gtb = [box(coord, p) for _, p in gt]
        present = sorted({c for c, _ in gt})
        rel = lambda a, b: [k for k, (c, _) in enumerate(gt) if c == a and any(
            cb == b and kk != k and gap(gtb[k], gtb[kk]) <= args.near for kk, (cb, _) in enumerate(gt))]
        queries = [("IS", a, None) for a in present]
        rng = np.random.default_rng(abs(hash(name)) % 2 ** 31)
        pairs = [(a, b) for a in present for b in present if a != b and rel(a, b)]
        queries += [("NEAR", *pairs[k]) for k in rng.permutation(len(pairs))[:args.rel_per_scene]]
        gt_for = lambda q: [k for k, (c, _) in enumerate(gt) if c == q[1]] if q[0] == "IS" else rel(q[1], q[2])

        np.random.seed(abs(hash(name)) % 2 ** 31)
        torch.manual_seed(0)
        t0 = time.time()
        m.encode(coord)
        base = materialize(m, m.propose_c2f() if args.proposals == "c2f" else None)
        build = time.time() - t0
        inv = m.scene["inverse"].cpu().numpy()
        nvox = len(m.scene["vcoord"])

        def to_points(vox):
            vm = np.zeros(nvox, bool)
            vm[vox.cpu().numpy() if torch.is_tensor(vox) else vox] = True
            return np.nonzero(vm[inv])[0]
        rec = dict(scene=name, region=REGION(name), n_gt=len(gt), build_s=build, results={})
        for vname, v in VARIANTS.items():
            table = copy.deepcopy(base)
            if not v["dense"]:
                table.pooled = torch.zeros(len(table), 0, device=table.token.device)
            m.cfg = dataclasses.replace(Config(), relay_weight=v["relay"])
            eng = Engine(m, table, refine=v["refine"], refine_below=0.3, dedup=0.0)
            out = []
            for q in queries:
                text = f"SELECT WHERE IS('{q[1]}')" if q[0] == "IS" else \
                    f"SELECT WHERE IS('{q[1]}') AND NEAR(IS('{q[2]}'), {args.near})"
                res, st = eng.run(text)
                objs = [(to_points(table.points[i]), s) for i, s in res]
                s, iou = score(objs, gt_for(q), gt, len(coord))
                out.append(dict(kind=q[0], a=q[1], b=q[2], scores=s, ious=iou.tolist(), n_gt=len(gt_for(q)), ms=st["ms"]))
            rec["results"][vname] = out

        # OpenScene 3D points -> connected components (dense branch of the same model, per point)
        t0 = time.time()
        p = m.dense_probs(LABELS)[torch.from_numpy(inv).cuda()].cpu().numpy()
        lab = p.argmax(1)
        oc = {a: [(c, float(p[c, a].mean())) for c in components(coord, np.nonzero(lab == a)[0])] if LABELS[a] not in STUFF else []
              for a in range(len(LABELS))}
        sources = {"openscene_cluster": (lambda a: oc[a], time.time() - t0)}
        for bname, (root, kind) in baselines.items():
            path = os.path.join(root, name + ".npz")
            if os.path.exists(path):
                sources[bname] = (load_instances(path, kind), None)
        for bname, (get, sec) in sources.items():
            out = []
            for q in queries:
                t1 = time.time()
                a = LABELS.index(q[1])
                objs = get(a) if q[0] == "IS" else compose(get(a), get(LABELS.index(q[2])), coord, args.near)
                ms = 1000 * (time.time() - t1)
                s, iou = score(objs, gt_for(q), gt, len(coord))
                out.append(dict(kind=q[0], a=q[1], b=q[2], scores=s, ious=iou.tolist(), n_gt=len(gt_for(q)), ms=ms))
            rec["results"][bname] = out
        recs.append(rec)
        print(f"[{len(recs)}/{len(files)}] {name} gt={len(gt)} queries={len(queries)} objs={len(base)} "
              f"methods={list(rec['results'])}", flush=True)
        del base, table, eng, p, oc, sources
        m.scene = None
        torch.cuda.empty_cache()

    res = {}
    methods = sorted({k for r in recs for k in r["results"]})
    for v in methods:
        for kind in ("IS", "NEAR"):
            for region in (None, "Germany", "UK", "Singapore"):
                per = {}
                for r in recs:
                    if v not in r["results"] or (region and r["region"] != region):
                        continue
                    for q in r["results"][v]:
                        if q["kind"] == kind:
                            per.setdefault(q["a"] if kind == "IS" else (q["a"], q["b"]), []).append((q["scores"], q["ious"], q["n_gt"]))
                if not per:
                    continue
                o = {}
                for mname, thrs in (("AP", np.arange(0.5, 0.96, 0.05)), ("AP50", [0.5]), ("AP25", [0.25])):
                    o[mname] = float(np.nanmean([np.nanmean([average_precision(it, t) for t in thrs]) for it in per.values()]))
                if region is None:
                    o["per_class_AP25"] = {str(c): float(average_precision(it, 0.25)) for c, it in per.items()} if kind == "IS" else {}
                    o["ms"] = float(np.mean([q["ms"] for r in recs if v in r["results"] for q in r["results"][v] if q["kind"] == kind]))
                    o["scenes"] = sum(v in r["results"] for r in recs)
                res[f"{v}/{kind}" + (f"/{region}" if region else "")] = o
    res["systems"] = dict(build_s=float(np.mean([r["build_s"] for r in recs])), gt_objects=int(sum(r["n_gt"] for r in recs)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(args), results=res, records=recs), open(args.out, "w"))
    for k, o in res.items():
        if "/" in k and k.count("/") == 1:
            print(f"{k:32s} AP {100 * o['AP']:5.1f}  AP50 {100 * o['AP50']:5.1f}  AP25 {100 * o['AP25']:5.1f}  {o['ms']:.1f} ms")


if __name__ == "__main__":
    main()
