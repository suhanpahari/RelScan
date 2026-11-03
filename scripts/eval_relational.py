"""Relational queries IS(a) AND NEAR(IS(b), d) against retrieval baselines built from the same stored objects.

Every row scores the same objects (same proposals, same queries as eval_queries.py); only the query semantics differ.
  ovql_B4 / ovql_B234      OVQL, calibrated against a 4-word or the 234-word training-vocabulary background
  ovql_B234_image          + the image column (SigLIP crops) as a third expert (Replica, posed frames)
  cos_pred/<f>             vector search + the same spatial predicates: relu(cos(f_o, a)) * max_{o' near} relu(cos(f_o', b))
  label_sql/<f>            labelling then a spatial filter: label(o) = argmax cos over B234 + {a, b}; answers are objects
                           labelled a with a neighbour labelled b, ranked by cos(f_o, a)
  label_sql_closed/<f>     the same with the dataset's own class list as the label set (a closed-set system)
  phrase/<f>               cosine with the whole phrase 'a near a b' (OpenMask3D / Search3D / OVI-MAP query interface)
<f> is the per-object vector: 'clip' = mask-pooled OpenScene CLIP features, 'siglip' = multi-view SigLIP crops.
"""
import argparse
import dataclasses
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import paths  # noqa: E402
import eval_queries as Q  # noqa: E402

NICE = os.path.join(paths.DOWNLOADS, "replica_nice", "Replica")
MARGINS = (0.999, 0.95, 0.92, 0.9, 0.87, 0.85)  # CLIP text-similarity radius of a concept's neighbourhood (1.0 = the word only)


def worker(rank, gpu, files, args, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        _worker(files, args, queue)
    except Exception:
        import traceback
        traceback.print_exc()
    queue.put(None)


def _worker(files, args, queue):
    from opensnap.core import SCENE_TEMPLATES, Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import materialize
    from opensnap.query.engine import BACKGROUND, Engine
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    clip_snap = TextEncoder("ViT-B/32")
    dense = OpenSceneModel(args.openscene_ckpt, args.feature, voxel_size=args.dense_voxel)
    clip_dense = TextEncoder(dense.clip_name) if dense.clip_name != "ViT-B/32" else clip_snap
    vocab = snap_train_vocab()
    m = OpenSNAP(SnapModel(args.snap_ckpt, domain="Indoor"), clip_snap, dense, clip_dense,
                 dataclasses.replace(Config(), relay_weight=0.8), anchors=vocab)
    expert = None
    if args.images:
        from opensnap.views import ImageExpert, PosedFrames, object_embeddings
        expert = ImageExpert(args.image_encoder, cache_dir=os.path.join(paths.DOWNLOADS, "hf", "hub"))
        intr = json.load(open(os.path.join(NICE, "cam_params.json")))["camera"]
    classes = Q.classes_for(args.dataset)
    _, _, segf = Q.DATA[args.dataset]
    replay = None
    if args.queries_from:  # the exact relational queries of an earlier eval_queries run (they are sampled with hash())
        replay = {r["scene"]: [(q["a"], q["b"]) for q in r["results"]["opensnap"]["queries"] if q["kind"] == "NEAR"]
                  for r in json.load(open(args.queries_from))["records"]}

    def scene_record(f):
        scene = os.path.basename(f)
        coord = np.load(os.path.join(f, "coord.npy")).astype(np.float32)
        ins = np.load(os.path.join(f, "instance.npy")).reshape(-1).astype(np.int64)
        seg = np.load(os.path.join(f, f"{segf}.npy")).reshape(-1).astype(np.int64)
        np.random.seed(abs(hash(scene)) % 2 ** 31)
        torch.manual_seed(0)
        m.encode(coord)
        table = materialize(m, m.propose_c2f() if args.proposals == "c2f" else None)
        vc = m.scene["vcoord"].cpu().numpy()
        inv = m.scene["inverse"].cpu().numpy()
        objs = [(c, mk, (vc[mk].min(0), vc[mk].max(0))) for c, mk in Q.gt_objects(ins, seg, inv, len(vc), classes, Q.SKIP[args.dataset])]
        present = sorted({c for c, _, _ in objs})
        queries = [("IS", c, None) for c in present]
        if args.absent:  # concepts with no annotated point in the scan: a correct answer set is empty
            seen = {classes[k] for k in np.unique(seg) if 0 <= k < len(classes)}
            queries += [("IS", c, None) for c in classes if c not in seen and c not in Q.SKIP[args.dataset]]
        pairs = [(a, b) for a in present for b in present if a != b and Q.relational_gt(objs, a, b, args.near)]
        if replay is not None:
            queries += [("NEAR", a, b) for a, b in replay[scene]]
        elif args.rel_per_scene:
            rng = np.random.default_rng(abs(hash(scene)) % 2 ** 31)
            queries += [("NEAR", *pairs[k]) for k in rng.permutation(len(pairs))[:args.rel_per_scene]]
        else:
            queries += [("NEAR", a, b) for a, b in pairs]

        pm = np.zeros((len(table), len(vc)), bool)
        for i, p in enumerate(table.points):
            pm[i, p] = True
        gm = np.stack([o[1] for o in objs]) if objs else np.zeros((0, len(vc)), bool)
        inter = pm.astype(np.float32) @ gm.T.astype(np.float32)
        iou = inter / np.maximum(pm.sum(1)[:, None] + gm.sum(1)[None] - inter, 1)

        pooled = table.pooled
        if not pooled.numel():  # the collapse guard dropped the dense features; the baselines still get their vector
            d = m.openscene.voxel_features(coord, m.scene["inverse"], len(vc))
            w = torch.zeros(len(table), len(vc), device=d.device)
            for i, p in enumerate(table.points):
                w[i, torch.as_tensor(p, device=d.device, dtype=torch.long)] = 1.0
            pooled = F.normalize(w @ d, dim=-1)
        feats = {"clip": (pooled, lambda t: clip_dense(t, SCENE_TEMPLATES))}
        img = None
        if expert is not None:
            frames = PosedFrames(os.path.join(NICE, scene), intr, step=args.frame_step)
            emb, _ = object_embeddings(frames, expert, coord, pm[:, inv])
            img = emb.float()
            feats["siglip"] = (F.normalize(img, dim=-1), expert.text)
        engines = {"ovql_B4": Engine(m, table, refine=False, background=BACKGROUND),
                   "ovql_B234": Engine(m, table, refine=False, background=tuple(vocab))}
        near = engines["ovql_B4"].relation("NEAR", args.near)
        near.fill_diagonal_(0)

        def ovql_image(text):
            """Eq. (4) with the image expert: product of the 3D distribution and the SigLIP distribution over [text] + B234."""
            labels = [text, *[b for b in vocab if b != text]]
            terms = [(table.pooled @ clip_dense(labels, SCENE_TEMPLATES).T / m.cfg.tau).softmax(-1)]
            ps = m.snap_probs(F.normalize(table.token, dim=-1), labels)
            terms.append(torch.where((table.token.norm(dim=-1) > 0)[:, None], ps, terms[0]))
            l3 = (m.cfg.alpha * terms[0].clamp(min=1e-8).log() + (1 - m.cfg.alpha) * terms[1].clamp(min=1e-8).log()).log_softmax(-1)
            li = (img @ expert.text(labels).T / args.image_tau).log_softmax(-1)
            fused = ((1 - args.image_weight) * l3 + args.image_weight * li).softmax(-1)[:, 0]
            return torch.where(img.norm(dim=-1) > 0, fused, l3.exp()[:, 0])

        def concept_dist(text, image):
            """Full fused distribution over [text] + B234 and each word's CLIP text similarity to `text`."""
            labels = [text, *[b for b in vocab if b != text]]
            E = F.normalize(clip_dense(labels, SCENE_TEMPLATES).float(), dim=-1)
            terms = [(F.normalize(pooled.float(), dim=-1) @ E.T / m.cfg.tau).softmax(-1)]
            ps = m.snap_probs(F.normalize(table.token, dim=-1), labels)
            terms.append(torch.where((table.token.norm(dim=-1) > 0)[:, None], ps, terms[0]))
            l3 = (m.cfg.alpha * terms[0].clamp(min=1e-8).log() + (1 - m.cfg.alpha) * terms[1].clamp(min=1e-8).log()).log_softmax(-1)
            P = l3.exp()
            if image:
                li = (img @ expert.text(labels).T / args.image_tau).log_softmax(-1)
                P = torch.where((img.norm(dim=-1) > 0)[:, None], ((1 - args.image_weight) * l3 + args.image_weight * li).softmax(-1), P)
            return P, E @ E[0]

        def marginal(text, image, d):
            """p(object is an instance of `text`): mass of the words whose text similarity to `text` is >= d."""
            P, sim = concept_dist(text, image)
            return P[:, sim >= d].sum(1)

        def cosine(fname, text):
            v, enc = feats[fname]
            if not v.numel():
                return torch.zeros(len(table), device=near.device)
            return (F.normalize(v.float(), dim=-1) @ enc([text]).T.float())[:, 0]

        def label_of(fname, a, b, closed=False):
            v, enc = feats[fname]
            pool = classes if closed else vocab
            words = [a, *([b] if b else []), *[w for w in pool if w not in (a, b)]]
            return (F.normalize(v.float(), dim=-1) @ enc(words).T.float()).argmax(1)

        out = {}
        for kind, a, b in queries:
            sc = {}
            for name, eng in engines.items():
                pa = eng.is_prob(a)
                sc[name] = pa if kind == "IS" else pa * (near * eng.is_prob(b)[None]).max(1).values
            for d in MARGINS:
                for image in ([False, True] if img is not None else [False]):
                    name = f"ovql_marg{d}{'_image' if image else ''}"
                    pa = marginal(a, image, d)
                    sc[name] = pa if kind == "IS" else pa * (near * marginal(b, image, d)[None]).max(1).values
            if img is not None:
                pa = ovql_image(a)
                sc["ovql_B234_image"] = pa if kind == "IS" else pa * (near * ovql_image(b)[None]).max(1).values
            for fname in feats:
                ca = cosine(fname, a).clamp(min=0)
                if args.full and kind == "IS":
                    sc[f"cosraw/{fname}"] = cosine(fname, a)
                labs = {"label_sql": label_of(fname, a, b), "label_sql_closed": label_of(fname, a, b, closed=True)}
                if kind == "IS":
                    sc[f"cos_pred/{fname}"] = ca
                    for k, lab in labs.items():
                        sc[f"{k}/{fname}"] = ca * (lab == 0)
                    sc[f"phrase/{fname}"] = ca
                else:
                    cb = cosine(fname, b).clamp(min=0)
                    sc[f"cos_pred/{fname}"] = ca * (near * cb[None]).max(1).values
                    for k, lab in labs.items():
                        sc[f"{k}/{fname}"] = ca * (lab == 0) * (near * (lab == 1).float()[None]).max(1).values
                    sc[f"phrase/{fname}"] = cosine(fname, f"{a} near a {b}")
            gt_idx = [k for k, o in enumerate(objs) if o[0] == a] if kind == "IS" else Q.relational_gt(objs, a, b, args.near)
            for name, s in sc.items():
                s = s.float().cpu().numpy()
                keep = np.arange(len(s)) if args.full else np.nonzero(s > 0)[0]
                keep = keep[np.argsort(-s[keep])][:args.topk or None]
                out.setdefault(name, []).append(dict(kind=kind, a=a, b=b, scores=s[keep].tolist(), idx=keep.tolist(),
                                                     ious=iou[np.ix_(keep, gt_idx)].tolist(), n_gt=len(gt_idx)))
        names = {f: [([*vocab][int(i)]) for i in (F.normalize(v.float(), dim=-1) @ enc(vocab).T.float()).argmax(1).tolist()]
                 for f, (v, enc) in feats.items()}
        return dict(scene=scene, n_objects=len(table), results=out, labels=names)

    for f in files:
        for attempt in range(3):  # the GPU is shared; another job can briefly take the memory
            try:
                queue.put(scene_record(f))
                break
            except (torch.OutOfMemoryError, MemoryError):
                if attempt == 2:
                    raise
                torch.cuda.empty_cache()
                time.sleep(120)


def ece(ps, ys, bins=15):
    ps, ys = np.asarray(ps), np.asarray(ys)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        b = (ps > lo) & (ps <= lo + 1 / bins)
        if b.any():
            e += b.mean() * abs(ps[b].mean() - ys[b].mean())
    return float(e)


def summarize(recs):
    out = {}
    for v in dict.fromkeys(v for r in recs for v in r["results"]):
        for kind in ("IS", "NEAR"):
            per = {}
            ps, ys = [], []
            for r in recs:
                for q in r["results"].get(v, []):  # a room with no annotated object has no queries
                    if q["kind"] != kind:
                        continue
                    per.setdefault(q["a"] if kind == "IS" else (q["a"], q["b"]), []).append((q["scores"], q["ious"], q["n_gt"]))
                    for s, io in zip(q["scores"], q["ious"]):
                        ps.append(s)
                        ys.append(float(max(io, default=0) >= 0.5))
            res = {}
            for name, thrs in (("AP", np.arange(0.5, 0.96, 0.05)), ("AP50", [0.5]), ("AP25", [0.25])):
                res[name] = float(np.nanmean([np.nanmean([Q.average_precision(it, t) for t in thrs]) for it in per.values()]))
            res["ECE"] = ece(ps, ys) if v.startswith("ovql") else None
            out[f"{v}/{kind}"] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="s3dis", choices=["s3dis", "replica", "scannet"])
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/matterport_openseg.pth.tar")
    ap.add_argument("--feature", default="openseg")
    ap.add_argument("--dense_voxel", type=float, default=0.02)
    ap.add_argument("--proposals", default="fps", choices=["fps", "c2f"])
    ap.add_argument("--near", type=float, default=0.5)
    ap.add_argument("--rel_per_scene", type=int, default=5, help="0 = every (a, b) pair with a true answer")
    ap.add_argument("--queries_from", default="", help="replay the relational queries of an eval_queries output")
    ap.add_argument("--topk", type=int, default=0, help="keep the k best answers per query (0 = all, as eval_queries)")
    ap.add_argument("--absent", type=int, default=0, help="also ask IS(c) for every class absent from the scan")
    ap.add_argument("--full", type=int, default=0, help="store every object's score (needed for thresholded answer sets)")
    ap.add_argument("--images", type=int, default=0)
    ap.add_argument("--image_encoder", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--image_weight", type=float, default=0.75)
    ap.add_argument("--image_tau", type=float, default=0.02)
    ap.add_argument("--frame_step", type=int, default=10)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--procs_per_gpu", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/compare/relational_s3dis.json")
    args = ap.parse_args()
    root, splits, _ = Q.DATA[args.dataset]
    files = sorted(f for s in splits for f in glob.glob(os.path.join(paths.DOWNLOADS, "data", root, s, "*")))
    files = files[:args.limit] if args.limit else files
    gpus = [g for g in args.gpus.split(",") for _ in range(args.procs_per_gpu)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, g, files[r::len(gpus)], args, queue)) for r, g in enumerate(gpus)]
    for p in procs:
        p.start()
    recs, done = [], 0
    while done < len(procs):
        r = queue.get()
        if r is None:
            done += 1
            continue
        recs.append(r)
        print(f"[{len(recs)}/{len(files)}] {r['scene']} objs={r['n_objects']}", flush=True)
    for p in procs:
        p.join()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(args), records=recs), open(args.out, "w"))
    res = summarize(recs)
    json.dump(dict(args=vars(args), results=res, records=recs), open(args.out, "w"))
    for k, v in res.items():
        print(f"{k:28s} " + " ".join(f"{m} {100 * x:.1f}" if m != "ECE" else f"ECE {x:.3f}" for m, x in v.items() if x is not None))


if __name__ == "__main__":
    main()
