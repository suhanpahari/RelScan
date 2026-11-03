"""Closed-set benchmark protocols shared by OpenMask3D, Search3D and OVI-MAP: every stored object is labelled with its most
likely class from the benchmark list (instance AP), and every vertex with its most likely class (mIoU / mAcc).

  scannet200   ScanNet200 val (312 scenes), OpenMask3D/Search3D protocol (198 classes, head/common/tail) + OVI-MAP (200)
  scannet18    the 18 ScanNet scenes of OVI-MAP (3 of them, scene0168_0x, are ScanNet train scenes)
  replica      the 8 Replica scenes of OVI-MAP / OpenNeRF, scored on the NICE-SLAM GT mesh vertices with its 51 classes
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import bench, paths  # noqa: E402
from opensnap.core import Config  # noqa: E402

VOCAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opensnap", "vocab")
PC = os.path.join(paths.DOWNLOADS, "data", "pointcept")
NICE = os.path.join(paths.DOWNLOADS, "replica_nice", "Replica")
SCANNET_2D = os.path.join(paths.DOWNLOADS, "data", "scannet_2d")
OVIMAP_GT = os.path.join(paths.DOWNLOADS, "papers", "ovimap", "scripts", "datasets")
SCANNET18 = ["scene0011_00", "scene0011_01", "scene0050_00", "scene0050_01", "scene0050_02", "scene0084_00", "scene0084_01",
             "scene0084_02", "scene0168_00", "scene0168_01", "scene0168_02", "scene0231_00", "scene0231_01", "scene0231_02",
             "scene0378_00", "scene0378_01", "scene0378_02", "scene0518_00"]
REPLICA = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]
VARIANTS = {"opensnap": dict(), "dense_only": dict(use_snap=False), "token_only": dict(use_dense=False)}
IMAGE_VARIANTS = {"opensnap+image": dict(image=True), "image_only": dict(use_dense=False, use_snap=False, beta=0.0, image=True)}


S3DIS13 = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"]


def benchmark(name):
    if name == "s3dis_dev":
        scenes = sorted(os.listdir(os.path.join(PC, "s3dis", "Area_1")))
        ids = list(range(1, 14))
        return S3DIS13, ids, scenes, {"9": [i + 1 for i, l in enumerate(S3DIS13) if l not in ("ceiling", "floor", "wall", "clutter")]}
    if name.startswith("scannet"):
        v = json.load(open(os.path.join(VOCAB, "scannet200.json")))
        labels, ids = v["labels"], v["ids"]
        if name == "scannet200":
            scenes = sorted(os.listdir(os.path.join(PC, "scannet", "val")))
        elif name == "scannet_sel":
            scenes = sorted(os.listdir(os.path.join(PC, "scannet", "train")))[::40]
        else:
            scenes = SCANNET18
        inst = {"198": [i for i, l in zip(ids, labels) if l not in ("wall", "floor")], "200": list(ids)}
        return labels, ids, scenes, inst
    v = json.load(open(os.path.join(VOCAB, "replica51.json")))
    labels = [l.replace("-", " ") for l in v["labels"]]  # replica and replica_video
    ids = list(range(1, len(labels) + 1))
    inst = {"48": [v["labels"].index(l) + 1 for l in v["instance_labels"]]}
    return labels, ids, REPLICA, inst


def load_scannet(scene):
    for split in ("val", "train"):
        d = os.path.join(PC, "scannet", split, scene)
        if os.path.isdir(d):
            break
    coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
    seg = np.load(os.path.join(d, "segment200.npy")).reshape(-1).astype(np.int64)
    ins = np.load(os.path.join(d, "instance.npy")).reshape(-1).astype(np.int64)
    ids = json.load(open(os.path.join(VOCAB, "scannet200.json")))["ids"]
    sem = np.where(seg >= 0, np.asarray(ids)[seg.clip(0)], 0)
    gt = np.zeros(len(coord), np.int64)
    for k in np.unique(ins[(ins >= 0) & (sem > 0)]):
        m = (ins == k) & (sem > 0)
        gt[m] = np.bincount(sem[m]).argmax() * 1000 + k + 1
    return coord, coord, sem, gt, split, coord


ORACLE_SCALE = False


def load_s3dis(scene):
    d = os.path.join(PC, "s3dis", "Area_1", scene)
    coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
    seg = np.load(os.path.join(d, "segment.npy")).reshape(-1).astype(np.int64)
    ins = np.load(os.path.join(d, "instance.npy")).reshape(-1).astype(np.int64)
    sem = np.where(seg >= 0, seg + 1, 0)
    gt = np.zeros(len(coord), np.int64)
    for k in np.unique(ins[(ins >= 0) & (sem > 0)]):
        m = (ins == k) & (sem > 0)
        gt[m] = np.bincount(sem[m]).argmax() * 1000 + k + 1
    return coord, coord, sem, gt, "dev", coord


def load_replica_video(scene):
    """RGB-only reconstruction; ORACLE_SCALE (diagnostic only) rescales it by the GT similarity scale before processing."""
    d = os.path.join(paths.DOWNLOADS, "runs", "video_replica", scene)
    coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
    al = np.load(os.path.join(d, "align.npz"))
    _, verts, sem, gt, _, _ = load_replica(scene, mesh_only=True)
    match = (al["s"] * coord @ al["R"].T + al["t"]).astype(np.float32)
    if ORACLE_SCALE:
        coord = (coord * al["s"]).astype(np.float32)
    return coord, verts, sem, gt, "video", match


def load_replica(scene, mesh_only=False):
    from plyfile import PlyData
    coord = None if mesh_only else np.load(os.path.join(PC, "replica", "all", scene, "coord.npy")).astype(np.float32)
    ply = PlyData.read(os.path.join(NICE, f"{scene}_mesh.ply"))
    verts = np.stack([ply["vertex"][k] for k in "xyz"], 1).astype(np.float32)
    red = json.load(open(os.path.join(VOCAB, "replica51.json")))["map_to_reduced"]
    s = np.loadtxt(os.path.join(OVIMAP_GT, "replica_gt_semantics", f"semantic_labels_{scene}.txt"), dtype=np.int64)
    i = np.loadtxt(os.path.join(OVIMAP_GT, "replica_gt_instances", f"instance_labels_{scene}.txt"), dtype=np.int64)
    sem = np.vectorize(lambda x: red.get(str(x), -1) + 1)(s)
    gt = np.zeros(len(verts), np.int64)
    for c in np.unique(sem):
        if c == 0:
            continue
        for k, iid in enumerate(np.unique(i[sem == c])):
            if iid == 0:
                continue
            gt[(i == iid) & (sem == c)] = c * 1000 + k + 1
    return coord, verts, sem, gt, "eval", coord


@torch.no_grad()
def gt_proposals(m, coord, verts, gt, match, min_points):
    """Oracle proposals (ablation only): one mask per ground-truth instance, a voxel belonging to the instance that holds
    most of its points. The segmenter token comes from one click at the instance voxel nearest to its centroid; the mask
    itself is the ground truth and the predicted quality is set to 1."""
    from scipy.spatial import cKDTree
    from opensnap.core import Proposals
    g = gt if verts is coord else gt[cKDTree(verts).query(match, k=1, workers=8)[1]]
    inv = m.scene["inverse"].cpu().numpy()
    V = len(m.scene["vcoord"])
    pair, cnt = np.unique(np.stack([inv, g], 1), axis=0, return_counts=True)
    order = np.lexsort((-cnt, pair[:, 0]))
    first = np.r_[True, pair[order[1:], 0] != pair[order[:-1], 0]]
    vgt = np.zeros(V, np.int64)
    vgt[pair[order[first], 0]] = pair[order[first], 1]
    vc = m.scene["vcoord"]
    masks, prompts = [], []
    for k in np.unique(vgt[vgt > 0]):
        idx = np.nonzero(vgt == k)[0]
        if len(idx) < min_points:
            continue
        mk = torch.zeros(V, dtype=torch.bool, device=vc.device)
        mk[torch.as_tensor(idx, device=vc.device)] = True
        pts = vc[mk]
        masks.append(mk)
        prompts.append(pts[torch.cdist(pts.mean(0, keepdim=True), pts).argmin()][None])
    if not masks:
        return m.propose()
    prompts = torch.stack(prompts)
    tokens = torch.cat([m.snap.decode(prompts[i:i + 64])[1] for i in range(0, len(prompts), 64)])
    logits = torch.stack(masks).float() * 10 - 5
    return Proposals(logits, tokens, torch.ones(len(masks), device=vc.device), prompts)


def worker(rank, gpu, scenes, args, queue):
    try:
        _worker(rank, gpu, scenes, args, queue)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"worker {rank} failed: {e}", flush=True)
    queue.put(None)


def _worker(rank, gpu, scenes, args, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    global ORACLE_SCALE
    ORACLE_SCALE = bool(args.oracle_scale)
    from scipy.spatial import cKDTree
    from opensnap.core import OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    cfg = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__})
    clip = TextEncoder("ViT-B/32")
    dense = OpenSceneModel(args.openscene_ckpt, args.feature, voxel_size=args.dense_voxel)
    clip_dense = clip if dense.clip_name == "ViT-B/32" else TextEncoder(dense.clip_name)
    m = OpenSNAP(SnapModel(args.snap_ckpt, domain="Indoor"), clip, dense, clip_dense, cfg, anchors=snap_train_vocab())
    labels, ids, _, _ = benchmark(args.bench)
    load = {"replica": load_replica, "replica_video": load_replica_video, "s3dis_dev": load_s3dis}.get(args.bench, load_scannet)
    expert = None
    intr = json.load(open(os.path.join(NICE, "cam_params.json")))["camera"] if args.bench.startswith("replica") else None
    from opensnap.views import PosedFrames
    if args.images:
        from opensnap.views import ImageExpert, PosedFrames, ScanNetFrames, VideoFrames, object_embeddings
        expert = ImageExpert(cache_dir=os.path.join(paths.DOWNLOADS, "hf", "hub"))

    for scene in scenes:
        coord, verts, sem, gt, split, match = load(scene)
        np.random.seed(abs(hash(scene)) % 2 ** 31)
        torch.manual_seed(0)
        t0 = time.time()
        m.encode(coord)
        props = {"c2f": lambda: m.propose_c2f(v0=args.c2f_v0, levels=args.c2f_levels), "fps": m.propose,
                 "union": lambda: m.propose_union(v0=args.c2f_v0, levels=args.c2f_levels),
                 "gt": lambda: gt_proposals(m, coord, verts, gt, match, args.min_points)}[args.proposals]()
        n_snap = len(props.scores)
        if args.lift2d:
            from opensnap.lift2d import lift, merge_proposals
            fr = PosedFrames(os.path.join(NICE, scene), intr, step=args.lift_step)
            lifted = lift(fr, os.path.join(args.lift2d, scene), coord, m.scene["inverse"].cpu().numpy(), len(m.scene["vcoord"]))
            props = merge_proposals(props, lifted, len(m.scene["vcoord"]))
        nn = None
        if verts is not coord:
            d, nn = cKDTree(match).query(verts, k=1, workers=8)
            nn[d > args.max_dist] = -1
        inv = m.scene["inverse"].cpu().numpy()
        vmask = (props.logits > 0).cpu().numpy()
        rec = dict(scene=scene, split=split, n_props=len(props.scores), n_lifted=len(props.scores) - n_snap, sem={}, preds={},
                   dense_collapsed=m.scene["dense_collapsed"])
        to_v = lambda a, fill: a if nn is None else np.where(nn >= 0, a[nn.clip(0)], fill)
        variants = dict(VARIANTS)
        if expert is not None:
            t1 = time.time()
            frames = VideoFrames(os.path.join(paths.DOWNLOADS, "runs", "video_replica", scene, "views.npz"),
                                 scale=float(np.load(os.path.join(paths.DOWNLOADS, "runs", "video_replica", scene, "align.npz"))["s"])
                                 if args.oracle_scale else 1.0) \
                if args.bench == "replica_video" else ScanNetFrames(os.path.join(SCANNET_2D, scene)) if args.bench.startswith("scannet") \
                else PosedFrames(os.path.join(NICE, scene), intr, step=args.frame_step)
            emb, nviews = object_embeddings(frames, expert, coord, vmask[:, inv])
            rec["image_sec"], rec["views_per_object"] = time.time() - t1, float(nviews.mean())
            variants.update({k: {**v, "image": (emb, expert)} for k, v in IMAGE_VARIANTS.items()})
        for name, kw in variants.items():
            pts = m.semantic(props, labels, **kw).argmax(1).cpu().numpy()
            pv = to_v(np.asarray(ids)[pts], 0)
            C = len(ids) + 1
            idx = {c: k for k, c in enumerate([0, *ids])}
            gi = np.vectorize(idx.get)(sem)
            pi = np.vectorize(idx.get)(pv)
            ok = sem > 0
            rec["sem"][name] = np.bincount(gi[ok] * C + pi[ok], minlength=C * C).reshape(C, C)
            if nn is not None:
                ob = ok & (nn >= 0)
                rec.setdefault("sem_observed", {})[name] = np.bincount(gi[ob] * C + pi[ob], minlength=C * C).reshape(C, C)
                rec["observed"] = float((nn >= 0).mean())
            pm = m.mask_probs(props, labels, **{k: v for k, v in kw.items() if k != "beta"})
            p, c = pm.max(1)
            rec["preds"][name] = [(int(ids[int(c[k])]), float(p[k]), float(props.scores[k])) for k in range(len(c))]
        if args.dump:
            terms = {"dense": m.mask_probs(props, labels, use_snap=False), "token": m.mask_probs(props, labels, use_dense=False)}
            if expert is not None:
                terms["image"] = m.mask_probs(props, labels, use_dense=False, use_snap=False, image=(emb, expert))
            dump = dict(terms={k: v.half().cpu() for k, v in terms.items()}, scores=props.scores.cpu(),
                        w=((props.logits > 0).float() * props.logits.sigmoid()).half().cpu(), inverse=m.scene["inverse"].cpu(),
                        dense_pts=m.dense_probs(labels).half().cpu() if m.scene["dense"] is not None else None, nn=nn, sem=sem, ids=ids)
            os.makedirs(args.dump, exist_ok=True)
            torch.save(dump, os.path.join(args.dump, f"{scene}.pt"))
        rec["masks"] = [np.nonzero(to_v(vmask[k][inv], False))[0].astype(np.int32) for k in range(len(vmask))]
        rec["gt"] = gt
        rec["sec"] = time.time() - t0
        queue.put(rec)


def score(recs, args, inst_sets, labels, ids):
    sp = json.load(open(os.path.join(VOCAB, "scannet200_splits.json"))) if args.bench.startswith("scannet") else None
    name_of = dict(zip(ids, labels))
    out = {}
    for v in recs[0]["sem"]:
        conf = sum(r["sem"][v] for r in recs)
        res = {}
        tp = np.diag(conf)[1:].astype(float)
        gtc = conf[1:].sum(1).astype(float)
        prc = conf[:, 1:].sum(0).astype(float)
        pres = gtc > 0
        res["semantic"] = dict(mIoU=float((tp / np.maximum(gtc + prc - tp, 1))[pres].mean()), mAcc=float((tp / np.maximum(gtc, 1))[pres].mean()))
        if "sem_observed" in recs[0]:
            co = sum(r["sem_observed"][v] for r in recs)
            tp, gtc, prc = np.diag(co)[1:].astype(float), co[1:].sum(1).astype(float), co[:, 1:].sum(0).astype(float)
            pres = gtc > 0
            res["semantic_observed"] = dict(mIoU=float((tp / np.maximum(gtc + prc - tp, 1))[pres].mean()),
                                            mAcc=float((tp / np.maximum(gtc, 1))[pres].mean()),
                                            observed_fraction=float(np.mean([r["observed"] for r in recs])))
        for cname, cf in (("p_x_iou", lambda p, s: p * s), ("p", lambda p, s: p)):
            for sname, valid in inst_sets.items():
                scenes = []
                for r in recs:
                    n = len(r["gt"])
                    preds = []
                    for (lab, p, s), pts in zip(r["preds"][v], r["masks"]):
                        mk = np.zeros(n, bool)
                        mk[pts] = True
                        preds.append((mk, lab, cf(p, s)))
                    scenes.append(bench.match_scene(r["gt"], preds, valid))
                ap = bench.evaluate(scenes, valid)
                res[f"inst{sname}_{cname}"] = bench.summarize_ap(ap, [name_of[i] for i in valid], sp)
        out[v] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="scannet200", choices=["scannet200", "scannet18", "scannet_sel", "replica", "replica_video", "s3dis_dev"])
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/matterport_openseg.pth.tar")
    ap.add_argument("--feature", default="openseg")
    ap.add_argument("--dense_voxel", type=float, default=0.02)
    ap.add_argument("--proposals", default="c2f", choices=["fps", "c2f", "union", "gt"])
    ap.add_argument("--c2f_v0", type=float, default=0.8)
    ap.add_argument("--c2f_levels", type=int, default=3)
    ap.add_argument("--max_dist", type=float, default=0.05)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--procs_per_gpu", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/std.json")
    for k, f in Config.__dataclass_fields__.items():
        ap.add_argument(f"--{k}", type=type(f.default), default=f.default)
    ap.add_argument("--images", type=int, default=0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--oracle_scale", type=int, default=0)
    ap.add_argument("--lift2d", default="")
    ap.add_argument("--lift_step", type=int, default=20)
    ap.add_argument("--frame_step", type=int, default=10)
    args = ap.parse_args()
    labels, ids, scenes, inst_sets = benchmark(args.bench)
    scenes = scenes[:args.limit] if args.limit else scenes
    gpus = [g for g in args.gpus.split(",") for _ in range(args.procs_per_gpu)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, g, scenes[r::len(gpus)], args, q)) for r, g in enumerate(gpus)]
    for p in procs:
        p.start()
    recs, done = [], 0
    while done < len(procs):
        r = q.get()
        if r is None:
            done += 1
            continue
        recs.append(r)
        print(f"[{len(recs)}/{len(scenes)}] {r['scene']} props={r['n_props']} {r['sec']:.1f}s", flush=True)
    for p in procs:
        p.join()
    if len(recs) < len(scenes):
        print(f"INCOMPLETE: {len(recs)} of {len(scenes)} scenes; results marked incomplete", flush=True)
    torch.save(recs, args.out.replace(".json", "_recs.pt"))
    res = dict(all=score(recs, args, inst_sets, labels, ids))
    if args.bench == "scannet18":
        res["val_only"] = score([r for r in recs if r["split"] == "val"], args, inst_sets, labels, ids)
    for k in res:
        for v in res[k]:
            for kk, a in res[k][v].items():
                if (k, v) != ("all", "opensnap") and isinstance(a, dict):
                    a.pop("per_class_AP", None)
    json.dump(dict(args=vars(args), n_scenes=len(recs), expected_scenes=len(scenes), complete=len(recs) == len(scenes), sec_per_scene=float(np.mean([r["sec"] for r in recs])), results=res),
              open(args.out, "w"), indent=1)
    for k in res:
        for v, r in res[k].items():
            line = f"{k}/{v}: mIoU {100 * r['semantic']['mIoU']:.1f} mAcc {100 * r['semantic']['mAcc']:.1f}"
            for s in inst_sets:
                a = r[f"inst{s}_p_x_iou"]
                line += f" | inst{s} AP {100 * a['AP']:.1f} AP50 {100 * a['AP50']:.1f} AP25 {100 * a['AP25']:.1f}"
                if "AP_head" in a:
                    line += f" (h/c/t {100 * a['AP_head']:.1f}/{100 * a['AP_common']:.1f}/{100 * a['AP_tail']:.1f})"
            print(line)


if __name__ == "__main__":
    main()
