"""Per-object raw evidence for offline query-semantics experiments (answer sets, presence, concept marginalization).

For every stored object of a scene this saves the raw similarities that OVQL's Eq. (4) is built from, over one fixed
label set L = SNAP training vocabulary (234) + the dataset's classes + the ScanNet200 classes:
  dense    <stored pooled OpenScene feature, CLIP-L text(w)>  (+ its norm)            (N, |L|)
  snap     cos(SNAP token, CLIP-B text(w)) and the relay terms post (N, 234), anchor_text (234, |L|)
  image    <multi-view SigLIP crop embedding, SigLIP text(w)> (+ its norm), --images  (N, |L|)
plus IoU of every object with every GT instance, the GT classes, the classes seen in the scan, and the NEAR relation.
v3 (round 3 of results/NOVELTY.md) adds, per object:
  dense_max / dense_p90   max and 90th percentile over the object's voxels of cos(voxel OpenScene feature, text(w))
  image3c_<name>          SceneSplat three-crop fused SigLIP embedding cosines (views.object_embeddings_3crop), --image3c
  obj_points / obj_offsets / inverse   the proposal masks (voxel indices) and raw point -> voxel map, so experts
                          computed later (e.g. Chorus, scripts/chorus_expert.py) pool over exactly these objects
`opensnap/evidence.py` rebuilds OVQL's distribution over any subset of L from these (identical to Engine with
refine=False), so new scoring ideas need no GPU.
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
SCANNET_2D = os.path.join(paths.DOWNLOADS, "data", "scannet_2d")


def label_set(dataset):
    from opensnap.data import snap_train_vocab
    out = list(snap_train_vocab())
    for c in [*Q.classes_for(dataset), *Q.classes_for("scannet")]:
        if c not in out:
            out.append(c)
    return out


def worker(rank, gpu, files, args, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        _worker(files, args, queue)
    except Exception:
        import traceback
        traceback.print_exc()
    queue.put(None)


def _worker(files, args, queue):
    from opensnap.core import SCENE_TEMPLATES, SNAP_TEMPLATES, Config, OpenSNAP
    from opensnap.data import snap_train_vocab
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.query.catalog import materialize
    from opensnap.query.engine import Engine
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    clip_snap = TextEncoder("ViT-B/32")
    dense = OpenSceneModel(args.openscene_ckpt, args.feature, voxel_size=args.dense_voxel)
    clip_dense = TextEncoder(dense.clip_name) if dense.clip_name != "ViT-B/32" else clip_snap
    vocab = snap_train_vocab()
    m = OpenSNAP(SnapModel(args.snap_ckpt, domain="Indoor"), clip_snap, dense, clip_dense,
                 dataclasses.replace(Config(), relay_weight=0.8), anchors=vocab)
    L = label_set(args.dataset)
    T_dense = F.normalize(clip_dense(L, SCENE_TEMPLATES).float(), dim=-1)
    T_snap = clip_snap(L, SNAP_TEMPLATES).float()
    anchor_text = (m.anchor_emb.float() @ T_snap.T).half().cpu()
    expert = None
    if args.images:
        from opensnap.views import ImageExpert, PosedFrames, ScanNetFrames, object_embeddings
        expert = ImageExpert(args.image_encoder, cache_dir=os.path.join(paths.DOWNLOADS, "hf", "hub"))
        T_img = F.normalize(expert.text(L).float(), dim=-1)
        intr = json.load(open(os.path.join(NICE, "cam_params.json")))["camera"] if args.dataset == "replica" else None
    experts3c = {}
    for name in filter(None, args.image3c.split(",")):
        from opensnap.views import ImageExpert
        short = "siglip2" if "siglip2" in name else "so400m" if "so400m" in name else name.split("/")[-1]
        e = expert if expert is not None and name == args.image_encoder else ImageExpert(name, cache_dir=os.path.join(paths.DOWNLOADS, "hf", "hub"))
        experts3c[short] = (e, F.normalize(e.text(L).float(), dim=-1))
    if experts3c and expert is None:
        from opensnap.views import PosedFrames, ScanNetFrames  # noqa: F811
        intr = json.load(open(os.path.join(NICE, "cam_params.json")))["camera"] if args.dataset == "replica" else None
    classes = Q.classes_for(args.dataset)
    _, _, segf = Q.DATA[args.dataset]

    def scene_record(f):
        scene = os.path.basename(f)
        t0 = time.time()
        coord = np.load(os.path.join(f, "coord.npy")).astype(np.float32)
        ins = np.load(os.path.join(f, "instance.npy")).reshape(-1).astype(np.int64)
        seg = np.load(os.path.join(f, f"{segf}.npy")).reshape(-1).astype(np.int64)
        np.random.seed(abs(hash(scene)) % 2 ** 31)
        torch.manual_seed(0)
        m.encode(coord)
        table = materialize(m, m.propose_c2f() if args.proposals == "c2f" else None)
        vc = m.scene["vcoord"].cpu().numpy()
        inv = m.scene["inverse"].cpu().numpy()
        objs = Q.gt_objects(ins, seg, inv, len(vc), classes, Q.SKIP[args.dataset])
        pm = np.zeros((len(table), len(vc)), bool)
        for i, p in enumerate(table.points):
            pm[i, p] = True
        gm = np.stack([o[1] for o in objs]) if objs else np.zeros((0, len(vc)), bool)
        inter = pm.astype(np.float32) @ gm.T.astype(np.float32)
        iou = inter / np.maximum(pm.sum(1)[:, None] + gm.sum(1)[None] - inter, 1)
        pt = torch.from_numpy(pm).float().cuda()
        contain = ((pt @ pt.T) / pt.sum(1).clamp(min=1)[:, None]).half().cpu()  # |o ∩ o'| / |o|: o is a fragment of o'
        if args.contain_only:
            ref = torch.load(os.path.join(args.out, f"{scene}.pt"), weights_only=False)
            same = ref["n_objects"] == len(table) and torch.equal(ref["n_points"], torch.from_numpy(pm.sum(1)))
            torch.save(dict(scene=scene, contain=contain, same=same), os.path.join(args.out + "_contain", f"{scene}.pt"))
            return dict(scene=scene, n_objects=len(table), sec=time.time() - t0, same=same)
        pooled, pooled_ok = table.pooled, bool(table.pooled.numel())
        if not pooled_ok:  # the collapse guard dropped the dense features; keep a pooled vector anyway
            d = m.openscene.voxel_features(coord, m.scene["inverse"], len(vc))
            w = torch.zeros(len(table), len(vc), device=d.device)
            for i, p in enumerate(table.points):
                w[i, torch.as_tensor(p, device=d.device, dtype=torch.long)] = 1.0
            pooled = F.normalize(w @ d, dim=-1)
        tok = F.normalize(table.token.float(), dim=-1)
        rec = dict(scene=scene, labels=L, n_objects=len(table),
                   dense=(pooled.float() @ T_dense.T).half().cpu(), dense_norm=pooled.float().norm(dim=-1).cpu(), pooled_ok=pooled_ok,
                   snap=(tok @ T_snap.T).half().cpu(),
                   post=(tok @ m.anchor_emb.float().T / m.cfg.relay_tau).softmax(-1).half().cpu(),
                   anchor_text=anchor_text, token_valid=(table.token.norm(dim=-1) > 0).cpu(),
                   iou=torch.from_numpy(iou).half(), gt_classes=[c for c, _ in objs],
                   seen=sorted({classes[k] for k in np.unique(seg) if 0 <= k < len(classes)}),
                   near=Engine(m, table, refine=False).relation("NEAR", args.near).half().cpu(),
                   n_points=torch.from_numpy(pm.sum(1)), contain=contain)
        eng = Engine(m, table, refine=False, background=tuple(vocab))  # reference values for the offline reconstruction
        rec["check"] = {w: eng.is_prob(w).float().cpu() for w in ("chair", "table", "door", "kettle")}
        if expert is not None:
            frames = PosedFrames(os.path.join(NICE, scene), intr, step=args.frame_step) if args.dataset == "replica" \
                else ScanNetFrames(os.path.join(SCANNET_2D, scene))
            emb, _ = object_embeddings(frames, expert, coord, pm[:, inv])
            emb = emb.float()
            rec["image"] = (emb @ T_img.T).half().cpu()  # raw dot, as OVQL uses it; cosine = image / image_norm
            rec["image_norm"] = emb.norm(dim=-1).cpu()
            rec["image_valid"] = (emb.norm(dim=-1) > 0).cpu()
        if args.pool_stats:
            d = m.scene["dense"] if m.scene["dense"] is not None else m.openscene.voxel_features(coord, m.scene["inverse"], len(vc))
            mx, p90 = [], []
            for p in table.points:
                S = d[torch.as_tensor(p, device=d.device, dtype=torch.long)].float() @ T_dense.T  # (n, |L|)
                mx.append(S.max(0).values)
                p90.append(S.kthvalue(max(1, int(np.ceil(0.9 * len(S)))), 0).values)
            rec["dense_max"] = torch.stack(mx).half().cpu()
            rec["dense_p90"] = torch.stack(p90).half().cpu()
        if args.masks:
            rec["obj_points"] = torch.from_numpy(np.concatenate(table.points).astype(np.int32))
            rec["obj_offsets"] = torch.from_numpy(np.cumsum([0] + [len(p) for p in table.points]).astype(np.int64))
            rec["inverse"] = torch.from_numpy(inv.astype(np.int32))
        if experts3c:
            from opensnap.views import object_embeddings_3crop
            frames = PosedFrames(os.path.join(NICE, scene), intr, step=args.frame_step) if args.dataset == "replica" \
                else ScanNetFrames(os.path.join(SCANNET_2D, scene))
            for short, (e, T) in experts3c.items():
                emb, _ = object_embeddings_3crop(frames, e, coord, pm[:, inv])
                emb = emb.float()
                rec[f"image3c_{short}"] = (emb @ T.T).half().cpu()  # unit-norm embeddings: cosines
                rec[f"image3c_{short}_valid"] = (emb.norm(dim=-1) > 0).cpu()
        rec["sec"] = time.time() - t0
        torch.save(rec, os.path.join(args.out, f"{scene}.pt"))
        return dict(scene=scene, n_objects=len(table), sec=rec["sec"])

    for f in files:
        done = os.path.join(args.out + "_contain" if args.contain_only else args.out, f"{os.path.basename(f)}.pt")
        if os.path.exists(done) or (args.contain_only and not os.path.exists(os.path.join(args.out, f"{os.path.basename(f)}.pt"))):
            queue.put(dict(scene=os.path.basename(f), n_objects=-1, sec=0))
            continue
        for attempt in range(10):  # the GPU is shared; another job can take the memory for a while
            try:
                queue.put(scene_record(f))
                break
            except (torch.OutOfMemoryError, MemoryError):
                if attempt == 9:
                    raise
                torch.cuda.empty_cache()
                time.sleep(180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="s3dis", choices=["s3dis", "replica", "scannet"])
    ap.add_argument("--split", default="", help="override the dataset split (e.g. Area_1, train)")
    ap.add_argument("--every", type=int, default=1, help="keep every k-th scene (ScanNet train[::40] = the dev scenes)")
    ap.add_argument("--offset", type=int, default=0, help="start of the every-k slice (train[5::10] = the head's training scenes)")
    ap.add_argument("--contain_only", type=int, default=0, help="add the proposal containment matrix to existing dumps in --out")
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/matterport_openseg.pth.tar")
    ap.add_argument("--feature", default="openseg")
    ap.add_argument("--dense_voxel", type=float, default=0.02)
    ap.add_argument("--proposals", default="fps", choices=["fps", "c2f"])
    ap.add_argument("--near", type=float, default=0.5)
    ap.add_argument("--images", type=int, default=0)
    ap.add_argument("--image_encoder", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--frame_step", type=int, default=10)
    ap.add_argument("--image3c", default="", help="comma list of SigLIP encoders for three-crop image evidence (v3)")
    ap.add_argument("--pool_stats", type=int, default=0, help="v3: dense_max / dense_p90 per object")
    ap.add_argument("--masks", type=int, default=0, help="v3: store the proposal masks")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, splits, _ = Q.DATA[args.dataset]
    splits = [args.split] if args.split else splits
    files = sorted(f for s in splits for f in glob.glob(os.path.join(paths.DOWNLOADS, "data", root, s, "*")))[args.offset::args.every]
    files = files[:args.limit] if args.limit else files
    os.makedirs(args.out + "_contain" if args.contain_only else args.out, exist_ok=True)
    gpus = args.gpus.split(",")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, g, files[r::len(gpus)], args, queue)) for r, g in enumerate(gpus)]
    for p in procs:
        p.start()
    done, n = 0, 0
    while done < len(procs):
        r = queue.get()
        if r is None:
            done += 1
            continue
        n += 1
        print(f"[{n}/{len(files)}] {r['scene']} objs={r['n_objects']} {r['sec']:.1f}s" + (f" same={r['same']}" if "same" in r else ""), flush=True)
    for p in procs:
        p.join()
    print("EVIDENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
