"""Open-vocabulary 3D semantic segmentation benchmark (OpenScene protocol)."""
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
from opensnap.core import Config  # noqa: E402
from opensnap.data import LABELSETS, estimate_normals, load_scene, scene_files, snap_train_vocab  # noqa: E402

METHODS = {
    "openscene": dict(use_dense=False, use_snap=False),
    "snap_token": dict(use_dense=False, beta=0.0, relay_weight=0.0),
    "snap_token_relay": dict(use_dense=False, beta=0.0),
    "mask_pool": dict(use_snap=False, beta=0.0),
    "mask_pool_dense": dict(use_snap=False),
    "opensnap_norelay": dict(relay_weight=0.0),
    "opensnap": dict(),
}


def worker(rank, gpu, files, args, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from opensnap.core import OpenSNAP
    from opensnap.openscene_model import OpenSceneModel
    from opensnap.snap_model import SnapModel
    from opensnap.text import TextEncoder
    cfg = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__})
    clip_snap = TextEncoder("ViT-B/32")
    dense = OpenSceneModel(args.openscene_ckpt, args.feature, voxel_size=args.dense_voxel)
    clip_dense = clip_snap if dense.clip_name == "ViT-B/32" else TextEncoder(dense.clip_name)
    model = OpenSNAP(SnapModel(args.snap_ckpt, chunk=args.chunk, domain=args.domain), clip_snap, dense, clip_dense, cfg,
                     anchors=snap_train_vocab())
    labels = LABELSETS[args.dataset]
    C = len(labels)
    for f in files:
        coord, color, gt = load_scene(f)
        hf = f[:-4] + ".hist.npy"
        hist = torch.from_numpy(np.load(hf)).cuda().double() if os.path.exists(hf) else None
        np.random.seed(abs(hash(os.path.basename(f))) % 2 ** 31)
        torch.manual_seed(0)
        t0 = time.time()
        model.encode(coord, color, estimate_normals(coord) if args.normals else None)
        props = model.propose()
        out, stats = {}, dict(scene=os.path.basename(f), masks=len(props.scores), points=len(coord))
        v = torch.from_numpy(gt != 255).cuda()
        g = torch.from_numpy(gt).cuda()[v]
        for name in args.methods:
            kw = dict(METHODS[name])
            over = {k: kw.pop(k) for k in ("relay_weight",) if k in kw}
            model.cfg = dataclasses.replace(cfg, **over)
            pred = model.semantic(props, labels, **kw).argmax(1)
            if hist is not None:  # score every raw point with its voxel's prediction
                out[name] = torch.zeros(C, C, dtype=torch.float64, device="cuda").index_add_(0, pred, hist).long().cpu().numpy()
            else:
                out[name] = torch.bincount(pred[v] * C + g, minlength=C * C).view(C, C).cpu().numpy()
        model.cfg = cfg
        stats["sec"] = time.time() - t0
        if args.per_scene:
            stats["conf"] = {m: c.tolist() for m, c in out.items()}
        queue.put((out, stats))
    queue.put(None)


def summarize(conf, labels, seen):
    tp = np.diag(conf).astype(np.float64)
    gt_count = conf.sum(0).astype(np.float64)
    denom = conf.sum(0) + conf.sum(1) - np.diag(conf)
    present = gt_count > 0
    iou = np.where(present, tp / np.maximum(denom, 1), np.nan)
    acc = np.where(present, tp / np.maximum(gt_count, 1), np.nan)
    res = dict(mIoU=float(np.nanmean(iou)), mAcc=float(np.nanmean(acc)),
               allAcc=float(tp.sum() / conf.sum()),
               mF1=float(np.nanmean(np.where(present, 2 * tp / np.maximum(conf.sum(0) + conf.sum(1), 1), np.nan))),
               conf=conf.tolist(),
               class_iou={labels[i]: float(iou[i]) for i in range(len(labels)) if present[i]})
    if seen is not None:
        s = np.array([l.lower() in seen for l in labels])
        res["mIoU_seen"] = float(np.nanmean(iou[s & present]))
        res["mIoU_unseen"] = float(np.nanmean(iou[~s & present]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scannet", choices=list(LABELSETS))
    ap.add_argument("--split", default="val")
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--openscene_ckpt", default="../downloads/ckpt/scannet_lseg.pth.tar")
    ap.add_argument("--feature", default="lseg", choices=["lseg", "openseg"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--procs_per_gpu", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--normals", type=int, default=0)
    ap.add_argument("--domain", default="Indoor", choices=["Indoor", "Outdoor", "Aerial"])
    ap.add_argument("--dense_voxel", type=float, default=0.02)
    ap.add_argument("--per_scene", type=int, default=0)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", default="results/eval.json")
    for k, f in Config.__dataclass_fields__.items():
        ap.add_argument(f"--{k}", type=type(f.default), default=f.default)
    args = ap.parse_args()

    files = scene_files(args.dataset, args.split, args.data_root)[::args.stride]
    files = files[:args.limit] if args.limit else files
    gpus = [g for g in args.gpus.split(",") for _ in range(args.procs_per_gpu)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, g, files[r::len(gpus)], args, queue)) for r, g in enumerate(gpus)]
    for p in procs:
        p.start()
    labels = LABELSETS[args.dataset]
    conf = {m: np.zeros((len(labels), len(labels)), np.int64) for m in args.methods}
    stats, done = [], 0
    while done < len(procs):
        item = queue.get()
        if item is None:
            done += 1
            continue
        out, s = item
        for m in out:
            conf[m] += out[m]
        stats.append(s)
        print(f"[{len(stats)}/{len(files)}] {s['scene']} masks={s['masks']} {s['sec']:.1f}s", flush=True)
    for p in procs:
        p.join()

    seen = set(snap_train_vocab()) if args.dataset.startswith("matterport") or args.dataset == "underonefacade" else None
    results = {m: summarize(conf[m], labels, seen) for m in args.methods}
    report = dict(args=vars(args), num_scenes=len(stats),
                  mean_masks=float(np.mean([s["masks"] for s in stats])),
                  sec_per_scene=float(np.mean([s["sec"] for s in stats])), results=results)
    if args.per_scene:
        report["per_scene"] = stats
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    cols = ["mIoU", "mAcc", "allAcc"] + (["mIoU_seen", "mIoU_unseen"] if seen else [])
    print(f"\n{args.dataset}/{args.split}  scenes={len(stats)}  masks/scene={report['mean_masks']:.1f}")
    print("| method | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for m, r in results.items():
        print(f"| {m} | " + " | ".join(f"{100 * r[c]:.1f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
