"""Interactive 3D instance segmentation: IoU@k clicks and NoC@{80,85,90} with simulated error-region clicks."""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402

DATA = {"s3dis": ("pointcept/s3dis", ["Area_5"]), "scannet": ("pointcept/scannet", ["val"]),
        "nuscenes": ("pointcept/nuscenes", ["val"]), "semantickitti": ("pointcept/semantickitti", ["val"]),
        "replica": ("pointcept/replica", ["all"]), "underonefacade": ("pointcept/underonefacade", ["test"])}
STRUCT = {"s3dis": {0, 1, 2}, "scannet": set(), "nuscenes": set(), "semantickitti": set(), "replica": set(), "underonefacade": set()}


def load(path, use_color, stuff=0):
    d = {k: np.load(os.path.join(path, f"{k}.npy")) for k in ("coord", "instance")}
    for k in ("color", "normal"):
        f = os.path.join(path, f"{k}.npy")
        d[k] = np.load(f) if os.path.exists(f) else np.zeros_like(d["coord"])
    seg_file = "segment.npy" if os.path.exists(os.path.join(path, "segment.npy")) else "segment200.npy"
    d["segment"] = np.load(os.path.join(path, seg_file))
    ins = d["instance"].reshape(-1).astype(np.int64)
    seg = d["segment"].reshape(-1).astype(np.int64)
    color = d["color"].astype(np.float32) if use_color else None
    if stuff:
        ins = np.where((ins < 0) & (seg >= stuff), 100000 + seg, ins)
    return d["coord"].astype(np.float32), color, d["normal"].astype(np.float32), ins, seg


def majority(labels):
    labels = labels[labels >= 0]
    return int(np.bincount(labels).argmax()) if len(labels) else -1


def next_click(pred, gt, vcoord):
    """Positive click at the centre of the missed object region (SNAP supports positive clicks only)."""
    err = gt & ~pred
    if err.sum() == 0:
        return None
    return centre(err, vcoord)


def centre(region, vcoord):
    """Point of `region` farthest from everything outside it."""
    r = vcoord[region]
    out = vcoord[~region]
    sub = out[torch.randperm(len(out), device=out.device)[:20000]]
    d = torch.cat([torch.cdist(r[i:i + 20000], sub).min(1).values for i in range(0, len(r), 20000)])
    return torch.nonzero(region)[d.argmax(), 0]


def click_object_pn(seg, vcoord, gt, args):
    """Corrective protocol: each click goes to the larger error region (positive on misses, negative on false positives)."""
    pos, neg, ious = [], [], []
    pred = torch.zeros_like(gt)
    for k in range(args.max_clicks):
        fn, fp = gt & ~pred, pred & ~gt
        if not fn.any() and not fp.any():
            ious += [ious[-1]] * (args.max_clicks - k)
            break
        if fn.sum() >= fp.sum():
            pos.append(int(centre(fn, vcoord)))
        else:
            neg.append(int(centre(fp, vcoord)))
        pred = seg(vcoord, torch.as_tensor(pos, dtype=torch.long, device=vcoord.device),
                   torch.as_tensor(neg, dtype=torch.long, device=vcoord.device))
        ious.append(float((pred & gt).sum() / (pred | gt).sum()))
    return ious


def click_object(snap, vcoord, gt, args):
    if args.protocol == "pn":
        return click_object_pn(args.seg, vcoord, gt, args)
    clicks, ious = [], []
    pred = torch.zeros_like(gt)
    for k in range(args.max_clicks):
        c = next_click(pred, gt, vcoord)
        if c is None:
            ious += [ious[-1]] * (args.max_clicks - k)
            break
        clicks.append(c)
        if args.model == "plus":
            pred = snap.decode_clicks(vcoord[torch.stack(clicks)], select=args.select, gt=gt)
        else:
            pred = snap.decode_clicks(vcoord[torch.stack(clicks)])
        ious.append(float((pred & gt).sum() / (pred | gt).sum()))
    return ious


def worker(rank, gpu, files, args, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    snap = build_model(args)
    if args.adapter:
        snap.load_adapter(args.adapter)
    baseline = None
    if args.method in ("agile3d", "interobject3d", "pointsam"):  # published segmenters, same clicks and crops
        from click_baselines import BASELINES
        args.seg = baseline = BASELINES[args.method]()
    elif args.protocol == "pn":
        from opensnap.clicks import ClickSegmenter, CorrectiveSegmenter
        tuner = ClickSegmenter(snap, "ttt" if args.method == "v5ttt" else args.method, steps=args.steps, lr=args.lr, reg=args.reg,
                               margin=args.margin, radius=args.radius, tta=args.tta, tta_radius=args.tta_radius)
        args.seg = CorrectiveSegmenter(snap, args.corr_ckpt, tuner if args.method == "v5ttt" else None) \
            if args.method in ("v5", "v5ttt") else tuner
    struct = STRUCT[args.dataset]
    for f in files:
        coord, color, normal, ins, seg = load(f, args.color, args.stuff)
        ids = [i for i in np.unique(ins) if i >= 0 and (ins == i).sum() >= args.min_points
               and (not args.skip_structure or majority(seg[ins == i]) not in struct)]
        if not args.crops:
            vcoord, inv = snap.encode(coord, color, normal if args.normals else None)
            ins_t = torch.from_numpy(ins).to(vcoord.device)
        for i in ids:
            if args.crops:
                c = coord[ins == i].mean(0)
                sel = (np.abs(coord[:, 0] - c[0]) < args.crop_half) & (np.abs(coord[:, 1] - c[1]) < args.crop_half)
                vcoord, inv = snap.encode(coord[sel], None if color is None else color[sel], normal[sel] if args.normals else None)
                gt = torch.zeros(len(vcoord), dtype=torch.bool, device=vcoord.device)
                gt[inv[torch.from_numpy(ins[sel] == i).to(vcoord.device)]] = True
            else:
                gt = torch.zeros(len(vcoord), dtype=torch.bool, device=vcoord.device)
                gt[inv[ins_t == i]] = True
            if not gt.any():  # object lost in voxelization/crop
                continue
            if baseline is not None:
                pts = np.zeros_like(coord) if color is None else color  # --color 0: XYZ-only baseline
                pts = pts[sel] if args.crops else pts
                vcolor = torch.zeros(len(vcoord), 3, device=vcoord.device).index_add_(0, inv, torch.from_numpy(pts).to(vcoord.device))
                cnt = torch.bincount(inv, minlength=len(vcoord)).clamp(min=1)[:, None]
                baseline.prepare(vcoord, (vcolor / cnt).cpu().numpy())
            ious = click_object(snap, vcoord, gt, args)
            queue.put(dict(scene=os.path.basename(f), obj=int(i), size=int(gt.sum()), seg=majority(seg[ins == i]), ious=ious))
    queue.put(None)


def build_model(args):
    from opensnap.snap_model import SnapModel
    if args.model == "snap":
        return SnapModel(args.ckpt, chunk=1, domain=args.domain)
    from opensnap.snap_plus import SnapPlusModel
    return SnapPlusModel(args.ckpt, use_sem=bool(args.use_sem))


def summarize(recs, max_clicks):
    I = np.array([r["ious"] for r in recs])
    out = {f"IoU@{k}": float(I[:, k - 1].mean()) for k in (1, 3, 5, 10) if k <= max_clicks}
    for t in (0.8, 0.85, 0.9):
        hit = I >= t
        noc = np.where(hit.any(1), hit.argmax(1) + 1, max_clicks)
        out[f"NoC@{int(t * 100)}"] = float(noc.mean())
    out["objects"] = len(recs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="s3dis", choices=list(DATA))
    ap.add_argument("--model", default="snap", choices=["snap", "plus"])
    ap.add_argument("--ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--max_clicks", type=int, default=10)
    ap.add_argument("--select", default="iou", choices=["iou", "first", "oracle"])
    ap.add_argument("--crops", type=int, default=0)
    ap.add_argument("--crop_half", type=float, default=1.5)
    ap.add_argument("--use_sem", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_points", type=int, default=100)
    ap.add_argument("--protocol", default="pos", choices=["pos", "pn"])
    ap.add_argument("--method", default="snap", choices=["snap", "subtract", "ttt", "v5", "v5ttt", "agile3d", "interobject3d",
                                                          "pointsam"])
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--tta_radius", type=float, default=0.1)
    ap.add_argument("--corr_ckpt", default="../downloads/checkpoints/opensnap_corrective_v5_multi.pth")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reg", type=float, default=10.0)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--radius", type=float, default=0.0)
    ap.add_argument("--domain", default="Indoor", choices=["Indoor", "Outdoor", "Aerial"])
    ap.add_argument("--stuff", type=int, default=0, help="classes >= this id become one segment each (nuScenes: 10)")
    ap.add_argument("--skip_structure", type=int, default=0)
    ap.add_argument("--color", type=int, default=1)
    ap.add_argument("--normals", type=int, default=1)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--procs_per_gpu", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="", help="override the dataset's default split(s)")
    ap.add_argument("--out", default="results/interactive.json")
    args = ap.parse_args()
    root, splits = DATA[args.dataset]
    splits = args.split.split(",") if args.split else splits
    files = sorted(f for s in splits for f in glob.glob(os.path.join(paths.DOWNLOADS, "data", root, s, "*")))
    files = files[::args.stride]
    files = files[:args.limit] if args.limit else files
    gpus = [g for g in args.gpus.split(",") for _ in range(args.procs_per_gpu)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, g, files[r::len(gpus)], args, queue)) for r, g in enumerate(gpus)]
    for p in procs:
        p.start()
    recs, done, t0 = [], 0, time.time()
    while done < len(procs):
        r = queue.get()
        if r is None:
            done += 1
            continue
        recs.append(r)
        if len(recs) % 100 == 0:
            print(f"{len(recs)} objects {time.time() - t0:.0f}s", flush=True)
    for p in procs:
        p.join()
    res = summarize(recs, args.max_clicks)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(args), results=res, records=recs), open(args.out, "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
