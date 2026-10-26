"""Render-then-lift baselines for zero-shot facade segmentation (UnderOneFacade has no camera images).

Each facade is rendered from virtual pinhole cameras on its street side (frontal, two obliques, zoomed tiles), a 2D
open-vocabulary segmenter labels every view, and labels are lifted to the visible points through a z-buffer of point
indices and summed over views; unseen points take the label of their nearest seen point. 2D segmenters:
  gsam        GroundingDINO (all class names as one prompt) boxes -> SAM masks (Grounded-SAM)
  sam_siglip  SAM automatic masks, each classified by SigLIP (the image expert of OVI-MAP)
Colour: real RGB where the scan has it (UK/Singapore), Lambert-shaded geometry otherwise; --shade geom shades all.
Output matches eval_semseg's report (confusion over every raw point via the per-voxel label histograms).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402
from opensnap.data import LABELSETS, scene_files  # noqa: E402
from eval_semseg import summarize  # noqa: E402

LABELS = LABELSETS["underonefacade"]
C = len(LABELS)


def normals(coord):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coord.astype(np.float64)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=200))
    return np.asarray(pcd.normals, np.float32)


def cameras(coord, fov=60.0, tile=10.0, band=8.0):
    """Street-side cameras: frontal + two obliques of the whole facade, then frontal tiles (~tile x band metres)."""
    xy = coord[:, :2] - coord[:, :2].mean(0)
    w, v = np.linalg.eigh(np.cov(xy.T))
    a = np.r_[v[:, 1], 0.0]  # along the facade
    n = np.r_[v[:, 0], 0.0]  # facade normal (horizontal)
    c = np.r_[coord[:, :2].mean(0), 0.0]
    low = coord[:, 2] < coord[:, 2].min() + 1.0  # ground points lie on the scanned (street) side
    if np.median((coord[low] - c) @ n) < 0:
        n = -n
    s, z = (coord - c) @ a, coord[:, 2]
    s0, s1, z0, z1 = np.percentile(s, 1), np.percentile(s, 99), np.percentile(z, 1), np.percentile(z, 99)
    t = np.tan(np.radians(fov / 2))
    cams = []

    def add(sc, zc, half, yaw=0.0):
        target = c + a * sc + np.r_[0, 0, zc]
        d = half / t * 1.1
        direction = np.cos(np.radians(yaw)) * n + np.sin(np.radians(yaw)) * a
        cams.append((target + d * direction, target, d))

    half = max((s1 - s0) / 2, (z1 - z0) / 2)
    for yaw in (0.0, -35.0, 35.0):
        add((s0 + s1) / 2, (z0 + z1) / 2, half, yaw)
    for sc in np.arange(s0 + tile / 2, max(s1 - tile / 2, s0 + tile / 2) + 1e-6, tile * 0.7):
        for zc in np.arange(z0 + band / 2, max(z1 - band / 2, z0 + band / 2) + 1e-6, band * 0.7):
            add(sc, zc, max(tile, band) / 2)
    return cams


def look_at(eye, target):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, (0, 0, 1.0))
    r /= np.linalg.norm(r)
    return np.stack([r, -np.cross(r, f), f])


def render(xyz, rgb, eye, target, dist, size=1024, fov=60.0, spacing=0.05, shade_depth=False):
    """Z-buffered splats; returns image and per-pixel point index (-1 = empty)."""
    R = torch.as_tensor(look_at(eye, target), dtype=torch.float32, device=xyz.device)
    cam = (xyz - torch.as_tensor(eye, dtype=torch.float32, device=xyz.device)) @ R.T
    z = cam[:, 2]
    ok = z > 0.1
    idx = torch.nonzero(ok)[:, 0]
    focal = size / 2 / np.tan(np.radians(fov / 2))
    u = (focal * cam[ok, 0] / z[ok] + size / 2).long()
    v = (focal * cam[ok, 1] / z[ok] + size / 2).long()
    rad = int(np.clip(np.ceil(focal * spacing / dist), 1, 6))
    key = torch.full((size * size,), torch.iinfo(torch.int64).max, dtype=torch.int64, device=xyz.device)
    packed = (z[ok] * 1000).long().clamp(max=2 ** 30) * (1 << 32) + idx
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            uu, vv = u + dx, v + dy
            m = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)
            key.scatter_reduce_(0, (vv[m] * size + uu[m]), packed[m], reduce="amin")
    hit = key != torch.iinfo(torch.int64).max
    pix = torch.where(hit, key & ((1 << 32) - 1), torch.full_like(key, -1)).view(size, size)
    img = torch.full((size * size, 3), 255.0, device=xyz.device)
    img[hit] = rgb[pix.view(-1)[hit]]
    if shade_depth:  # depth cue so surfaces at different depths separate in geometry-only renders
        zz = z[pix.view(-1)[hit]]
        img[hit] *= (1.15 - 0.45 * ((zz - zz.min()) / (zz.max() - zz.min() + 1e-6)))[:, None]
    return Image.fromarray(img.view(size, size, 3).clamp(0, 255).byte().cpu().numpy()), pix


class GroundedSAM:
    def __init__(self, box_thr=0.25, text_thr=0.2):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
        cache = os.path.join(paths.DOWNLOADS, "hf", "hub")
        self.dp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base", cache_dir=cache)
        self.dm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base", cache_dir=cache).cuda().eval()
        self.sp = SamProcessor.from_pretrained("facebook/sam-vit-huge", cache_dir=cache)
        self.sm = SamModel.from_pretrained("facebook/sam-vit-huge", cache_dir=cache).cuda().eval()
        self.prompt = " . ".join(LABELS) + " ."
        self.box_thr, self.text_thr = box_thr, text_thr

    def phrase(self, p):
        p = p.strip().lower()
        hits = [i for i, l in enumerate(LABELS) if l in p] or [i for i, l in enumerate(LABELS) if p and p in l]
        return max(hits, key=lambda i: len(LABELS[i])) if hits else -1

    @torch.no_grad()
    def __call__(self, img):
        x = self.dp(images=img, text=self.prompt, return_tensors="pt").to("cuda")
        out = self.dm(**x)
        r = self.dp.post_process_grounded_object_detection(out, x.input_ids, threshold=self.box_thr,
                                                           text_threshold=self.text_thr, target_sizes=[img.size[::-1]])[0]
        lab = [self.phrase(p) for p in r.get("text_labels", r["labels"])]
        keep = [i for i, l in enumerate(lab) if l >= 0]
        if not keep:
            return []
        boxes, scores = r["boxes"][keep], r["scores"][keep]
        s = self.sp(img, input_boxes=[boxes.tolist()], return_tensors="pt").to("cuda")
        o = self.sm(**s, multimask_output=False)
        masks = self.sp.image_processor.post_process_masks(o.pred_masks.cpu(), s["original_sizes"].cpu(),
                                                           s["reshaped_input_sizes"].cpu())[0][:, 0]
        return [(masks[i].numpy(), lab[k], float(scores[i])) for i, k in enumerate(keep)]


class SAMSigLIP:
    def __init__(self):
        from transformers import pipeline
        from opensnap.views import ImageExpert
        cache = os.path.join(paths.DOWNLOADS, "hf", "hub")
        self.gen = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0, model_kwargs=dict(cache_dir=cache))
        self.expert = ImageExpert(cache_dir=cache)

    @torch.no_grad()
    def __call__(self, img):
        masks = self.gen(img, points_per_batch=128, pred_iou_thresh=0.86)["masks"]
        if not masks:
            return []
        a = np.asarray(img)
        crops = []
        for m in masks:
            ys, xs = np.nonzero(m)
            pad = int(0.1 * max(np.ptp(ys), np.ptp(xs))) + 4
            y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad
            crops.append(Image.fromarray(a[y0:y1, x0:x1]))
        p = self.expert.probs(self.expert.images(crops), LABELS)
        conf, lab = p.max(1)
        return [(m, int(l), float(c)) for m, l, c in zip(masks, lab.tolist(), conf.tolist())]


def merge_instances(dets, iou=0.3, min_points=30):
    """Lifted 2D masks -> 3D instances: greedy by score, a mask joins a same-label instance it overlaps (point IoU)."""
    inst = []  # [label, score, sorted point indices]
    for pts, l, s in sorted(dets, key=lambda d: -d[2]):
        if len(pts) < min_points:
            continue
        for it in inst:
            if it[0] == l:
                inter = len(np.intersect1d(pts, it[2], assume_unique=True))
                if inter / (len(pts) + len(it[2]) - inter) > iou:
                    it[2] = np.union1d(it[2], pts)
                    it[1] = max(it[1], s)
                    break
        else:
            inst.append([l, s, pts])
    return inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--method", default="gsam", choices=["gsam", "sam_siglip"])
    ap.add_argument("--shade", default="rgb", choices=["rgb", "geom"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_views", default="")
    ap.add_argument("--inst_out", default="", help="also save lifted 3D instances (npz per scene) for the search benchmark")
    ap.add_argument("--out", default="results/facade_render_lift.json")
    args = ap.parse_args()
    seg = GroundedSAM() if args.method == "gsam" else SAMSigLIP()
    files = scene_files("underonefacade", args.split)
    files = files[:args.limit] if args.limit else files
    conf_all, stats = np.zeros((C, C)), []
    for f in files:
        t0 = time.time()
        coord, color, _ = torch.load(f, weights_only=False)
        hist = np.load(f[:-4] + ".hist.npy").astype(np.float64)
        rgb = (np.asarray(color) + 1) * 127.5 if not np.isscalar(color) else np.zeros_like(coord)
        if args.shade == "geom" or rgb.max() < 1:
            nrm = normals(coord)
            rgb = np.repeat(60 + 170 * np.abs(nrm @ np.array([0.4, -0.5, 0.77]))[:, None], 3, 1)
        shaded = args.shade == "geom" or ((np.asarray(color) + 1) * 127.5).max() < 1 if not np.isscalar(color) else True
        xyz, col = torch.from_numpy(coord).cuda(), torch.from_numpy(rgb.astype(np.float32)).cuda()
        votes = torch.zeros(len(coord), C, device="cuda")
        cams = cameras(coord)
        dets = []
        for k, (eye, target, d) in enumerate(cams):
            img, pix = render(xyz, col, eye, target, d, size=args.size, shade_depth=shaded)
            if args.save_views and k < 3:
                os.makedirs(args.save_views, exist_ok=True)
                img.save(os.path.join(args.save_views, f"{os.path.basename(f)[:-4]}_{k}.jpg"))
            label = torch.full(pix.shape, -1, dtype=torch.long, device="cuda")
            score = torch.zeros(pix.shape, device="cuda")
            for m, l, s in sorted(seg(img), key=lambda r: -r[0].sum()):  # smaller masks painted last win
                m = torch.as_tensor(m, device="cuda").bool()
                label[m], score[m] = l, s
                if args.inst_out:
                    dets.append((torch.unique(pix[m & (pix >= 0)]).cpu().numpy(), l, s))
            ok = (pix >= 0) & (label >= 0)
            votes.index_put_((pix[ok], label[ok]), score[ok], accumulate=True)
        seen = votes.sum(1) > 0
        pred = votes.argmax(1).cpu().numpy()
        if seen.any() and not seen.all():
            s_idx = torch.nonzero(seen)[:, 0].cpu().numpy()
            u_idx = torch.nonzero(~seen)[:, 0].cpu().numpy()
            pred[u_idx] = pred[s_idx[cKDTree(coord[s_idx]).query(coord[u_idx])[1]]]
        if args.inst_out:
            os.makedirs(args.inst_out, exist_ok=True)
            inst = merge_instances(dets)
            np.savez_compressed(os.path.join(args.inst_out, os.path.basename(f)[:-4] + ".npz"),
                                labels=np.array([i[0] for i in inst], np.int64), scores=np.array([i[1] for i in inst]),
                                sizes=np.array([len(i[2]) for i in inst], np.int64),
                                points=np.concatenate([i[2] for i in inst]) if inst else np.zeros(0, np.int64))
        cm = np.zeros((C, C))
        np.add.at(cm, pred, hist)
        conf_all += cm
        stats.append(dict(scene=os.path.basename(f)[:-4], views=len(cams), seen=float(seen.float().mean()),
                          sec=time.time() - t0, conf={args.method: cm.tolist()}))
        print(f"[{len(stats)}/{len(files)}] {stats[-1]['scene']} views={len(cams)} seen={stats[-1]['seen']:.2f} "
              f"{stats[-1]['sec']:.0f}s", flush=True)
    res = summarize(conf_all.astype(np.int64), LABELS, None)
    report = dict(args=vars(args), num_scenes=len(stats), results={args.method: res}, per_scene=stats)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print({k: round(100 * res[k], 1) for k in ("mIoU", "mAcc", "allAcc", "mF1")})


if __name__ == "__main__":
    main()
