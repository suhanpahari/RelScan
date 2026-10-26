"""3D shape-text baseline: OpenShape (spconv-all, aligned to OpenCLIP ViT-bigG-14) labels each segmenter proposal.

Same proposals as OpenSNAP's mask-anchored pooling (SNAP, Outdoor, farthest-point prompts, same seeds), so the comparison
isolates the classifier: a 3D shape-text embedding of each mask (OpenMask3D-style, but 3D-only) instead of pooled
dense features. Masks are painted largest first; points no mask covers take the label of the nearest covered point.
Run with downloads/competitors/click_env/bin/python (open_clip). Output format = eval_semseg's report.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opensnap import paths  # noqa: E402
from opensnap.core import Config, OpenSNAP  # noqa: E402
from opensnap.data import LABELSETS, load_scene, scene_files  # noqa: E402
from opensnap.snap_model import SnapModel  # noqa: E402
from eval_semseg import summarize  # noqa: E402
from click_baselines import COMP  # noqa: E402

LABELS = LABELSETS["underonefacade"]
C = len(LABELS)


def load_openshape():
    import MinkowskiEngine as ME
    from huggingface_hub import hf_hub_download
    import importlib.util
    import types
    src = os.path.join(COMP, "OpenShape_code", "src", "models")
    saved = sys.modules.pop("models", None)  # load only the sparse-conv files; the package __init__ needs PointBERT deps
    sys.modules["models"] = pkg = types.ModuleType("models")
    pkg.__path__ = [src]
    try:
        for name in ("Minkowski_resnet", "Minkowski"):
            spec = importlib.util.spec_from_file_location(f"models.{name}", os.path.join(src, f"{name}.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"models.{name}"] = mod
            spec.loader.exec_module(mod)
        Minkowski = mod
    finally:
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[k]
        if saved is not None:
            sys.modules["models"] = saved
    cfg = SimpleNamespace(model=SimpleNamespace(name="MinkResNet34", in_channel=6, out_channel=1280, embedding_channel=1024,
                                                voxel_size=0.02))
    model = Minkowski.MinkResNet34(cfg).cuda()
    sd = torch.load(hf_hub_download("OpenShape/openshape-spconv-all", "model.pt"), map_location="cpu")["state_dict"]
    model.load_state_dict(OrderedDict((re.sub("module.", "", k), v) for k, v in sd.items() if "module" in k))
    return model.eval(), ME


@torch.no_grad()
def text_features():
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k",
                                                     cache_dir=os.path.join(paths.DOWNLOADS, "clip", "open_clip"))
    m = m.cuda().eval()
    t = F.normalize(m.encode_text(open_clip.tokenizer.tokenize(LABELS).cuda()).float(), dim=1)
    del m
    torch.cuda.empty_cache()
    return t


@torch.no_grad()
def classify(model, ME, coord, rgb, masks, text, n=10000, bs=16):
    probs = []
    for i in range(0, len(masks), bs):
        xyz, feats = [], []
        for m in masks[i:i + bs]:
            idx = np.nonzero(m)[0]
            idx = idx[np.random.choice(len(idx), n, replace=len(idx) < n)]
            p = coord[idx] - coord[idx].mean(0)
            p = p / max(np.linalg.norm(p, axis=1).max(), 1e-6)  # the authors' normalize_pc
            xyz.append(torch.from_numpy(p).float())
            feats.append(torch.from_numpy(np.concatenate([p, rgb[idx]], 1)).float())
        emb = model(ME.utils.batched_coordinates(xyz, dtype=torch.float32), torch.cat(feats), device="cuda",
                    quantization_size=0.02)
        probs.append((F.normalize(emb.float(), dim=1) @ text.T * 100).softmax(1))
    return torch.cat(probs).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--snap_ckpt", default="../downloads/ckpt/SNAP_C.pth")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--inst_out", default="", help="also save masks + class probabilities (npz per scene) for search")
    ap.add_argument("--out", default="results/aux/facade_openshape.json")
    args = ap.parse_args()
    text = text_features()
    model, ME = load_openshape()
    snap = OpenSNAP(SnapModel(args.snap_ckpt, chunk=16, domain="Outdoor"), None, None, None, Config())
    files = scene_files("underonefacade", args.split)
    files = files[:args.limit] if args.limit else files
    conf_all, stats = np.zeros((C, C)), []
    for f in files:
        t0 = time.time()
        coord, color, _ = load_scene(f)
        hist = np.load(f[:-4] + ".hist.npy").astype(np.float64)
        rgb = np.full_like(coord, 0.4) if color is None or color.max() < 1 else color / 255.0  # authors' default grey
        np.random.seed(abs(hash(os.path.basename(f))) % 2 ** 31)
        torch.manual_seed(0)
        snap.encode(coord, None, None)
        inv = snap.scene["inverse"].cpu().numpy()
        props = snap.propose()  # the proposals eval_semseg pools over
        masks = (props.logits > 0).cpu().numpy()[:, inv]
        masks = masks[masks.sum(1) >= 30]
        pred = np.full(len(coord), -1)
        if len(masks):
            p = classify(model, ME, coord, rgb, masks, text)
            for i in np.argsort(-masks.sum(1)):
                pred[masks[i]] = p[i].argmax()
            if args.inst_out:
                os.makedirs(args.inst_out, exist_ok=True)
                np.savez_compressed(os.path.join(args.inst_out, os.path.basename(f)[:-4] + ".npz"), probs=p,
                                    sizes=masks.sum(1), points=np.concatenate([np.nonzero(m)[0] for m in masks]))
        seen = pred >= 0
        if seen.any() and not seen.all():
            pred[~seen] = pred[seen][cKDTree(coord[seen]).query(coord[~seen])[1]]
        pred[pred < 0] = 0
        cm = np.zeros((C, C))
        np.add.at(cm, pred, hist)
        conf_all += cm
        stats.append(dict(scene=os.path.basename(f)[:-4], masks=int(len(masks)), cover=float(seen.mean()),
                          sec=time.time() - t0, conf={"openshape": cm.tolist()}))
        print(f"[{len(stats)}/{len(files)}] {stats[-1]['scene']} masks={len(masks)} cover={seen.mean():.2f} "
              f"{stats[-1]['sec']:.0f}s", flush=True)
    res = summarize(conf_all.astype(np.int64), LABELS, None)
    json.dump(dict(args=vars(args), num_scenes=len(stats), results={"openshape": res}, per_scene=stats), open(args.out, "w"),
              indent=1)
    print({k: round(100 * res[k], 1) for k in ("mIoU", "mAcc", "allAcc", "mF1")})


if __name__ == "__main__":
    main()
