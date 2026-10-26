"""Diagnostic for paper.md §8.2 (UnderOneFacade): usage facade_oracle.py <scene dir> <domain> <out.json>.
Mask-oracle mIoU on facades: every proposal takes its GT-majority class (small masks painted last), rest -> wall."""
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
from opensnap.core import Config, OpenSNAP  # noqa: E402
from opensnap.data import load_scene  # noqa: E402
from opensnap.snap_model import SnapModel  # noqa: E402

root, domain, out = sys.argv[1], sys.argv[2], sys.argv[3]
C = 15
snap = SnapModel("../downloads/ckpt/SNAP_C.pth", chunk=16, domain=domain)
model = OpenSNAP(snap, None, None, None, Config())
conf = {k: np.zeros((C, C)) for k in ("fps", "c2f", "union")}
rows = []
for f in sorted(glob.glob(os.path.join(root, "*.pth"))):
    coord, _, gt = load_scene(f)
    hist = np.load(f[:-4] + ".hist.npy").astype(np.float64)
    np.random.seed(0)
    torch.manual_seed(0)
    model.encode(coord, None, None)
    inv = model.scene["inverse"].cpu().numpy()
    for k, fn in (("fps", model.propose), ("c2f", model.propose_c2f), ("union", model.propose_union)):
        props = fn()
        m = (props.logits > 0).cpu().numpy()[:, inv]  # masks on input points
        pred = np.zeros(len(coord), np.int64)
        covered = np.zeros(len(coord), bool)
        for i in np.argsort(-m.sum(1)):
            h = hist[m[i]].sum(0)
            pred[m[i]] = h.argmax()
            covered |= m[i]
        cm = np.zeros((C, C))
        np.add.at(cm, pred, hist)
        conf[k] += cm
        rows.append(dict(scene=os.path.basename(f), kind=k, masks=int(len(m)), cover=float(hist[covered].sum() / hist.sum())))
        print(rows[-1], flush=True)


def miou(c):
    tp = np.diag(c)
    present = c.sum(0) > 0
    return float(np.mean((tp / np.maximum(c.sum(0) + c.sum(1) - tp, 1))[present]))


res = {k: dict(mIoU=miou(c), class_iou=(np.diag(c) / np.maximum(c.sum(0) + c.sum(1) - np.diag(c), 1)).round(3).tolist())
       for k, c in conf.items()}
print(json.dumps({k: round(100 * v["mIoU"], 1) for k, v in res.items()}))
json.dump(dict(results=res, rows=rows), open(out, "w"), indent=1)
