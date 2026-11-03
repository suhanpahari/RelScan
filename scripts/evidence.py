"""Rebuild OVQL's per-object distributions from dump_evidence.py records, over any contrast set, on CPU.

dist(rec, cols, ...) returns P (N, len(cols)) = Eq. (4) restricted to the label columns `cols` (indices into
rec["labels"]): the same product of experts as Engine.is_prob (refine=False) when cols = [query] + background, plus
the optional image expert as in eval_relational's ovql_image.
"""
import glob
import os

import torch

TAU, ALPHA, RELAY_W = 0.01, 0.7, 0.8  # opensnap.core.Config (relay_weight 0.8 as in eval_queries)


def load(d):
    return {os.path.basename(f)[:-3]: torch.load(f, weights_only=False) for f in sorted(glob.glob(os.path.join(d, "*.pt")))}


def log3d(rec, cols):
    """Log-probabilities of the 3D experts (dense + SNAP token with relay) over `cols`."""
    cols = torch.as_tensor(cols)
    direct = (rec["snap"][:, cols].float() / TAU).softmax(-1)
    relay = (rec["post"].float() @ rec["anchor_text"][:, cols].float() / TAU).softmax(-1)
    ps = direct ** (1 - RELAY_W) * relay ** RELAY_W  # snap_probs (unnormalized product, as in core.py)
    if not rec["pooled_ok"]:
        return ps.clamp(min=1e-8).log().log_softmax(-1)
    pd = (rec["dense"][:, cols].float() / TAU).softmax(-1)
    ps = torch.where(rec["token_valid"][:, None], ps, pd)
    return (ALPHA * pd.clamp(min=1e-8).log() + (1 - ALPHA) * ps.clamp(min=1e-8).log()).log_softmax(-1)


def logimg(rec, cols, tau=0.02):
    return (rec["image"][:, torch.as_tensor(cols)].float() / tau).log_softmax(-1)


def dist(rec, cols, image_w=0.0, image_tau=0.02):
    l3 = log3d(rec, cols)
    if image_w and "image" in rec:
        fused = ((1 - image_w) * l3 + image_w * logimg(rec, cols, image_tau)).softmax(-1)
        return torch.where(rec["image_valid"][:, None], fused, l3.exp())
    return l3.exp()


def dist_bayes(rec, cols, image_w=1.0, image_tau=0.02):
    """Independent-experts product p ∝ p_3D^a · p_img^b with a = 1, b = image_w (image_w = 1: the parameter-free
    Bayesian product under a uniform prior; the geometric mean in dist() uses a + b = 1 and is under-confident)."""
    l3 = log3d(rec, cols)
    if image_w and "image" in rec:
        fused = (l3 + image_w * logimg(rec, cols, image_tau)).softmax(-1)
        return torch.where(rec["image_valid"][:, None], fused, l3.exp())
    return l3.exp()


def cosine(rec, col, feat="image"):
    """Raw cosine of every object with label column `col` (baselines)."""
    if feat == "image":
        return rec["image"][:, col].float() / rec["image_norm"].clamp(min=1e-6)
    if feat == "dense":
        return rec["dense"][:, col].float() / rec["dense_norm"].clamp(min=1e-6)
    return rec[feat][:, col].float()  # round-3 experts (dense_p90, image3c_*, chorus) are stored as cosines
