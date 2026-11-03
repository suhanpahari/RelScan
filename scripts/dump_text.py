"""Text-side evidence for the answer-set study: label-label similarities in each encoder's space, and SigLIP's own
sigmoid calibration (logit scale, bias). Saved per dataset next to the evidence dumps."""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import paths  # noqa: E402
from dump_evidence import label_set  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../downloads/runs/evidence/text")
    ap.add_argument("--image_encoder", default="google/siglip-so400m-patch14-384")
    a = ap.parse_args()
    from opensnap.core import SCENE_TEMPLATES, SNAP_TEMPLATES
    from opensnap.text import TextEncoder
    from opensnap.views import ImageExpert
    clip_l, clip_b = TextEncoder("ViT-L/14@336px"), TextEncoder("ViT-B/32")
    expert = ImageExpert(a.image_encoder, cache_dir=os.path.join(paths.DOWNLOADS, "hf", "hub"))
    for ds in ("s3dis", "scannet", "replica"):
        L = label_set(ds)
        E = {"clip_l": F.normalize(clip_l(L, SCENE_TEMPLATES).float(), dim=-1),
             "clip_b": F.normalize(clip_b(L, SNAP_TEMPLATES).float(), dim=-1),
             "siglip": F.normalize(expert.text(L).float(), dim=-1)}
        rec = dict(labels=L, sim={k: (e @ e.T).half().cpu() for k, e in E.items()},
                   siglip_scale=float(expert.model.logit_scale.exp()), siglip_bias=float(expert.model.logit_bias))
        torch.save(rec, os.path.join(a.out, f"{ds}.pt"))
        print(ds, len(L), rec["siglip_scale"], rec["siglip_bias"])


if __name__ == "__main__":
    main()
