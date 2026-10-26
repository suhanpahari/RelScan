"""Published interactive 3D segmenters behind eval_interactive's corrective-click interface: seg(vcoord, pos, neg) -> mask.

Each baseline sees the same crop points (SNAP's voxels), the same click sequence and the same ground truth as OpenSNAP,
with its released weights and its own input conventions (grid, colour scaling, click encoding). Colour is passed
through (zero where the scan has none); OpenSNAP itself is XYZ-only. Point-SAM needs torkit3d, so run with
downloads/competitors/click_env/bin/python.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

COMP = os.environ.get("OPENSNAP_COMPETITORS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "downloads", "competitors"))


class Baseline:
    """Tracks click order across calls (pos/neg lists only ever grow by one click)."""

    def prepare(self, vcoord, vcolor):
        self.n_pos = self.n_neg = 0
        self.order = []  # (voxel index, is_positive) in click order
        self.encode(vcoord, vcolor)

    def __call__(self, vcoord, pos, neg):
        if len(pos) > self.n_pos:
            self.order.append((int(pos[-1]), True))
        if len(neg) > self.n_neg:
            self.order.append((int(neg[-1]), False))
        self.n_pos, self.n_neg = len(pos), len(neg)
        return self.segment(vcoord)


class AGILE3D(Baseline):
    def __init__(self, ckpt=None):
        import importlib
        import MinkowskiEngine as ME
        root = os.path.join(COMP, "AGILE3D")
        # AGILE3D's top-level `models`/`utils` packages collide with SNAP's: import them in isolation, then restore
        clash = ("models", "utils", "datasets")
        saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in clash}
        for k in saved:
            del sys.modules[k]
        sys.path.insert(0, root)
        try:
            build_model = importlib.import_module("models").build_model
        finally:
            sys.path.remove(root)
            for k in [k for k in sys.modules if k.split(".")[0] in clash]:
                del sys.modules[k]
            sys.modules.update(saved)
        # model defaults of AGILE3D's eval_single_obj.py
        args = SimpleNamespace(dialations=[1, 1, 1, 1], conv1_kernel_size=5, bn_momentum=0.02, voxel_size=0.05, hidden_dim=128,
                               dim_feedforward=1024, num_heads=8, num_decoders=3, num_bg_queries=10, dropout=0.0,
                               pre_norm=False, normalize_pos_enc=True, positional_encoding_type="fourier", gauss_scale=1.0,
                               hlevels=[4], shared_decoder=False, aux=True)
        self.voxel = args.voxel_size
        self.model = build_model(args).cuda().eval()
        sd = torch.load(ckpt or os.path.join(root, "weights", "checkpoint1099.pth"), map_location="cpu")["model"]
        missing, _ = self.model.load_state_dict(sd, strict=False)
        assert not missing, missing
        self.ME = ME

    @torch.no_grad()
    def encode(self, vcoord, vcolor):
        c = vcoord.cpu().numpy()
        c = c - c.min(0)
        qv, uniq, inv = self.ME.utils.sparse_quantize(coordinates=c, quantization_size=self.voxel, return_index=True,
                                                      return_inverse=True)
        self.inv = torch.as_tensor(inv, device="cuda").long()
        feats = torch.from_numpy(vcolor[uniq] / 255.0).float()
        data = self.ME.SparseTensor(coordinates=self.ME.utils.batched_coordinates([qv]), features=feats, device="cuda")
        raw = torch.from_numpy(c[uniq]).float().cuda()
        self.feat = self.model.forward_backbone(data, raw_coordinates=raw)

    @torch.no_grad()
    def segment(self, vcoord):
        clicks, times = {"0": [], "1": []}, {"0": [], "1": []}
        for t, (i, positive) in enumerate(self.order):
            k = "1" if positive else "0"
            clicks[k].append(int(self.inv[i]))
            times[k].append(t)
        out = self.model.forward_mask(*self.feat, click_idx=[clicks], click_time_idx=[times])
        pred = out["pred_masks"][0].argmax(-1)
        for k, ids in clicks.items():  # as in AGILE3D's evaluation: clicked voxels take their click label
            pred[ids] = int(k)
        return (pred == 1)[self.inv]


class InterObject3D(Baseline):
    def __init__(self, ckpt=None, cube=0.05):
        import MinkowskiEngine as ME
        sys.path.insert(0, os.path.join(COMP, "InterObject3D", "InterObject3D", "Minkowski", "training"))
        from examples.minkunet import MinkUNet34C
        self.model = MinkUNet34C(in_channels=5, out_channels=2, D=3).cuda().eval()
        self.model.load_state_dict(torch.load(ckpt or os.path.join(COMP, "InterObject3D", "weights", "weights_exp14_14.pth"),
                                              map_location="cpu"))
        self.ME, self.cube, self.voxel = ME, cube, 0.05

    def encode(self, vcoord, vcolor):
        self.coord = vcoord.float()
        self.rgb = torch.from_numpy(vcolor / 255.0).float().cuda()
        self.coords = self.ME.utils.batched_coordinates([self.coord / self.voxel], device="cuda")

    @torch.no_grad()
    def segment(self, vcoord):
        click = torch.zeros(len(self.coord), 2, device="cuda")
        for i, positive in self.order:  # each click marks a cube of half-edge `cube` (the authors' --cubeedge)
            near = ((self.coord - self.coord[i]).abs() < self.cube).all(1)
            click[near, 0 if positive else 1] = 1
        x = self.ME.SparseTensor(features=torch.cat([self.rgb, click], 1), coordinates=self.coords,
                                 quantization_mode=self.ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE, device="cuda")
        return self.model(x).slice(x).F.argmax(1) == 1


class PointSAM(Baseline):
    def __init__(self, ckpt=None):
        import hydra
        from omegaconf import OmegaConf
        from safetensors.torch import load_model
        root = os.path.join(COMP, "Point-SAM")
        sys.path.insert(0, root)
        with hydra.initialize_config_dir(os.path.join(root, "configs"), version_base=None):
            cfg = hydra.compose(config_name="large")
            OmegaConf.resolve(cfg)
        self.model = hydra.utils.instantiate(cfg.model)
        load_model(self.model, ckpt or os.path.join(root, "pretrained", "model.safetensors"))
        self.model.cuda().eval()

    @torch.no_grad()
    def encode(self, vcoord, vcolor):
        from pc_sam.model.pc_sam import AuxInputs
        m, n = self.model, len(vcoord)
        g = m.pc_encoder.patch_embed.grouper  # the authors' rule from evaluation/eval_kitti.py
        g.num_groups, g.group_size = (2048, 256) if n > 30000 else (min(n, 2048), 256 if n >= 256 else 2)
        c = vcoord.float()
        self.shift = c.mean(0)
        self.scale = (c - self.shift).norm(dim=1).max()
        self.xyz = ((c - self.shift) / self.scale)[None]
        self.rgb = ((torch.from_numpy(vcolor).float().cuda() / 255.0 - 0.5) / 0.5)[None]
        emb, patches = m.pc_encoder(self.xyz, self.rgb)
        self.emb, self.patches = emb, patches
        self.aux = AuxInputs(coords=self.xyz, features=self.rgb, centers=patches["centers"])
        self.pe = m.point_encoder.pe_layer(patches["centers"])
        self.prev = None

    @torch.no_grad()
    def segment(self, vcoord):
        m = self.model
        idx = torch.as_tensor([i for i, _ in self.order], device="cuda")
        lab = torch.as_tensor([int(p) for _, p in self.order], device="cuda")[None]
        sparse = m.point_encoder(self.xyz[:, idx], lab)
        dense = m.mask_encoder(self.prev, self.xyz, self.patches["centers"], self.patches["knn_idx"])
        masks, iou = m.mask_decoder(self.emb, self.pe, sparse, dense, aux_inputs=self.aux,
                                    multimask_output=self.prev is None)
        best = masks[0, iou[0].argmax()]
        self.prev = best[None]  # the demo's refinement: best logits feed the next click
        return best > 0


BASELINES = {"agile3d": AGILE3D, "interobject3d": InterObject3D, "pointsam": PointSAM}
