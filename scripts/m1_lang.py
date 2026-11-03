"""Round 4, M1 of results/NOVELTY.md: a set-trained language head on the frozen Chorus trunk.

Chorus' lang projector (LayerNorm 1232 -> MLP 4928 -> 1152) is trained to regress per-point SigLIP2 image features; no
text enters its training. M1 starts from those weights and trains them on *objects against names*: the mean trunk
feature of a GT instance (scripts/chorus_gt.py) against the SigLIP2 text of every training name, with a sigmoid
membership loss whose negatives are up-weighted for the confusions the error budget measured (WordNet ancestors and
siblings of the positive, names close in text space).

Open-vocabulary hygiene: ScanNet200's queried classes are split once (seed 0) into seen / held-out halves. M1 never sees
a held-out name (neither as a positive nor as a negative) nor an instance of a held-out class; model selection uses
seen classes on held-out rooms only.

Writes, into every record of each --apply DATASET:DUMPS:TRUNK (atomic rewrite), cosines over the record's labels:
  m1         trained head
  m1_frozen  Chorus' own projector on the pooled trunk (no training): separates training from pooling-then-projecting

  python scripts/m1_lang.py --gt TR/gt --split results/compare/round4/split.json \
      --apply scannet:EV/scannet_train:TR/scannet_train scannet:EV/scannet_dev:TR/scannet_dev s3dis:EV/s3dis_dev:TR/s3dis_dev
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import paths  # noqa: E402
import sets_lab as S  # noqa: E402

TEXT = "google/siglip2-so400m-patch16-512"


class Projector(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.norm = nn.LayerNorm(1232)
        self.mlp = nn.Sequential(nn.Linear(1232, 4928), nn.GELU(), nn.Linear(4928, 1152))
        self.norm.load_state_dict({"weight": sd["input_norm.weight"], "bias": sd["input_norm.bias"]})
        self.mlp[0].load_state_dict({"weight": sd["mlp_head.0.weight"], "bias": sd["mlp_head.0.bias"]})
        self.mlp[2].load_state_dict({"weight": sd["mlp_head.2.weight"], "bias": sd["mlp_head.2.bias"]})
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, x):
        return F.normalize(self.mlp(self.norm(x)), dim=-1)

    def logits(self, x, T):
        return self.logit_scale.exp() * self.forward(x) @ T.T + self.logit_bias


TEXT_CACHE = os.path.join(paths.DOWNLOADS, "runs", "trunk3", "text_siglip2.pt")


@torch.no_grad()
def text_bank(labels, mem={}):
    """SigLIP2 text embeddings per name, cached on disk (loading the model on the shared CPU takes minutes)."""
    if not mem:
        mem.update(torch.load(TEXT_CACHE) if os.path.exists(TEXT_CACHE) else {})
    missing = sorted(set(labels) - set(mem))
    if missing:
        from opensnap.views import ImageExpert
        ex = ImageExpert(TEXT, cache_dir=os.path.join(paths.DOWNLOADS, "hf", "hub"), device="cpu")
        for w, v in zip(missing, F.normalize(ex.text(missing).float().cpu(), dim=-1)):
            mem[w] = v
        torch.save(dict(mem), TEXT_CACHE)
    return torch.stack([mem[w] for w in labels])


def make_split(lab, seed=0):
    cls = sorted(lab.classes)
    perm = np.random.default_rng(seed).permutation(len(cls))
    return dict(seen=sorted(cls[i] for i in perm[: len(cls) // 2 + len(cls) % 2]),
                held=sorted(cls[i] for i in perm[len(cls) // 2 + len(cls) % 2:]), seed=seed)


def hard_weights(lab, names, w_hard, near, max_subtree=15):
    """W[i, j] = weight of name j as a negative when the positive is names[i]: ancestors of names[i], its siblings under
    a small WordNet subtree (<= max_subtree labels, so broad roots do not make everything a sibling), text neighbours."""
    ix = {w: k for k, w in enumerate(names)}
    W = torch.ones(len(names), len(names))
    small = {h for h, below in lab.wn["hyp"].items() if len(below) <= max_subtree}
    parents = {c: lab.up.get(c, set()) & small for c in names}
    sim = lab.sim["clip_l"]
    for c in names:
        i = ix[c]
        hard = set(lab.up.get(c, set())) | {h for h in names if parents[c] & parents[h]}  # all ancestors, close siblings
        hard |= {lab.L[k] for k in torch.nonzero(sim[lab.idx[c]] >= near).flatten().tolist()} if c in lab.idx else set()
        hard.add("object")  # ScanNet200's annotated catch-all class wins many confusions (bottle->object, box->object)
        for h in hard & set(names):
            if h != c:
                W[i, ix[h]] = w_hard
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--projector", default="")
    ap.add_argument("--split", required=True, help="json written on first use, then reused")
    ap.add_argument("--apply", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w_hard", type=float, default=4.0)
    ap.add_argument("--near", type=float, default=0.85)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    ev = os.path.join(paths.DOWNLOADS, "runs", "evidence3", "scannet_dev")
    lab = S.Lab("scannet", {"_": torch.load(sorted(glob.glob(os.path.join(ev, "*.pt")))[0], weights_only=False)})
    if os.path.exists(a.split):
        split = json.load(open(a.split))
    else:
        split = make_split(lab)
        json.dump(split, open(a.split, "w"), indent=1)
    held = set(split["held"])
    names = [w for w in lab.L if w not in held]  # training names: seen classes, wall/floor/ceiling, extra vocabulary
    nix = {w: k for k, w in enumerate(names)}
    T = text_bank(names)
    W = hard_weights(lab, names, a.w_hard, a.near)
    files = sorted(f for f in glob.glob(os.path.join(a.gt, "scene*.pt")))
    rng = np.random.default_rng(0)
    val_rooms = set(rng.choice(len(files), max(1, int(a.val_frac * len(files))), replace=False).tolist())
    data = {"train": ([], []), "val": ([], []), "val_held": ([], [])}
    for k, f in enumerate(files):
        g = torch.load(f, weights_only=False)
        for c, x in zip(g["classes"], g["mean"].float()):
            if c in held:
                if k in val_rooms:
                    data["val_held"][0].append(x), data["val_held"][1].append(c)
                continue
            part = "val" if k in val_rooms else "train"
            data[part][0].append(x), data[part][1].append(nix[c])
    X, y = torch.stack(data["train"][0]), torch.tensor(data["train"][1])
    Xv, yv = torch.stack(data["val"][0]), torch.tensor(data["val"][1])
    print(f"rooms {len(files)} (val {len(val_rooms)}); train instances {len(y)}, val {len(yv)}, val held-out-class "
          f"{len(data['val_held'][1])}; names {len(names)} (held out {len(held)})", flush=True)
    sd = torch.load(a.projector or os.path.join(paths.DOWNLOADS, "runs", "trunk3", "scannet_dev", "projector_lang.pt"))
    model, frozen = Projector(sd), Projector(sd).eval()
    held_names = sorted(held | set(names))  # zero-shot diagnostic: held-out instances against all names
    Th = text_bank(held_names)

    def evaluate(m):
        m.eval()
        with torch.no_grad():
            top = (m(Xv) @ T.T).argmax(1)
            acc = float((top == yv).float().mean())
            Xh, ch = data["val_held"]
            acc_h = float(np.mean([held_names[j] == c for j, c in zip((m(torch.stack(Xh)) @ Th.T).argmax(1).tolist(), ch)])) if Xh else float("nan")
            L = m.logits(Xv, T)
            Y = F.one_hot(yv, len(names)).float()
            loss = float(F.binary_cross_entropy_with_logits(L, Y, weight=torch.where(Y > 0, 1.0, W[yv])))
        m.train()
        return acc, acc_h, loss

    acc, acc_h, _ = evaluate(frozen)
    print(f"frozen projector: val top-1 seen {acc:.3f}  held-out classes (zero-shot, diagnostic) {acc_h:.3f}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    best, best_state = 1e9, None
    for ep in range(a.epochs):
        perm = torch.randperm(len(y))
        for b in range(0, len(y), 256):
            i = perm[b:b + 256]
            L = model.logits(X[i], T)
            Y = F.one_hot(y[i], len(names)).float()
            loss = F.binary_cross_entropy_with_logits(L, Y, weight=torch.where(Y > 0, 1.0, W[y[i]]), reduction="sum") / len(i)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc, acc_h, vl = evaluate(model)
        print(f"epoch {ep + 1}: val loss {vl:.4f} top-1 seen {acc:.3f}  held-out classes (diagnostic) {acc_h:.3f}", flush=True)
        if vl < best:  # selection on seen-class val loss only
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    if a.save:
        torch.save(dict(state=best_state, split=split, args=vars(a)), a.save)
    for spec in a.apply:
        ds, dumps, trunk = spec.split(":")
        for f in sorted(glob.glob(os.path.join(dumps, "*.pt"))):
            rec = torch.load(f, weights_only=False)
            t = torch.load(os.path.join(trunk, os.path.basename(f)), weights_only=False)
            x = t["mean"].float()
            Tl = text_bank(rec["labels"])
            with torch.no_grad():
                rec["m1"] = (model(x) @ Tl.T).half()
                rec["m1_frozen"] = (frozen(x) @ Tl.T).half()
            tmp = f + ".tmp"
            torch.save(rec, tmp)
            os.replace(tmp, f)
        print(f"applied to {spec}", flush=True)
    print("M1_DONE", flush=True)


if __name__ == "__main__":
    main()
