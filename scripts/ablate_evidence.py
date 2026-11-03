"""Where does the accuracy come from: proposals, image evidence, or the semantic decoding?

Rescores saved dumps (eval_standard --dump, image_terms) under the OVI-MAP protocol, with every setting fixed to the one
chosen on development scenes (tuned.json, image_dev_a.json). Proposals: OpenSNAP's automatic ones (union_l4) or one
ground-truth mask per instance (--proposals gt; an oracle, for the ablation only). Evidence / decoding rows:
  dense only, token only, 3D product of experts, image only (argmax SigLIP, OVI-MAP style), 3D + image (OpenSNAP).
ScanNet is reported on all 18 OVI-MAP scenes and on the 15 that are ScanNet val scenes.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_standard as E  # noqa: E402
import loso_image as L  # noqa: E402
import tune_image_dev as TI  # noqa: E402

R = "../downloads/runs"
SETS = {"replica": [("ours", f"{R}/dump_replica_ul4", "results/compare/replica_ul4_recs.pt"),
                    ("gt", f"{R}/dump_replica_gt", "results/compare/replica_gt_recs.pt")],
        "scannet18": [("ours", f"{R}/dump_scannet18_img", "results/compare/scannet18_img_recs.pt"),
                      ("gt", f"{R}/dump_scannet18_gt", "results/compare/scannet18_gt_recs.pt")]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/compare/tuned.json")
    ap.add_argument("--image", default="results/compare/image_dev_a.json")
    ap.add_argument("--out", default="results/compare/ablate_evidence.json")
    a = ap.parse_args()
    base = tuple(json.load(open(a.base))["chosen"][:4])
    w, mix, t_img, gamma = json.load(open(a.image))["chosen"]
    rows = {"dense only": ((1.0, *base[1:]), (0.0, mix, t_img, gamma)),
            "token only": ((0.0, *base[1:]), (0.0, mix, t_img, gamma)),
            "3D experts (geometry only)": (base, (0.0, mix, t_img, gamma)),
            "image only (argmax SigLIP)": (base, (1.0, mix, t_img, gamma)),
            "3D + image (OpenSNAP)": (base, (w, mix, t_img, gamma)),
            "image only, SigLIP-L (OVI-MAP's encoder)": (base, (1.0, "L", t_img, gamma)),
            "3D + image, SigLIP-L (OVI-MAP's encoder)": (base, (w, "L", t_img, gamma))}
    labels, ids, _, inst = E.benchmark("scannet200")
    v = json.load(open(os.path.join(L.VOCAB, "replica51.json")))
    valid = {"replica": [v["labels"].index(l) + 1 for l in v["instance_labels"]], "scannet18": inst["200"]}
    out = dict(base=list(base), image=[w, mix, t_img, gamma], results={})
    for bench, specs in SETS.items():
        for prop, dump, recs in specs:
            if not os.path.isdir(dump.rstrip("/") + "_img"):
                print("missing", dump, flush=True)
                continue
            scenes, D, I, G, M = TI.load(dump, recs)
            subsets = {"all": scenes}
            if bench == "scannet18":
                subsets["val15"] = [s for s in scenes if not s.startswith("scene0168")]
            TI.SH.update(G=G, M=M, valid=valid[bench])
            pool = mp.get_context("fork").Pool(12)
            for sub, sc in subsets.items():
                for name, (b, g) in rows.items():
                    r = TI.score(sc, D, I, G, M, b, g, valid[bench], pool=pool)
                    r = {k: r[k] for k in ("mIoU", "mAcc", "AP25", "AP50", "AP")}
                    out["results"].setdefault(f"{bench}/{sub}", {}).setdefault(prop, {})[name] = r
                    print(f"{bench}/{sub} {prop:4s} {name:28s}", " ".join(f"{k} {100 * x:.1f}" for k, x in r.items()), flush=True)
            pool.close()
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
