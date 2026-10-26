"""Regenerate results/TABLES.md from the JSON result files (every number traceable to a file)."""
import json
import os

import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
out = []


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x):
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def table(title, src, header, rows, note=""):
    out.append(f"### {title}\n\n_Source: {src}_\n")
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    out.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    if note:
        out.append(f"\n{note}")
    out.append("")


def pending(title, what):
    out.append(f"### {title}\n\n**Pending:** {what}\n")


def semseg():
    names = {"openscene": "OpenScene 3D (point-wise)", "snap_token": "Segmenter text token", "snap_token_relay": "+ vocabulary relay",
             "mask_pool": "Mask-anchored pooling", "mask_pool_dense": "+ dense blend", "opensnap_norelay": "OpenSNAP w/o relay",
             "opensnap": "**OpenSNAP (full)**"}
    sc, mp = load("scannet_val_xyz.json"), load("matterport160_test_xyz.json")
    if sc and mp:
        rows = [[n, pct(sc["results"][k]["mIoU"]), pct(sc["results"][k]["mAcc"]), pct(mp["results"][k]["mIoU"]),
                 pct(mp["results"][k]["mIoU_seen"]), pct(mp["results"][k]["mIoU_unseen"])] for k, n in names.items()]
        table("T1/T2 · Open-vocabulary 3D semantic segmentation and component ablation",
              "scannet_val_xyz.json (312 scenes), matterport160_test_xyz.json (406 regions)",
              ["method", "ScanNet mIoU", "ScanNet mAcc", "MP-160 mIoU", "MP-160 seen", "MP-160 unseen"], rows,
              "Unseen = the 63 Matterport-160 classes absent from the segmenter's training vocabulary. XYZ-only segmenter input.")
    os_ = load("scannet_val_xyz_openseg.json")
    if os_ and sc:
        table("T15 · Dense-feature source (ScanNet val mIoU)", "scannet_val_xyz.json vs scannet_val_xyz_openseg.json",
              ["method", "LSeg (ScanNet-trained)", "OpenSeg (Matterport-trained)"],
              [[n, pct(sc["results"][k]["mIoU"]), pct(os_["results"][k]["mIoU"])] for k, n in names.items() if k in os_["results"]])
    else:
        pending("T15 · Dense-feature source", "scannet_val_xyz_openseg.json")

    nu, nuo = load("nuscenes_val_xyz.json"), load("nuscenes_official_val.json")
    if nu or nuo:
        cell = lambda d, k: [pct(d["results"][k]["mIoU"]), pct(d["results"][k]["mAcc"])] if d and k in d["results"] else ["–", "–"]
        rows = [[n] + cell(nuo, k) + cell(nu, k) for k, n in names.items()]
        rows.append(["OpenScene 3D distill (published)", "36.7", "–", "", ""])
        rows.append(["OpenScene 2D-3D ensemble (published)", "42.1", "–", "", ""])
        table("T2b · Open-vocabulary LiDAR semantic segmentation, nuScenes val (16 classes)", "nuscenes_official_val.json, nuscenes_val_xyz.json",
              ["method", "mIoU (official, 6019 keyframes)", "mAcc", "mIoU (OpenScene-processed, 1003)", "mAcc"], rows,
              "Segmenter in Outdoor mode (XYZ-only, 5 cm grid); dense branch = OpenScene nuScenes OpenSeg 3D model. Official = single "
              "LiDAR keyframes with lidarseg labels (SNAP's 16-class map); OpenScene-processed = every 6th sweep of OpenScene's release. "
              "Published OpenScene numbers use the authors' script.")

    sk = load("semantickitti_val.json")
    if sk:
        rows = [[n, pct(sk["results"][k]["mIoU"]), pct(sk["results"][k]["mAcc"])] for k, n in names.items() if k in sk["results"]]
        table(f"T2c · Open-vocabulary LiDAR semantic segmentation, SemanticKITTI seq 08 (19 classes, {sk['num_scenes']} scans)",
              "semantickitti_val.json", ["method", "mIoU", "mAcc"], rows,
              "Dense branch = OpenScene's nuScenes model applied cross-dataset (OpenScene has no KITTI model), so the dense "
              "features are zero-shot here; the segmenter saw SemanticKITTI train. The weak cross-dataset dense branch (terrain 0.0, "
              "parking 0.1 IoU) costs mIoU on stuff classes; OpenSNAP gains on objects (other vehicle 2.9 -> 11.9) and in mAcc.")

    uf = load("underonefacade_test.json")
    if uf:
        lofg2 = np.array([0, 1, 1, 0, 2, 2, 0, 0, 0, 3, 3, 4, 1, 4, 4])  # authors' LoFG3 -> LoFG2 map

        def scores(c, coarse=False):
            c = np.asarray(c, np.float64)
            if coarse:
                c2 = np.zeros((5, 5))
                np.add.at(c2, (lofg2[:, None], lofg2[None, :]), c)
                c = c2
            tp, g, p = np.diag(c), c.sum(0), c.sum(1)
            pr = g > 0
            return tp.sum() / c.sum(), np.mean((tp / np.maximum(g + p - tp, 1))[pr]), np.mean((2 * tp / np.maximum(g + p, 1))[pr])

        europe = lambda s: s.startswith(("DEBY", "Lenton"))
        part = lambda k, eu: sum(np.array(s["conf"][k], np.float64) for s in uf["per_scene"] if europe(s["scene"]) == eu)
        rows = []
        for k, n in names.items():
            c = uf["results"][k]["conf"]
            (oa3, m3, f3), (oa2, m2, _) = scores(c), scores(c, True)
            rows.append([n, pct(m3), pct(f3), pct(oa3), pct(m2), pct(oa2), pct(scores(part(k, False))[1]),
                         pct(scores(part(k, True))[1]), pct(uf["results"][k]["mIoU_seen"]), pct(uf["results"][k]["mIoU_unseen"])])
        rows.append(["Supervised, other continent (published, xyz)", "6.6–14.8 / 9.4–23.6", "–", "32.1–42.6 / 33.9–51.9",
                     "25.7–40.8 / 28.8–40.5", "48.6–60.9 / 48.9–65.8", "6.6–14.8", "9.4–23.6", "", ""])
        rows.append(["Supervised, same data (published, xyz, DGCNN)", "33.4", "–", "62.7", "–", "–", "–", "–", "", ""])
        table(f"T2d · Zero-shot facade semantic segmentation, UnderOneFacade test ({uf['num_scenes']} facades, TLS/MLS, UK/DE/SG)",
              "underonefacade_test.json",
              ["method", "LoFG3 mIoU", "LoFG3 mF1", "LoFG3 OA", "LoFG2 mIoU", "LoFG2 OA", "Asia LoFG3", "Europe LoFG3",
               "seen", "unseen"], rows,
              "No facade data in training. Segmenter in Outdoor mode (selected on 22 validation facades among Outdoor/Indoor/"
              "Aerial x nuScenes/ScanNet/Matterport dense models); dense branch = OpenScene nuScenes model, cross-dataset. Input "
              "voxelized at 5 cm (7.8 cm for one 36M-point facade); every raw point is scored with its voxel's prediction. Raw "
              "label 16 (Singapore) -> other, as in the authors' loader. Seen = 8 class names in the segmenter's training "
              "vocabulary (wall, window, door, decoration, column, stairs, terrain, blinds). Published rows (Wang et al., ECCV "
              "2026, Tables 2 and 6) are trained on facades: 'other continent' = Europe->Asia / Asia->Europe ranges over six "
              "architectures, evaluated on the authors' target-continent test sets (Europe includes ZAHA scans not in this "
              "release), so they bracket rather than match our split. Facade parts lie on one surface, which the object-level "
              "masks do not separate: wall 44 IoU, window/balcony/molding 0.")


def facade_compare():
    """UnderOneFacade: OpenSNAP against other zero-shot ideas (same split, same scoring)."""
    uf = load("underonefacade_test.json")
    lofg2 = np.array([0, 1, 1, 0, 2, 2, 0, 0, 0, 3, 3, 4, 1, 4, 4])

    def sc(c, coarse=False):
        c = np.asarray(c, np.float64)
        if coarse:
            c2 = np.zeros((5, 5))
            np.add.at(c2, (lofg2[:, None], lofg2[None, :]), c)
            c = c2
        tp, g, p = np.diag(c), c.sum(0), c.sum(1)
        pr = g > 0
        return [np.mean((tp / np.maximum(g + p - tp, 1))[pr]), np.mean((2 * tp / np.maximum(g + p, 1))[pr]), tp.sum() / c.sum()]

    rows = []
    entries = [("OpenScene 3D, point-wise (dense 3D CLIP)", uf, "openscene"),
               ("**OpenSNAP: mask-anchored pooling**", uf, "mask_pool"), ("OpenSNAP (full)", uf, "opensnap"),
               ("OpenShape on the same masks (3D shape-text)", load("aux/facade_openshape.json"), "openshape"),
               ("Render-then-lift: Grounded-SAM, RGB renders", load("aux/facade_rl_gsam.json"), "gsam"),
               ("Render-then-lift: Grounded-SAM, geometry renders", load("aux/facade_rl_gsam_geom.json"), "gsam"),
               ("Render-then-lift: SAM + SigLIP, RGB renders", load("aux/facade_rl_sam_siglip.json"), "sam_siglip")]
    for name, d, k in entries:
        if d and k in d["results"]:
            c = d["results"][k]["conf"]
            rows.append([name] + [pct(x) for x in sc(c)] + [pct(sc(c, True)[0])])
        else:
            rows.append([name, "pending", "", "", ""])
    if uf:
        table("T2e · Zero-shot facade semantic segmentation against other zero-shot ideas (UnderOneFacade test, 17 facades)",
              "underonefacade_test.json, aux/facade_openshape.json, aux/facade_rl_*.json",
              ["method", "LoFG3 mIoU", "LoFG3 mF1", "LoFG3 OA", "LoFG2 mIoU"], rows,
              "All methods zero-shot on facades and scored on every raw point. OpenShape (spconv-all, OpenCLIP ViT-bigG) labels "
              "exactly the segmenter proposals OpenSNAP pools over. Render-then-lift: virtual street-side cameras (frontal, "
              "+-35 deg obliques, ~10 x 8 m tiles), 1024 px z-buffered splats, 2D labels lifted through the z-buffer and summed "
              "over views, unseen points take the nearest seen point's label (scripts/facade_render_lift.py); RGB renders use "
              "real colour for UK/Singapore and shaded geometry for Germany (no colour). OpenSNAP and OpenScene use XYZ only.")

    reg = lambda s: "Germany" if s.startswith("DEBY") else ("UK" if s.startswith("Lenton") else "Singapore")
    methods = [("pn_facade_snap", "SNAP (negatives ignored)", "XYZ"), ("pn_facade_v5mttt", "**OpenSNAP: decoder + token tuning**", "XYZ"),
               ("pn_facade_interobject3d", "InterObject3D (ICRA'23, ScanNet)", "XYZ+RGB"),
               ("pn_facade_agile3d", "AGILE3D (ICLR'24, ScanNet)", "XYZ+RGB"), ("pn_facade_agile3d_xyz", "AGILE3D, colour zeroed", "XYZ"),
               ("pn_facade_pointsam", "Point-SAM (ViT-L, part + scene mixture)", "XYZ+RGB"),
               ("pn_facade_pointsam_xyz", "Point-SAM, colour zeroed", "XYZ")]
    rows = []
    for f, name, inp in methods:
        d = load(f"aux/{f}.json")
        if not d:
            rows.append([name, inp, "pending"] + [""] * 7)
            continue
        r = d["results"]
        by = {g: np.array([x["ious"] for x in d["records"] if reg(x["scene"]) == g]) for g in ("Germany", "UK", "Singapore")}
        rows.append([name, inp, pct(r["IoU@1"]), pct(r["IoU@3"]), pct(r["IoU@5"]), pct(r["IoU@10"]), f"{r['NoC@90']:.2f}"]
                    + [f"{pct(v[:, 4].mean())} / {pct(v[:, 9].mean())}" for v in by.values()])
    table("T5l · Corrective clicking on facade parts against published interactive segmenters (UnderOneFacade test)",
          "aux/pn_facade_*.json",
          ["method", "input", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90", "Germany IoU@5/10", "UK IoU@5/10", "Singapore IoU@5/10"], rows,
          "1,603 parts (3D-connected components of window/door/balcony/molding/decoration/column/arch/stairs/blinds labels), "
          "3 m crops; identical crops, voxels, click sequences (centre of the larger error region) and ground truth for every "
          "method (scripts/click_baselines.py wraps the released weights with each method's own grid, colour scaling and click "
          "encoding). German scans have no colour, so the Germany column is XYZ-only for every method. None of the methods "
          "saw facades; Point-SAM was trained on a mixture of object-part (PartNet, Fusion360, ShapeNet) and scene (ScanNet) masks (configs/dataset/mixture.yaml).")


def facade_search():
    fs = load("facade_search.json")
    if not fs:
        return pending("T3f · Facade search", "facade_search.json")
    r = fs["results"]
    build = {"opensnap": ("underonefacade_test.json", None), "gsam": ("aux/facade_rl_gsam.json", None),
             "sam_siglip": ("aux/facade_rl_sam_siglip.json", None), "openshape": ("aux/facade_openshape.json", None)}
    sec = {k: np.mean([x["sec"] for x in load(f)["per_scene"]]) for k, (f, _) in build.items() if load(f)}
    names = [("snap_native", "Segmenter token only (OVQL engine)", "opensnap"), ("opensnap", "**OpenSNAP (OVQL engine)**", "opensnap"),
             ("opensnap_refine", "OpenSNAP + cracking", "opensnap"), ("openscene_cluster", "OpenScene 3D + connected components", "opensnap"),
             ("openshape", "OpenShape on the segmenter's masks", "openshape"),
             ("gsam", "Render-then-lift: Grounded-SAM", "gsam"), ("sam_siglip", "Render-then-lift: SAM + SigLIP", "sam_siglip")]
    rows = []
    for k, n, b in names:
        if f"{k}/IS" not in r:
            continue
        i, nr = r[f"{k}/IS"], r.get(f"{k}/NEAR", {})
        reg = [pct(r.get(f"{k}/IS/{g}", {}).get("AP25")) for g in ("Germany", "UK", "Singapore")]
        rows.append([n, pct(i["AP"]), pct(i["AP50"]), pct(i["AP25"]), pct(nr.get("AP50")), pct(nr.get("AP25"))] + reg
                    + [f"{i['ms']:.1f}", f"{sec[b]:.0f}" if b in sec else "–"])
    pc = r["gsam/IS"]["per_class_AP25"]
    table("T3f · Comparative search on facades: concept (IS) and relational (IS AND NEAR 0.5 m) queries, instance AP "
          "(UnderOneFacade test, 1,606 part instances)", "facade_search.json",
          ["system", "IS AP", "IS AP50", "IS AP25", "NEAR AP50", "NEAR AP25", "Germany IS AP25", "UK IS AP25", "Singapore IS AP25",
           "ms / query", "build s / facade"], rows,
          "Same queries for every system (each element class present in a facade; up to 5 sampled NEAR pairs per facade), "
          "scored at point level against 3D-connected components of the part labels. All systems zero-shot. Baselines answer "
          "NEAR by composing their IS answers (bbox gap <= 0.5 m, score product); OpenSNAP uses its engine. Render-then-lift "
          "instances = 2D masks lifted through a z-buffer and merged across views (point IoU > 0.3, same label). Build = "
          "one-off ingest per facade (OpenSNAP: segmenter + dense features + relation; render-then-lift: rendering + 2D "
          "models + lifting); ms / query excludes build. Grounded-SAM per-class AP25: "
          + ", ".join(f"{c} {pct(v)}" for c, v in pc.items()) + ". No system localises facade parts reliably; "
          "render-then-lift does so only on photo-like UK colour renders.")


def queries():
    eng = {"snap_native": "Segmenter token only", "opensnap": "**OpenSNAP**", "opensnap_refine": "OpenSNAP + cracking"}
    s3, sn, s3a = load("s3dis_queries.json"), load("scannet_queries.json"), load("s3dis_queries_allclasses.json")
    if s3 and sn:
        rows = []
        for k, n in eng.items():
            a, b = s3["results"], sn["results"]
            rows.append([n, pct(a[f"{k}/IS"]["AP"]), pct(a[f"{k}/IS"]["AP50"]), pct(a[f"{k}/NEAR"]["AP"]),
                         pct(b[f"{k}/IS"]["AP"]), pct(b[f"{k}/IS"]["AP_head"]), pct(b[f"{k}/IS"]["AP_common"]),
                         pct(b[f"{k}/IS"]["AP_tail"]), pct(b[f"{k}/NEAR"]["AP"]), f"{b[f'{k}/IS']['ms']:.1f}"])
        table("T3 · OVQL query workloads (instance AP, IoU 0.5:0.95)", "s3dis_queries.json (68 rooms, object classes), scannet_queries.json (156 val scenes)",
              ["engine", "S3DIS IS AP", "S3DIS IS AP50", "S3DIS NEAR AP", "SN200 IS AP", "head", "common", "tail", "SN200 NEAR AP", "ms/query"], rows,
              "The segmenter's text head was trained on all 200 ScanNet200 classes (in-domain, incl. tail).")
    c2f = load("s3dis_queries_allclasses_c2f.json")
    if s3a:
        rows = [[n, pct(s3a["results"][f"{k}/IS"]["AP"]), pct(s3a["results"][f"{k}/IS"]["AP50"]), pct(s3a["results"][f"{k}/IS"]["AP25"])]
                + ([pct(c2f["results"][f"{k}/IS"]["AP"]), pct(c2f["results"][f"{k}/IS"]["AP50"]), pct(c2f["results"][f"{k}/IS"]["AP25"])] if c2f else ["–"] * 3)
                for k, n in eng.items()]
        rows.append(["SNAP-C@auto (published, Table 4)", "16.1", "23.8", "30.9", "", "", ""])
        table("T4 · Open-vocabulary instance segmentation on S3DIS Area 5 (all 13 classes)",
              "s3dis_queries_allclasses.json (one-shot proposals), s3dis_queries_allclasses_c2f.json (coarse-to-fine)",
              ["engine", "AP (FPS)", "AP50 (FPS)", "AP25 (FPS)", "AP (c2f)", "AP50 (c2f)", "AP25 (c2f)"], rows,
              "Published numbers use the authors' own evaluation; ours use one protocol for all engines.")

    rf, rc = load("replica_queries.json"), load("replica_queries_c2f.json")
    if rf:
        names = {**eng, "opensnap_refine_v4": "OpenSNAP + cracking (v4 refiner)"}
        cell = lambda d, k: [pct(d["results"][f"{k}/IS"][m]) for m in ("AP", "AP50", "AP25")] if d and f"{k}/IS" in d["results"] else ["–"] * 3
        rows = [[n] + cell(rf, k) + cell(rc, k) for k, n in names.items()]
        rows += [["SNAP-C@auto (published, Table 4)", "10.1", "11.7", "13.8", "", "", ""],
                 ["OpenMask3D (published)", "13.1", "18.4", "24.2", "", "", ""]]
        table("T4b · Open-vocabulary instance segmentation on Replica (8 scenes)", "replica_queries.json, replica_queries_c2f.json",
              ["engine", "AP (FPS)", "AP50 (FPS)", "AP25 (FPS)", "AP (c2f)", "AP50 (c2f)", "AP25 (c2f)"], rows,
              "Instances transferred from Replica habitat meshes (exact alignment); all object classes except structure/undefined. "
              "Published numbers use the OpenMask3D 51-class protocol.")

    nq = load("nuscenes_queries.json")
    if nq:
        r = nq["results"]
        rows = [[n, pct(r[f"{k}/IS"]["AP"]), pct(r[f"{k}/IS"]["AP50"]), pct(r[f"{k}/IS"]["AP25"]), pct(r[f"{k}/NEAR"]["AP"]),
                 f"{r[f'{k}/IS']['ms']:.1f}"] for k, n in eng.items() if f"{k}/IS" in r]
        table("T4c · Open-vocabulary instance segmentation on nuScenes LiDAR (10 thing classes)", "nuscenes_queries.json",
              ["engine", "AP", "AP50", "AP25", "NEAR AP", "ms/query"], rows,
              "Instances = lidarseg points inside annotated 3D boxes of the same category (how nuScenes-panoptic is built); "
              "every 4th val keyframe; objects with >= 20 voxels.")

    kq = load("semantickitti_queries.json")
    if kq:
        r = kq["results"]
        rows = [[n, pct(r[f"{k}/IS"]["AP"]), pct(r[f"{k}/IS"]["AP50"]), pct(r[f"{k}/IS"]["AP25"]), pct(r[f"{k}/NEAR"]["AP"]),
                 f"{r[f'{k}/IS']['ms']:.1f}"] for k, n in eng.items() if f"{k}/IS" in r]
        table("T4d · Open-vocabulary instance segmentation on SemanticKITTI seq 08 (8 thing classes)", "semantickitti_queries.json",
              ["engine", "AP", "AP50", "AP25", "NEAR AP", "ms/query"], rows,
              "Panoptic instance labels; every 4th scan; objects with >= 20 voxels.")


def refiner():
    q, sy = load("s3dis_queries_v4.json"), load("systems_s3dis_v4.json")
    if q:
        names = {"opensnap": "OpenSNAP (no cracking)", "opensnap_refine": "cracking, click-oracle refiner",
                 "opensnap_refine_v4": "**cracking, v4 text-decoder refiner (dedup 0.5)**", "opensnap_refine_v4_nodedup": "cracking, v4 refiner, no dedup"}
        r = q["results"]
        rows = [[n, pct(r[f"{k}/IS"]["AP"]), pct(r[f"{k}/IS"]["AP50"]), pct(r[f"{k}/NEAR"]["AP"]), f"{r[f'{k}/IS'].get('refined', 0):.2f}",
                 f"{r[f'{k}/IS']['ms']:.1f}"] for k, n in names.items() if f"{k}/IS" in r]
        table("T12b · Query-time refiner: click oracle vs trained text decoder (S3DIS Area 5 queries)", "s3dis_queries_v4.json",
              ["engine", "IS AP", "IS AP50", "NEAR AP", "refined/room", "IS ms"], rows)
    if sy:
        s = sy["summary"]
        rows = [[m, f"{s[m]['mean_ms']:.1f}", f"{s[m]['second_pass_ms']:.1f}", pct(s[m]["ap50"]), pct(s[m]["synonym_ap50"])]
                for m in ("no_refine", "refine_discard", "cracking")]
        table("T12c · Cracking workload with the v4 text-decoder refiner (S3DIS, 34 rooms)", "systems_s3dis_v4.json",
              ["mode", "mean ms", "2nd-pass ms", "AP50", "synonym AP50"], rows)


def interactive():
    def get(name, sub="results"):
        d = load(name)
        return d[sub] if d else None
    full = get("s3dis_interactive_snap_zeroshot_xyz.json")
    rows = []
    if full:
        rows.append(["SNAP-C (released, our protocol)", "full", pct(full["IoU@1"]), pct(full["IoU@3"]), pct(full["IoU@5"]), pct(full["IoU@10"]), f"{full['NoC@90']:.2f}"])
    rows.append(["SNAP-C (published)", "full", "53.6", "71.1", "77.6", "83.2", "–"])
    for name, label in [("aux/s3dis_inter_snapft_4k.json", "decoder fine-tune on ScanNet (v1 control, 4k)"),
                        ("aux/s3dis_inter_plus_4k.json", "SNAP+ v1 (sem in features, min-loss tokens, 4k)")]:
        d = get(name)
        if d:
            rows.append([label, "full", pct(d["IoU@1"]), pct(d["IoU@3"]), pct(d["IoU@5"]), pct(d["IoU@10"]), f"{d['NoC@90']:.2f}"])
    for name, label in [("aux/diag_snap.json", "SNAP-C"), ("aux/curve_v2_control_4000.json", "v2 control (4k)"),
                        ("aux/curve_v2_plus_4000.json", "SNAP+ v2 (sem keys, role tokens, 4k)"),
                        ("aux/curve_v3_plus_4000_fixed.json", "SNAP+ v3 (frozen segmenter, 4k)")]:
        d = get(name)
        if d:
            rows.append([label, "subset", pct(d["IoU@1"]), pct(d["IoU@3"]), pct(d["IoU@5"]), pct(d["IoU@10"]), f"{d['NoC@90']:.2f}"])
    table("T5 · Interactive segmentation, S3DIS Area 5 (zero-shot)", "s3dis_interactive_*.json, aux/*",
          ["model", "rooms", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], rows,
          "Subset = every 4th room (17 rooms). XYZ-only input; first click at object centre, then centre of the missed region.")
    c = get("s3dis_interactive_crops.json")
    if c:
        rows = [["SNAP-C (released, our run)", pct(c["IoU@1"]), pct(c["IoU@3"]), pct(c["IoU@5"]), f"{c['NoC@90']:.2f}", c["objects"]],
                ["SNAP-C (published)", "56.6", "73.8", "80.9", "–", "–"], ["AGILE3D (published)", "58.7", "77.4", "83.6", "–", "–"],
                ["Point-SAM (published)", "45.9", "77.6", "84.6", "–", "–"]]
        table("T5c · Interactive segmentation on S3DIS Area 5 object crops (SNAP Table 2 protocol)", "s3dis_interactive_crops.json",
              ["model", "IoU@1", "IoU@3", "IoU@5", "NoC@90", "objects"], rows,
              "±1.5 m crop around each object. OpenSNAP's click path is SNAP-exact, so its row equals the SNAP-C run.")
    n = get("nuscenes_interactive.json")
    if n:
        table("T5d · Interactive segmentation on nuScenes LiDAR val (Outdoor)", "nuscenes_interactive.json",
              ["model", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@80", "NoC@90", "objects"],
              [["SNAP-C (released, our run) = OpenSNAP click path", pct(n["IoU@1"]), pct(n["IoU@3"]), pct(n["IoU@5"]), pct(n["IoU@10"]),
                f"{n['NoC@80']:.2f}", f"{n['NoC@90']:.2f}", n["objects"]]],
              "Every 10th val keyframe; things = box-derived instances (>= 10 points), stuff = one segment per class (as in nuScenes-panoptic).")
    k = get("semantickitti_interactive.json")
    if k:
        table("T5e · Interactive segmentation on SemanticKITTI seq 08 (Outdoor)", "semantickitti_interactive.json",
              ["model", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@80", "NoC@90", "objects"],
              [["SNAP-C (released, our run) = OpenSNAP click path", pct(k["IoU@1"]), pct(k["IoU@3"]), pct(k["IoU@5"]), pct(k["IoU@10"]),
                f"{k['NoC@80']:.2f}", f"{k['NoC@90']:.2f}", k["objects"]]],
              "Every 10th scan; things = panoptic instances (>= 10 points), stuff = one segment per class.")
    sets = [("scannet_dev", "ScanNet val (dev subset)", None), ("s3dis_crops", "S3DIS crops", "s3dis_interactive_crops.json"), ("s3dis_full", "S3DIS rooms", "s3dis_interactive_snap_zeroshot_xyz.json"),
            ("replica", "Replica", None), ("nuscenes", "nuScenes", "nuscenes_interactive.json"),
            ("kitti", "SemanticKITTI", "semantickitti_interactive.json"),
            ("facade", "UnderOneFacade (facade parts, zero-shot)", "underonefacade_interactive.json")]
    methods = [("snap", "SNAP, corrective protocol (negatives ignored)"), ("subtract", "+ negative-mask subtraction"),
               ("ttt", "**OpenSNAP: test-time token tuning**"), ("v5", "OpenSNAP: corrective decoder (ScanNet-trained)"), ("v5m", "OpenSNAP: corrective decoder (multi-domain)"),
               ("v5ttt", "OpenSNAP: ScanNet decoder + token tuning"), ("v5mttt", "OpenSNAP: multi-domain decoder + token tuning")]
    rows = []
    for key, name, pos_file in sets:
        if pos_file and get(pos_file):
            r = get(pos_file)
            rows.append([name, "SNAP, positive clicks only", pct(r["IoU@1"]), pct(r["IoU@3"]), pct(r["IoU@5"]), pct(r["IoU@10"]), f"{r['NoC@90']:.2f}"])
        for m, label in methods:
            r = get(f"aux/pn_{key}_{m}.json") or (get(f"aux/s3dis_crops_pn_{m}.json") if key == "s3dis_crops" and m in ("snap", "subtract") else None)
            if r:
                rows.append([name, label, pct(r["IoU@1"]), pct(r["IoU@3"]), pct(r["IoU@5"]), pct(r["IoU@10"]), f"{r['NoC@90']:.2f}"])
        if key == "s3dis_crops":
            rows += [[name, "AGILE3D (published)", "58.7", "77.4", "83.6", "–", "–"], [name, "Point-SAM (published)", "45.9", "77.6", "84.6", "–", "–"]]
    if rows:
        table("T5f · Corrective clicking with positive and negative clicks", "aux/pn_*.json",
              ["dataset", "method", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], rows,
              "Corrective protocol: every click goes to the centre of the larger error region (positive on misses, negative on "
              "false positives). SNAP cannot use negative clicks. Token tuning: 30 Adam steps on the mask token over SNAP's frozen "
              "voxel embeddings, hyper-parameters chosen on ScanNet val only (aux/sweep_ttt_scannet_*.json). UnderOneFacade: "
              "test split, instances = 3D-connected components of window/door/balcony/molding/decoration/column/arch/stairs/"
              "blinds labels at 5 cm (>= 100 voxels; scripts/prep_facade_instances.py), 3 m crops.")
    row = lambda r: [pct(r.get(k)) for k in ("IoU@1", "IoU@3", "IoU@5", "IoU@10")] + ([f"{r['NoC@90']:.2f}"] if "IoU@10" in r else ["–"])
    pairs = [("S3DIS rooms", "s3dis_interactive_snap_zeroshot_xyz.json", "aux/pn_posonly_s3dis_full_v6.json", "held out"),
             ("S3DIS crops", "s3dis_interactive_crops.json", "aux/pn_posonly_s3dis_crops_v6.json", "held out"),
             ("nuScenes", "nuscenes_interactive.json", "aux/pn_posonly_nuscenes_v6.json", "train split seen"),
             ("SemanticKITTI", "semantickitti_interactive.json", "aux/pn_posonly_kitti_v6.json", "train split seen")]
    rows = []
    for name, base, ad, note in pairs:
        if get(base) and get(ad):
            rows += [[name, note, "SNAP"] + row(get(base)), [name, note, "**+ backbone LoRA adapters**"] + row(get(ad))]
    if rows:
        table("T5i · Backbone adaptation (LoRA in every PTv3 attention layer), positive clicks only", "aux/pn_posonly_*_v6.json",
              ["dataset", "adapter training data", "model", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], rows,
              "Adapters (rank 16, zero-init, 6k steps) trained jointly with the corrective decoder on ScanNet + nuScenes-train + "
              "SemanticKITTI-train with feature distillation to the frozen backbone; decoder for positive clicks is SNAP's own.")
    rows = []
    for name, key in [("S3DIS crops", "s3dis_crops"), ("S3DIS rooms", "s3dis_full"), ("Replica", "replica"), ("nuScenes", "nuscenes"),
                      ("SemanticKITTI", "kitti")]:
        for m, label in [("snap", "SNAP (negatives ignored)"), ("v6snap", "SNAP decoder + adapters"), ("ttt", "token tuning"),
                         ("v5mttt", "multi-domain decoder + token tuning"), ("v6ttt", "adapters + decoder + token tuning")]:
            r = get(f"aux/pn_{key}_{m}.json") or (get(f"aux/s3dis_crops_pn_{m}.json") if key == "s3dis_crops" and m == "snap" else None)
            if r:
                rows.append([name, label] + row(r))
    if rows:
        table("T5j · Corrective clicking with the adapted backbone", "aux/pn_*_v6*.json",
              ["dataset", "method", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], rows,
              "Adapters help where their training domains are (nuScenes first click +3.6) and are neutral-to-slightly-negative on "
              "held-out indoor data; token tuning alone remains best on indoor rooms.")
    rows = []
    for name, a, b in [("ScanNet val dev (selection set)", "aux/sweep_ttt_scannet_s30_r10_rad0.json", "aux/sweep_tta_scannet_k8_r0.1.json"),
                       ("S3DIS crops (held out)", "aux/pn_s3dis_crops_ttt.json", "aux/pn_s3dis_crops_tttj.json"),
                       ("nuScenes (held out)", "aux/pn_nuscenes_ttt.json", "aux/pn_nuscenes_tttj.json")]:
        if get(a) and get(b):
            ra = dict(get(a)) if "dev" not in name else {k: get(a)[k] for k in ("IoU@1", "IoU@3")}
            rows += [[name, "token tuning"] + row(ra), [name, "**+ click-jitter averaging (8 × 10 cm)**"] + row(get(b))]
    for r in ("0.05", "0.2"):
        x = get(f"aux/sweep_tta_scannet_k8_r{r}.json")
        if x:
            rows.append(["ScanNet val dev (selection set)", f"jitter radius {r} m"] + row(x))
    if rows:
        table("T5k · Click-jitter test-time augmentation for the first click", "aux/sweep_tta_*.json, aux/pn_*_tttj.json",
              ["dataset", "method", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], rows,
              "Dev sweep ran 3 clicks only. Radius chosen on ScanNet; helps indoors (+1.1 / +1.5 IoU@1), no effect on sparse outdoor LiDAR.")
    dc = load("aux/dense_collapse.json")
    if dc:
        bench = [r for r in dc["rows"] if r["source"] == "benchmark scan"]
        video = [r for r in dc["rows"] if r["source"] != "benchmark scan"]
        rows = [[f"{len(bench)} benchmark scans (ScanNet, S3DIS, Replica, Matterport)", f"{min(r['top_object_label_share'] for r in bench):.2f}–{max(r['top_object_label_share'] for r in bench):.2f}", "no"]]
        rows += [[f"{r['scene']} ({r['source']})", f"{r['top_object_label_share']:.2f} ('{r['label']}')", "**yes**"] for r in video]
        table("T18 · Dense-feature collapse guard (out-of-distribution scenes)", "aux/dense_collapse.json",
              ["scene", "top non-structural label share", f"guard fires (> {dc['threshold']})"], rows,
              "When it fires, OpenSNAP drops the dense 3D CLIP features and uses the segmenter's own tokens for relevance and naming.")
    lat = load("aux/click_latency.json")
    if lat:
        names = {"snap": "SNAP (negatives ignored)", "ttt": "token tuning (30 steps)", "v5": "corrective decoder"}
        table(f"T5h · Per-click latency with negative clicks (S3DIS rooms, {lat['mean_voxels'] / 1e3:.0f}k voxels)", "aux/click_latency.json",
              ["method", "median ms", "p90 ms", "clicks"], [[n, f"{lat[k]['median_ms']:.0f}", f"{lat[k]['p90_ms']:.0f}", lat[k]["n"]] for k, n in names.items()],
              "Backbone encoding is done once per scene and excluded; shared MIG 3g.40gb slice.")
    sw = []
    for st, rg, rad in [(30, 0.3, 0), (30, 1, 0), (30, 3, 0), (30, 10, 0), (30, 30, 0), (100, 1, 0), (30, 1, 0.05), (100, 0.3, 0.05)]:
        r = get(f"aux/sweep_ttt_scannet_s{st}_r{rg}_rad{rad}.json")
        if r:
            sw.append([st, rg, rad, pct(r["IoU@1"]), pct(r["IoU@3"]), pct(r["IoU@5"]), pct(r["IoU@10"]), f"{r['NoC@90']:.2f}"])
    if sw:
        table("T5g · Token-tuning hyper-parameters (ScanNet val crops, every 8th scene; selection set)", "aux/sweep_ttt_scannet_*.json",
              ["steps", "reg", "label radius (m)", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], sw,
              "Selected: 30 steps, reg 10, click-only labels; used unchanged on every other dataset.")
    diag = [("aux/diag_plus_iou.json", "IoU-head selection"), ("aux/diag_plus_oracle.json", "oracle best-of-K"),
            ("aux/diag_plus_first.json", "always token 0"), ("aux/diag_plus_nosem.json", "semantic features removed")]
    rows = [[l, pct(get(n)["IoU@1"]), pct(get(n)["IoU@3"]), pct(get(n)["IoU@5"]), pct(get(n)["IoU@10"])] for n, l in diag if get(n)]
    if rows:
        table("T5b · Diagnosis of SNAP+ v1 regression (S3DIS subset, 4k)", "aux/diag_*.json", ["selection", "IoU@1", "IoU@3", "IoU@5", "IoU@10"], rows)


def naming():
    order = [("snap", "Segmenter text head"), ("opensnap_3d", "OpenSNAP-3D"), ("vlm", "VLM free naming"), ("rav", "Render-Ask-Verify"),
             ("vlm_choose", "VLM choose (alone)"), ("choose", "**3D-propose / VLM-choose**"),
             ("openscene_point", "closed: dense feature at click"), ("opensnap_closed", "closed: OpenSNAP")]
    u, a3, a7 = load("naming_mp160_unseen_xyz.json"), load("naming_mp160_all_xyz_3b.json"), load("naming_mp160_all_xyz_7b.json")
    if u and a3 and a7:
        rows = [[n, pct(u["results"][k]["acc"]), pct(u["results"][k]["macc"]), pct(a3["results"][k]["acc"]), pct(a3["results"][k]["macc"]),
                 pct(a7["results"][k]["acc"]), pct(a7["results"][k]["macc"])] for k, n in order]
        table("T6/T7 · Click-to-name on Matterport-160 (open vocabulary unless marked closed)",
              f"naming_mp160_unseen_xyz.json ({u['clicks']} unseen-class clicks), naming_mp160_all_xyz_3b/7b.json ({a3['clicks']} clicks)",
              ["method", "unseen acc", "unseen mAcc", "all acc (3B)", "all mAcc (3B)", "all acc (7B)", "all mAcc (7B)"], rows,
              "Free-form names mapped to labels by CLIP ViT-L/14 text similarity. Run-to-run noise ≈ ±3 points on 212 clicks.")


def text_prompts():
    rows = []
    for name, label in [("aux/text_scannet_oracleclicks.json", "training-free: language as click oracle"),
                        ("aux/text_scannet_v3_4k_tok0.json", "trained v3 (frozen decoder), 4k"),
                        ("aux/text_v4_4000.json", "trained v4 (own text decoder), 4k"), ("aux/text_v4_8000.json", "trained v4, 8k"),
                        ("aux/text_v4_12000.json", "trained v4, 12k"), ("aux/text_v4_16000.json", "trained v4, 16k")]:
        d = load(name)
        if d:
            r = d["results"]
            rows.append([label, pct(r["mIoU"]), pct(r["mIoU_head"]), pct(r["mIoU_common"]), pct(r["mIoU_tail"])])
    table("T8 · Text-prompted segmentation, ScanNet200 val ('segment the <class>')", "aux/text_*.json (104 scenes)",
          ["method", "mIoU", "head", "common", "tail (unseen as text)"], rows, "Tail classes are held out of text training.")


def text_s3dis():
    a, b = load("text_s3dis_oracle.json"), load("text_s3dis_v4.json")
    if a and b:
        cls = sorted(b["per_class"])
        rows = [[n, pct(d["results"]["mIoU"])] + [pct(d["per_class"].get(c)) for c in cls]
                for n, d in (("training-free: language as click oracle", a), ("**trained v4 text decoder (16k)**", b))]
        table("T8b · Text-prompted segmentation on S3DIS Area 5 (zero-shot dataset, per class IoU)",
              "text_s3dis_oracle.json, text_s3dis_v4.json", ["method", "mIoU"] + cls, rows)


def single_click():
    d = load("aux/single_click_scannet_4k.json")
    if d:
        r = d["results"]
        rows = [["segmenter's own mask", pct(r["snap_token"])], ["oracle best of candidates", pct(r["oracle"])]] + \
               [[f"IoU-head selection, margin {k}", f"{pct(v[0])} (extra used {100 * v[1]:.0f}%)"] for k, v in r["rules"].items()]
        table("T9 · Single-click candidate selection (ScanNet val)", f"aux/single_click_scannet_4k.json ({r['objects']} objects)", ["rule", "mean IoU"], rows)


def proposals():
    d = load("proposals_s3dis_area5.json") or load("aux/proposals_s3dis_6rooms.json")
    if d:
        rows = [[k, pct(v["r50"]), pct(v["r25"]), f"{v['n']:.0f}", f"{v['s']:.1f}"] for k, v in d.items()]
        table("T10 · Proposal recall vs prompting scheme (S3DIS Area 5)",
              "proposals_s3dis_area5.json" if load("proposals_s3dis_area5.json") else "aux/proposals_s3dis_6rooms.json (pilot, 6 rooms)",
              ["scheme", "recall@0.5", "recall@0.25", "proposals/room", "s/room"], rows)


def systems():
    d = load("systems_s3dis.json")
    if d:
        s = d["summary"]
        table("T11 · Systems: ingest, latency, storage (S3DIS Area 5, 34 rooms)", "systems_s3dis.json",
              ["metric", "value"],
              [["encode per room (s)", f"{s['encode_s']:.2f}"], ["materialise per room (s)", f"{s['materialize_s']:.2f}"],
               ["query on stored relation (ms)", f"{s['query_cached_ms']:.1f}"], ["re-process scan per query (ms)", f"{s['query_reprocess_ms']:.0f}"],
               ["speed-up", f"{s['query_reprocess_ms'] / s['query_cached_ms']:.0f}×"], ["relation (MB)", f"{s['relation_MB']:.2f}"],
               ["raw XYZ (MB)", f"{s['raw_MB']:.2f}"], ["compression", f"{s['raw_MB'] / s['relation_MB']:.1f}×"]])
        rows = [[m, f"{s[m]['mean_ms']:.1f}", f"{s[m]['second_pass_ms']:.1f}", pct(s[m]["ap50"]), pct(s[m]["synonym_ap50"])]
                for m in ("no_refine", "refine_discard", "cracking")]
        table("T12/T14 · Semantic cracking over a query workload (class names, synonyms, absent terms; two passes)", "systems_s3dis.json",
              ["mode", "mean ms", "2nd-pass ms", "AP50", "synonym AP50"], rows)


def sweeps():
    rows = []
    for name in ("aux/q_scannet_train_sweep.json", "aux/q_scannet_train_sweep_dedup.json"):
        d = load(name)
        if not d:
            continue
        for k, v in d["results"].items():
            if k.endswith("/IS") and k.split("/")[0] not in [r[0] for r in rows]:
                v2 = d["results"].get(k.replace("/IS", "/NEAR"), {})
                rows.append([k.split("/")[0], pct(v["AP"]), pct(v["AP50"]), f"{v.get('refined', 0):.2f}", pct(v2.get("AP"))])
    if rows:
        rows.sort(key=lambda r: -float(r[1]))
        table("T13 · Cracking calibration ablation (ScanNet train, 61 scenes)", "aux/q_scannet_train_sweep*.json",
              ["setting", "IS AP", "IS AP50", "refined/scene", "NEAR AP"], rows,
              "g = confidence weight γ, t = trigger θ, dedup = IoU above which a refined mask is merged into an existing object.")


def building():
    d, lat = load("building_s3dis_area5.json"), load("building_latency.json")
    if d:
        s = d["summary"]
        rows = [["rooms / objects", f"{s['rooms']} / {s['objects']}"], ["total ingest (s)", f"{s['ingest_s_total']:.0f}"],
                ["raw XYZ (MB)", f"{s['raw_xyz_MB']:.0f}"], ["catalog on disk (MB)", f"{s['catalog_MB']:.0f}"],
                ["catalog load (s)", f"{s['catalog_load_s']:.2f}"], ["building IS AP / AP50", f"{pct(s['IS']['AP'])} / {pct(s['IS']['AP50'])}"],
                ["building NEAR AP / AP50", f"{pct(s['NEAR']['AP'])} / {pct(s['NEAR']['AP50'])}"]]
        if lat:
            rows += [[f"latency {k} (ms)", f"{v:.2f}"] for k, v in lat["latency_ms"].items()]
            rows.append(["partitioned join == per-room", lat.get("partition_check", "")])
        table("T16 · Building-level catalog and queries (S3DIS Area 5 as one database)", "building_s3dis_area5.json, building_latency.json", ["metric", "value"], rows)


OVIMAP_REPLICA = [("Mask3D+OpenMask3D (offline, RGB-D)", 18.7, 29.4, 6.8, 4.5, 3.2), ("Segment3D+OpenMask3D (offline, RGB-D)", 17.3, 30.9, 19.9, 15.2, 9.0),
                  ("Semantic Gaussian (offline)", 16.1, 22.1, None, None, None),
                  ("OpenScene (offline)", 19.8, 33.9, None, None, None), ("OpenNeRF (offline)", 19.2, 30.9, None, None, None),
                  ("OpenFusion (online)", 14.9, 22.4, None, None, None), ("OVO-SLAM (online)", 24.9, 34.0, 28.1, 17.5, 9.1),
                  ("OVI-MAP (online)", 26.5, 32.2, 34.5, 21.2, 8.5)]
OVIMAP_SCANNET = [("Mask3D+OpenMask3D (offline, RGB-D)", 8.6, 17.5, 10.4, 8.0, 5.1), ("Segment3D+OpenMask3D (offline, RGB-D)", 4.5, 13.5, 5.0, 3.9, 2.3),
                  ("Semantic Gaussian (offline)", 10.7, 17.8, None, None, None),
                  ("OpenScene (offline)", 10.3, 17.3, None, None, None), ("OpenFusion (online)", 8.3, 11.5, None, None, None),
                  ("OVO-SLAM (online)", 14.6, 27.8, 19.4, 12.6, 5.5), ("OVI-MAP (online)", 17.5, 27.6, 23.4, 15.7, 7.2)]
RAZER_SCANNET200 = [("SAM3D", 6.1, 14.2, 21.3, 7.0, 6.2, 4.6), ("OVIR-3D", 13.0, 24.9, 32.3, 14.4, 12.7, 11.7),
                    ("Open3DIS", 23.7, 29.4, 32.8, 27.8, 21.2, 21.8), ("OpenScene (2D fusion)", 11.7, 15.2, 17.8, 13.4, 11.6, 9.9),
                    ("OpenScene (ensemble)", 5.3, 6.7, 8.1, 11.0, 3.2, 1.1), ("OpenMask3D", 15.4, 19.9, 23.1, 17.1, 14.1, 14.9),
                    ("RAZER", 24.7, 31.7, 36.2, 27.8, 24.3, 21.6)]
SEARCH3D_SCANNET200 = [("Mask3D (closed vocabulary, supervised)", 26.9, 39.8, 21.7, 17.9), ("OpenScene 2D fusion + Mask3D masks", 11.7, 13.4, 11.6, 9.9),
                       ("OpenMask3D", 15.4, 17.1, 14.1, 14.9), ("Open3DIS 3D-only (Mask3D)", 18.9, 23.9, 17.4, 15.3),
                       ("Open3DIS 2D+3D", 23.7, 27.8, 21.2, 21.8), ("Search3D (CLIP)", 14.3, 16.1, 13.6, 12.9), ("Search3D (SigLIP)", 23.0, 26.3, 21.2, 21.4)]
OPENSCAN = [("OpenMask3D", (7.2, 7.5, 8.5, 12.8, 16.9, 13.0, 12.2, 18.8, 9.9)), ("SAI3D", (5.3, 5.8, 7.8, 11.3, 10.0, 10.0, 8.7, 11.3, 7.7)),
            ("MaskClustering", (6.2, 7.0, 7.1, 11.1, 16.2, 11.3, 7.4, 12.1, 8.1)), ("Open3DIS", (11.9, 12.8, 14.2, 19.2, 26.7, 19.2, 18.7, 28.3, 15.8))]
NAMES = {"opensnap": "OpenSNAP (XYZ only)", "dense_only": "  dense 3D expert only", "token_only": "  segmenter token only",
         "opensnap+image": "**OpenSNAP + image column**", "image_only": "  image column only"}


def f1(x):
    return "–" if x is None else f"{x:.1f}"


def compare():
    for names, title, ref in ((("compare/replica_ovimap_fps.json", "compare/replica_ovimap_c2f.json", "compare/replica_ul4.json", "compare/replica_lift.json"),
                               "C1 · Replica, OVI-MAP protocol (8 scenes, 51 classes, GT mesh vertices)", OVIMAP_REPLICA),
                              (("compare/scannet18.json",), "C2 · ScanNet, OVI-MAP protocol (18 scenes, ScanNet200 labels)", OVIMAP_SCANNET)):
        if not load(names[0]):
            pending(title, names[0])
            continue
        rows = [[n, f1(a), f1(b), f1(c), f1(e), f1(g)] for n, a, b, c, e, g in ref]
        for name in names:
            d = load(name)
            if not d:
                continue
            key = "48" if "replica" in name else "200"
            prop = {"fps": " [fps]", "c2f": " [c2f]", "ul4": " [union_l4, dev-selected]", "lift": " [union_l4 + lifted 2D]"}.get(
                next((k for k in ("fps", "c2f", "ul4", "lift") if k in name), ""), "")
            for split in d["results"]:
                for v, r in d["results"][split].items():
                    ins = r[f"inst{key}_p_x_iou"]
                    tag = prop + ("" if split == "all" else " (15 val scenes)")
                    rows.append([NAMES.get(v, v) + tag, pct(r["semantic"]["mIoU"]), pct(r["semantic"]["mAcc"]), pct(ins["AP25"]), pct(ins["AP50"]), pct(ins["AP"])])
        idev, ich = image_dev()
        for k, lab in ((("replica_union", "[union_l4]"), ("replica_lift", "[union_l4 + lifted 2D]")) if "replica" in names[0] else
                       (("scannet18", "[union_l4]"),)):
            if k in idev:
                c = idev[k]["dev-chosen image setting"]
                rows.append([f"**OpenSNAP + image column, all settings chosen on ScanNet train dev scenes** {lab}", pct(c["mIoU"]), pct(c["mAcc"]),
                             pct(c["AP25"]), pct(c["AP50"]), pct(c["AP"])])
        tu = load("compare/tuned.json")
        bench_name = "replica" if "replica" in names[0] else "scannet18"
        s18 = load("compare/scannet18_tuned.json") if bench_name == "scannet18" else None
        if s18:
            tu = {"test": {"scannet18": {"default": s18["default"], "dev-chosen": s18["chosen"]}}}
        if tu and bench_name in tu["test"]:
            t = tu["test"][bench_name]
            for k, lab in (("default", "OpenSNAP (XYZ only) [union_l4], default labelling setting, dump replay"),
                           ("dev-chosen", "OpenSNAP (XYZ only) [union_l4], labelling setting chosen on dev scenes"),
                           ("dev-chosen + image (LOSO w_img)", "**OpenSNAP + image column** [union_l4], dev-chosen setting, image weight by leave-one-scene-out")):
                if k in t:
                    c = t[k]
                    rows.append([lab, pct(c["mIoU"]), pct(c["mAcc"]), pct(c["AP25"]), pct(c["AP50"]), pct(c["AP"])])
        for f, tag in (("compare/replica_lift_loso.json", ""), ("compare/replica_lift_loso_tuned.json", ", dev-chosen setting")):
            lc = load(f) if "replica" in names[0] else None
            if not lc:
                continue
            c = lc["loso_cv"]
            rows.append([f"**OpenSNAP + image column + lifted 2D proposals, leave-one-scene-out** [union_l4{tag}]", pct(c["mIoU"]), pct(c["mAcc"]),
                         pct(c["AP25"]), pct(c["AP50"]), pct(c["AP"])])
            from collections import Counter
            mode = Counter(tuple(v) for v in lc["chosen"].values()).most_common(1)[0]
            if mode[1] < 8:
                g = lc["grid"][str(tuple(mode[0]))] if str(tuple(mode[0])) in lc["grid"] else lc["grid"][str(mode[0])]
                rows.append([f"  same, setting chosen in {mode[1]}/8 folds applied to all scenes (not a held-out number)", pct(g["mIoU"]), pct(g["mAcc"]),
                             pct(g["AP25"]), pct(g["AP50"]), pct(g["AP"])])
        cv = load("compare/replica_fusion_cv.json") if "replica" in names[0] else None
        if cv:
            c = cv["loso_cv"]
            rows.append(["**OpenSNAP + image column, fused, weights by leave-one-scene-out CV** [c2f]", pct(c["mIoU"]), pct(c["mAcc"]),
                         pct(c["AP25"]), pct(c["AP50"]), pct(c["AP"])])
        table(title, name + " (ours); OVI-MAP Table 3 (published)", ["method", "mIoU", "mAcc", "AP25", "AP50", "AP"], rows,
              "Published rows are copied from OVI-MAP (CVPR 2026) Table 3. Ours: same scenes, classes and per-vertex scoring "
              "(classes without GT vertices skipped; instance AP with the ScanNet evaluator, predictions < 100 vertices dropped).")
    cv = load("compare/replica_fusion_cv.json")
    if cv:
        rows = [[k.replace("w_img=", "").replace(" alpha=", " / "), pct(r["mIoU"]), pct(r["mAcc"]), pct(r["AP25"]), pct(r["AP50"]), pct(r["AP"])]
                for k, r in cv["grid"].items()]
        table("C1b · Image-column fusion weights on Replica (test-set grid: sensitivity only, not a result)", "compare/replica_fusion_cv.json",
              ["w_image / alpha", "mIoU", "mAcc", "AP25", "AP50", "AP"], rows,
              "Per-point semantics mix 30% dense per-point evidence (beta = 0.3) as in the system default. The leave-one-scene-out row in C1 is "
              "the only fused number to report; this grid shows how sensitive the fusion is to alpha (dense vs token split).")
    d = load("compare/scannet200_val.json")
    if d:
        rows = [[f"{n} (Search3D Tab. V)", f1(a), "–", "–", f1(h), f1(c), f1(t)] for n, a, h, c, t in SEARCH3D_SCANNET200]
        rows += [[f"{n} (RAZER Tab. II)", *[f1(x) for x in v]] for n, *v in RAZER_SCANNET200]
        sp = lambda r: [pct(r[k]) for k in ("AP", "AP50", "AP25", "AP_head", "AP_common", "AP_tail")]
        rows.append(["OpenSNAP, XYZ only", *sp(d["results"]["all"]["opensnap"]["inst198_p_x_iou"])])
        t = load("compare/scannet200_tuned.json")
        if t:
            rows.append(["OpenSNAP, XYZ only [union_l4], dev-chosen", *sp(t["chosen"])])
        idev, _ = image_dev()
        if "scannet200" in idev:
            rows.append(["**OpenSNAP + image column [union_l4], all settings chosen on ScanNet train dev scenes**",
                         *sp(idev["scannet200"]["dev-chosen image setting"])])
        table("C3 · ScanNet200 val open-vocabulary instance segmentation (312 scenes, 198 classes)",
              "compare/scannet200_val.json, scannet200_tuned.json, image_dev_b.json; Search3D Table V, RAZER Table II (published)",
              ["method", "AP", "AP50", "AP25", "head", "common", "tail"], rows,
              "Published rows use posed RGB(-D) images; most use Mask3D masks trained on ScanNet200. Head / common / tail are AP. "
              "Our image column uses OpenScene's processed ScanNet frames (every 20th frame, 320x240); dev scenes are ScanNet train scenes.")
    else:
        pending("C3 · ScanNet200 val", "compare/scannet200_val.json")

    d = load("compare/hl3d_partition.json")
    if d:
        b, agg = d["buildings"], d["aggregate"]
        keys = ["NEAR(0.25)", "NEAR(0.5)", "NEAR(1.0)", "NEAR(2.0)", "ABOVE(0)", "ON(0.1)"]
        rows = []
        for s, r in agg.items():
            rows.append([s] + [pct(r[k]["recall"]) for k in keys] + [f"{r['ms_per_query']:.2f}", f"{r['cells'] / 1e6:.2f}"])
        g = agg["global"]
        note = (f"{b['n']} buildings, {b['rooms']} rooms, {b['objects']} objects, {b['levels']} levels (levels found from region floor heights; "
                f"{pct(b['levels_match_hl3d'])}% coincide with a HouseLayout3D horizontal layout plane). Recall = share of the unpartitioned "
                f"(global) join's answers (score >= 0.5) that a partitioning returns; 'cells' = join work in millions of pairs per query. "
                f"Global vertical answers whose only witness is on another floor: ABOVE {g['ABOVE(0)'].get('cross_level_answers', '–')}, "
                f"BELOW {g['BELOW(0)'].get('cross_level_answers', '–')}, ON {g['ON(0.1)'].get('cross_level_answers', '–')} "
                f"(of {g['ABOVE(0)']['answers']}, {g['BELOW(0)']['answers']}, {g['ON(0.1)']['answers']}).")
        table("C4 · Spatial joins over whole multi-floor buildings (HouseLayout3D / Matterport3D)", "compare/hl3d_partition.json",
              ["partitioning"] + keys + ["ms / query", "cells (M)"], rows, note)
    d = load("compare/calibration.json")
    if d:
        sets = list(d["n_scenes"])
        rows = [[b] + [f"{r[n]['ECE']:.3f} / {r[n]['Brier']:.3f} / {pct(r[n]['AP'])}" for n in sets] for b, r in d["background"].items()]
        for tau, r in d["tau"].items():
            rows.append([f"tau = {tau}"] + [f"{r[n]['ECE']:.3f} / {r[n]['Brier']:.3f} / {pct(r[n]['AP'])}" for n in sets])
        if "platt_applied" in d:
            rows.append([f"Platt (A={d['platt']['A']:.2f}, B={d['platt']['B']:.2f}) fitted on ScanNet train"] +
                        [f"{d['platt_applied'][n]['ECE_after']:.3f} / {d['platt_applied'][n]['Brier_after']:.3f} / {pct(d['platt_applied'][n]['AP'])}" for n in sets])
        table("C5 · Calibration of IS(c): ECE / Brier / IS-query AP", "compare/calibration.json", ["setting"] + [f"{n} ({d['n_scenes'][n]})" for n in sets], rows,
              "ECE with 15 equal-width bins over every (object, class present in the scene) pair; y = object overlaps a GT instance of the class at IoU >= 0.5.")
        rows = []
        for name in ("alpha", "rho", "tau_r"):
            for v, r in d[name].items():
                rows.append([f"{name} = {v}"] + [f"{pct(r[n]['AP'])} ({r[n]['ECE']:.3f})" for n in sets])
        if "synonyms_paired" in d:
            sp = d["synonyms_paired"]
            rows.append(["synonym classes, queried by class name"] + [f"{pct(sp[n]['class_name_AP'])} ({sp[n]['class_name_ECE']:.3f})" for n in sets])
            rows.append(["same classes, queried by synonym"] + [f"{pct(sp[n]['synonym_AP'])} ({sp[n]['synonym_ECE']:.3f})" for n in sets])
        table("C6 · Sensitivity of IS(c) to expert weight, relay and temperature: AP (ECE)", "compare/calibration.json", ["setting"] + sets, rows,
              "Defaults: alpha 0.7, rho 0.8, tau 0.01, tau_r 0.05; all rows use B = the 234-word training vocabulary. Synonym rows: same classes, queried by class name vs by synonym.")
    d = load("compare/workload_s3dis.json")
    if d:
        rows = []
        for p, r in d["results"].items():
            last = r[max(r, key=int)]
            first = r[min(r, key=int)]
            rows.append([p] + [pct(r[c]["AP50"]) for c in r] + [f"{last['cracked']:.1f}", f"{last['MB']:.2f}", f"{last['rejected']:.1f}",
                                                             f"{last['evicted']:.1f}", f"{last['ms_per_query']:.1f}"])
        cps = list(next(iter(d["results"].values())))
        table("C7 · Long cracking workload: probe AP50 drift, growth and safeguards (S3DIS Area 5)", "compare/workload_s3dis.json",
              ["policy"] + [f"AP50 @{c}" for c in cps] + ["cracked rows", "relation MB", "rejected", "evicted", "ms / query"], rows,
              f"{d['rooms']} rooms x {d['queries_per_room']} random IS queries (50% LVIS nouns, 25% training vocabulary, 25% class names and synonyms). "
              "Probe = IS(c) for every annotated class present, scored after each checkpoint without refinement.")
    d = load("compare/scale.json")
    if d:
        rows = [[r["copies"], r["rooms"], r["objects"], f"{r['catalog_MB']:.0f}", f"{r['IS_ms']:.2f}",
                 "OOM" if r.get("padded_ms") is None else f"{r['padded_ms']:.2f}", f"{r['pushdown_ms']:.2f}", f"{r['candidates']:.0f}",
                 f"{r['peak_GPU_MB']:.0f}"] for r in d]
        table("C8 · Scaling past one building (HouseLayout3D catalogs replicated into a campus)", "compare/scale.json",
              ["copies", "rooms", "objects", "catalog MB", "IS ms", "NEAR padded ms", "NEAR pushdown ms", "candidates", "peak GPU MB"], rows,
              "pushdown = IS selections before the join + exact uniform-grid spatial join on surviving candidates (exact for threshold answers).")

    rows = []
    for name, tag in (("compare/replica_video.json", "RGB-only video, estimated metric scale"),
                      ("compare/replica_video_oracle_scale.json", "RGB-only video, oracle scale (diagnostic)")):
        d = load(name)
        if d:
            for v, r in d["results"]["all"].items():
                ins = r["inst48_p_x_iou"]
                ob = r.get("semantic_observed", {})
                rows.append([tag, NAMES.get(v, v), pct(r["semantic"]["mIoU"]), pct(r["semantic"]["mAcc"]), pct(ins["AP25"]), pct(ins["AP50"]), pct(ins["AP"]),
                             pct(ob.get("mIoU")), pct(ob.get("mAcc")), pct(ob.get("observed_fraction"))])
    if rows:
        rows = [["RGB-D + GT poses, 2000 frames", "OVI-MAP (published)", "26.5", "32.2", "34.5", "21.2", "8.5", "–", "–", "–"]] + rows
        rec = None
        p = os.path.join(R, "..", "..", "downloads", "runs", "video_replica", "reconstruction.json")
        if os.path.exists(p):
            rec = json.load(open(p))
        note = "60 RGB frames per sequence (no depth, no poses) -> VGGT + metric depth -> OpenSNAP; scored on the GT mesh after a similarity " \
               "alignment of predicted to GT camera centres (scoring only). Unobserved GT vertices count as wrong."
        if rec:
            note += (f" Metric-scale error {100 * np.mean([r['metric_scale_error'] for r in rec.values()]):.0f}% on average "
                     f"(range {100 * min(r['metric_scale_error'] for r in rec.values()):.0f}-{100 * max(r['metric_scale_error'] for r in rec.values()):.0f}%), "
                     f"camera RMSE {100 * np.mean([r['cam_rmse_m'] for r in rec.values()]):.1f} cm.")
        table("C10 · Phone-video pipeline on the 8 Replica sequences (RGB only)", "compare/replica_video*.json", ["input", "method", "mIoU", "mAcc", "AP25", "AP50", "AP", "mIoU (observed)", "mAcc (observed)", "GT observed"],
              rows, note)
    d = load("compare/guard.json")
    if d:
        rows = [[src, v["n"], v["v1_fired"], v["v2_fired"], v.get("v3_fired", "–")] for src, v in d["summary"].items()]
        table("C11 · Dense-collapse guard: share rule (v1), exact agreement (v2), soft agreement calibrated on clean scans (v3)", "compare/guard.json",
              ["scenes", "n", "v1 fired", "v2 fired", "v3 fired"], rows,
              f"v1: {d['rule_v1']}; v2: {d['rule_v2']}; v3: {d.get('rule_v3', '–')}. Calibration rows are random clean scans (false-alarm rate); "
              "phone-video rows should fire (the 8 RGB-only Replica clouds show 45-71% 'stair rail' dense labels). "
              "Benchmark rows are every clean scan on which v1 fired in the large runs (14/312 ScanNet val, 21/354 Matterport rooms).")
    d = load("compare/openscan.json")
    if d:
        aspects = ["affordance", "property", "type", "manner", "synonyms", "requirement", "element", "material", "mean"]
        rows = [[n] + [f1(x) for x in vals] for n, vals in OPENSCAN]
        for v, r in d["results"].items():
            rows.append([NAMES.get(v, v)] + [pct(r[a]["AP"]) for a in aspects])
        table("C12 · OpenScan / GOV-3D attribute queries (ScanNet200 val, instance AP)", "compare/openscan.json; OpenScan Table 3 (published)",
              ["method"] + aspects, rows, "Published rows use posed RGB images (CLIP crops) and pretrained 3D/2D mask proposals; ours XYZ only.")
    rows = []
    for ds in ("replica", "s3dis"):
        d = load(f"compare/sweep_{ds}.json")
        if d:
            for k, r in d["results"].items():
                if k.endswith("/IS") and (k.startswith("dedup") or k.startswith("opensnap")):
                    rows.append([ds, k[:-3], pct(r["AP"]), pct(r["AP50"]), f"{r.get('refined', 0):.2f}"])
    if rows:
        table("C13 · Cracking trigger theta and dedup IoU across datasets (IS queries)", "compare/sweep_*.json", ["dataset", "setting", "AP", "AP50", "refined / scene"], rows)
    rows = []
    for ds in ("s3dis", "replica", "scannet"):
        d = load(f"compare/agg_{ds}.json")
        if d:
            for v in ("opensnap", "opensnap_noisy_or", "opensnap_sum", "phrase_embedding"):
                r = d["results"].get(f"{v}/NEAR")
                if r:
                    ps, ys = [], []
                    for rec in d["records"]:
                        for q in rec["results"][v]["queries"]:
                            if q["kind"] == "NEAR":
                                for sc, io in zip(q["scores"], q["ious"]):
                                    ps.append(sc); ys.append(float(max(io, default=0) >= 0.5))
                    ps, ys = np.asarray(ps), np.asarray(ys)
                    e = sum(((ps > lo) & (ps <= lo + 1 / 15)).mean() * abs(ps[(ps > lo) & (ps <= lo + 1 / 15)].mean() - ys[(ps > lo) & (ps <= lo + 1 / 15)].mean())
                            for lo in np.linspace(0, 1, 16)[:-1] if ((ps > lo) & (ps <= lo + 1 / 15)).any())
                    name = {"opensnap": "OVQL, max witness (default)", "opensnap_noisy_or": "OVQL, noisy-OR", "opensnap_sum": "OVQL, clipped count",
                            "phrase_embedding": "phrase embedding 'a near a b' (OpenMask3D/Search3D/OVI-MAP style)"}[v]
                    rows.append([ds, name, pct(r["AP"]), pct(r["AP50"]), f"{e:.3f}"])
    if rows:
        table("C9 · Relational queries: OVQL semantics vs phrase-embedding retrieval (NEAR, d = 0.5 m)", "compare/agg_*.json",
              ["dataset", "query evaluation", "AP", "AP50", "ECE"], rows,
              "Same stored objects and evidence columns for every row; ground truth = GT objects of class a with a GT object of class b within 0.5 m.")

REPLICA51 = [  # (source, method, input, mIoU, mAcc)
    ("OVO-SLAM Tab. II", "OpenScene (distilled)", "mesh", 14.8, 23.0), ("OVO-SLAM Tab. II", "OpenScene (ensemble)", "mesh + RGB", 15.9, 24.6),
    ("OVO-SLAM Tab. II", "OpenNeRF", "RGB-D + GT poses", 20.4, 31.7), ("OVO-SLAM Tab. II", "HOV-SG", "RGB-D + GT poses", 22.5, 34.2),
    ("OVO-SLAM Tab. II", "Open3DIS (SigLIP)", "mesh + RGB", 25.6, 38.7), ("OVO-SLAM Tab. II", "ConceptGraphs", "RGB-D + GT poses", 16.7, 33.7),
    ("OVO-SLAM Tab. II", "Open-Fusion", "RGB-D + GT poses", 20.5, 34.8), ("OVO-SLAM Tab. II", "OVO-mapping", "RGB-D + GT poses", 27.0, 39.1),
    ("Ov3R Tab. 3", "O2V-mapping", "RGB-D + GT poses", 7.8, 13.9), ("Ov3R Tab. 3", "Ov3R", "RGB + GT geometry", 31.9, 42.3),
    ("Ov3R Tab. 3", "Ov3R (own reconstruction)", "RGB only", 30.4, 41.2),
    ("VOIM Tab. I", "VOIM", "RGB-D + GT poses", 28.6, None), ("VOIM Tab. I", "VOIM (monocular)", "RGB only", 27.8, None),
    ("OVI-MAP Tab. 3", "OVI-MAP", "RGB-D + GT poses", 26.5, 32.2),
    ("IRIS-SLAM Tab. II (own re-runs)", "OVO-SLAM", "RGB-D + GT poses", 29.93, 43.60), ("IRIS-SLAM Tab. II (own re-runs)", "Omni-Map", "RGB-D + GT poses", 29.06, 44.54),
    ("IRIS-SLAM Tab. II (own re-runs)", "IRIS-mapping", "RGB-D + GT poses", 31.15, 40.63), ("IRIS-SLAM Tab. II (own re-runs)", "LangSplat", "RGB-D + GT poses", 10.00, 22.93),
    ("IRIS-SLAM Tab. II (own re-runs)", "GraspSplats", "RGB-D + GT poses", 10.42, 23.79), ("IRIS-SLAM Tab. II (own re-runs)", "OpenGaussian", "RGB-D + GT poses", 6.82, 16.66),
    ("IRIS-SLAM Tab. II (own re-runs)", "ConceptFusion", "RGB-D + GT poses", 4.75, 19.29)]
SCANNET200_312 = [("OVO-SLAM Tab. III (FVS)", "Open3DIS (SigLIP)", 9.4, 17.0), ("OVO-SLAM Tab. III (FVS)", "OpenScene (ensemble)", 11.6, 22.8),
                  ("OVO-SLAM Tab. III (FVS)", "OVO-mapping", 17.4, 35.9)]


def image_dev():
    """Merged test results of the image setting chosen on ScanNet train development scenes (tune_image_dev.py)."""
    out, chosen = {}, None
    for f in ("compare/scannet_image_dev.json", "compare/image_dev_a.json", "compare/image_dev_b.json"):
        d = load(f)
        if d:
            out.update(d["test"])
            chosen = d["chosen"]
    return out, chosen


def landscape():
    rows = [[src, m, inp, f1(a), f1(b)] for src, m, inp, a, b in REPLICA51]
    for name, tag in (("compare/replica_ovimap_c2f.json", "c2f"), ("compare/replica_lift.json", "union_l4 + lifted 2D proposals")):
        d = load(name)
        if d:
            for v in ("opensnap", "opensnap+image", "image_only"):
                r = d["results"]["all"].get(v)
                if r:
                    rows.append(["ours", f"{NAMES.get(v, v)} [{tag}]", "mesh (+ RGB for image rows)", pct(r["semantic"]["mIoU"]), pct(r["semantic"]["mAcc"])])
    idev, _ = image_dev()
    if "replica_union" in idev:
        c = idev["replica_union"]["dev-chosen image setting"]
        rows.append(["ours", "**OpenSNAP + image column, all settings chosen on ScanNet train dev scenes** [union_l4]", "mesh + RGB",
                     pct(c["mIoU"]), pct(c["mAcc"])])
    tu = load("compare/tuned.json")
    if tu and "replica" in tu["test"]:
        c = tu["test"]["replica"]["dev-chosen + image (LOSO w_img)"]
        rows.append(["ours", "**OpenSNAP + image column** [union_l4], dev-chosen setting, image weight by leave-one-scene-out", "mesh + RGB",
                     pct(c["mIoU"]), pct(c["mAcc"])])
    for f, tag in (("compare/replica_lift_loso.json", ""), ("compare/replica_lift_loso_tuned.json", ", dev-chosen setting")):
        lc = load(f)
        if lc:
            rows.append(["ours", f"**OpenSNAP + image column + lifted 2D proposals, leave-one-scene-out** [union_l4{tag}]", "mesh + RGB",
                         pct(lc["loso_cv"]["mIoU"]), pct(lc["loso_cv"]["mAcc"])])
    table("C14 · Replica open-vocabulary 3D semantic segmentation, 51 classes, 8 scenes: published landscape", "papers cited per row; ours from compare/*.json",
          ["source", "method", "input", "mIoU", "mAcc"], rows,
          "All rows: Replica NICE-SLAM scenes, 51 classes, per-vertex scoring of the GT mesh. Groups differ in small details "
          "(OVO's k-NN vote vs OVI-MAP's mapping, frame subsampling); IRIS-SLAM re-ran its baselines and gets higher numbers "
          "for OVO than OVO itself, so compare within a group. Our points are the GT mesh vertices.")
    d = load("compare/scannet200_val.json")
    rows = [[src, m, f1(a), f1(b)] for src, m, a, b in SCANNET200_312]
    if d:
        r = d["results"]["all"]["opensnap"]["semantic"]
        rows.append(["ours", "OpenSNAP (XYZ only)", pct(r["mIoU"]), pct(r["mAcc"])])
    t = load("compare/scannet200_tuned.json")
    if t:
        rows.append(["ours", "OpenSNAP (XYZ only), union_l4, labelling setting chosen on dev scenes", pct(t["chosen"]["mIoU"]), pct(t["chosen"]["mAcc"])])
    idev, _ = image_dev()
    if "scannet200" in idev:
        c = idev["scannet200"]["dev-chosen image setting"]
        rows.append(["ours", "**OpenSNAP + image column, all settings chosen on ScanNet train dev scenes**", pct(c["mIoU"]), pct(c["mAcc"])])
    table("C15 · ScanNet200 semantic segmentation, all 312 val scenes", "OVO-SLAM Tab. III; compare/scannet200_val.json", ["source", "method", "mIoU", "mAcc"], rows)


if __name__ == "__main__":
    out.append("# OpenSNAP result tables\n\nGenerated by `scripts/make_tables.py` from the JSON files in `results/`. Do not edit by hand.\n")
    for f in (semseg, facade_compare, facade_search, queries, refiner, interactive, naming, text_prompts, text_s3dis, single_click, proposals, systems, sweeps, building, compare, landscape):
        f()
    open(os.path.join(R, "TABLES.md"), "w").write("\n".join(out))
    print("\n".join(out))
