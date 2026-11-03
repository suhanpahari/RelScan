# Technical response to the October 2026 review (Aruni Roychowdhury)

Scope: the technical criticisms only (baselines, overlap, ablations, missing rows). Paper framing is deferred.
Table numbers refer to the reviewed PDF (`OpenSNAP: Towards Open-Vocabulary Object Databases for 3D Scans`).
All settings below are fixed before looking at a test scene: the 3D labelling setting and the image column are the ones
chosen on development scenes (`results/compare/tuned.json`, `image_dev_a.json`; S3DIS Area 1 and 30 ScanNet *train* scenes).

## 1. ScanNet overlap (Table III)

- **What overlaps.** SNAP-C was trained on ScanNet, SemanticKITTI, nuScenes, Pandaset, STPLS3D and DALES; the OpenScene ScanNet model was
  distilled on ScanNet train, and 3 of OVI-MAP's 18 ScanNet scenes (`scene0168_00/01/02`) are ScanNet *train* scenes.
- **Do the other rows of Table III have it?** The offline rows do: Mask3D (behind OpenMask3D) is trained on ScanNet200
  train, Segment3D self-trains on ScanNet scenes, OpenScene is distilled on ScanNet train. The online rows (OVO-SLAM,
  OVI-MAP) use 2D foundation models only and have no 3D training on ScanNet.
- **Does it inflate our numbers?** No. On the 15 ScanNet *val* scenes every OpenSNAP row is as good or better than on all
  18 (`results/compare/ablate_evidence.json`):

| ScanNet, OVI-MAP protocol | scenes | mIoU | mAcc | AP25 | AP50 | AP |
|---|---|---|---|---|---|---|
| OVI-MAP (published) | 18 | 17.5 | 27.6 | 23.4 | 15.7 | 7.2 |
| OpenSNAP, geometry only | 18 | 15.1 | 21.2 | 18.4 | 14.0 | 9.5 |
| OpenSNAP, geometry only | 15 val | 16.3 | 22.9 | 19.8 | 16.4 | 11.5 |
| OpenSNAP + image, SigLIP-L (OVI-MAP's encoder) | 18 | 29.9 | 43.1 | 39.4 | 31.3 | 20.9 |
| OpenSNAP + image, SigLIP-L (OVI-MAP's encoder) | 15 val | 30.6 | 43.1 | 40.0 | 32.6 | 21.7 |
| OpenSNAP + image, SigLIP-so400m | 15 val | 29.4 | 42.8 | 39.2 | 31.0 | 20.5 |

  OVI-MAP's per-scene scores are not published, so its 15-scene number cannot be recomputed; Replica (held out from every
  trained component of OpenSNAP) remains the clean comparison.

## 2. OVI-MAP is online: compute and conditions (Table III)

| | OVI-MAP (published) | OpenSNAP (measured) |
|---|---|---|
| setting | online, incremental, RGB-D stream | offline, complete scan given up front |
| geometry | fused from depth while mapping | the GT mesh vertices (protocol input) |
| frames used for semantics | every 10th frame (about 200 per Replica sequence) | visibility over every 10th frame; top-5 views per object |
| VLM calls | about 18.6 SigLIP queries per instance | 15 SigLIP crops per object (5 views x 3 scales) |
| image encoder | SigLIP-L/16-384 | SigLIP-L/16-384 (row below) or so400m |
| cost | about 1.6 s GPU per keyframe (CropFormer 965 ms dominates), RTX 3090, threads in parallel; real-time at 30 fps with frame skipping | 37 s per Replica scene in total (14 s image evidence), one 3g.40gb A100 MIG slice |

The compute is of the same order; the real asymmetry is offline vs online (whole scan, all frames, final geometry).
Same encoder as OVI-MAP (SigLIP-L), Replica 8 scenes: **29.8 mIoU / 40.3 mAcc / 33.1 AP25 / 27.3 AP50 / 21.2 AP** vs
OVI-MAP 26.5 / 32.2 / 34.5 / 21.2 / 8.5 (AP25 is the one metric where OVI-MAP stays ahead without lifted 2D proposals).
Offline image-based systems on the same scenes (published): Open3DIS (SigLIP) 25.6 mIoU (OVO-SLAM Tab. II).

**Measured OVO-SLAM** (official code `tberriel/OVO`, default config: GT poses, SAM 2.1-L, SigLIP-384 with the learned
CLIP-merging predictor, a keyframe every 10th frame; all 8 Replica scenes, every frame). OVO uses the same OpenNeRF GT and
51 reduced classes, so its per-vertex labels and instance masks are also scored with our evaluator
(`scripts/score_ovo.py`, `results/compare/ovo_replica.json`); both evaluators give the same numbers.

| Replica, 8 scenes | mIoU | mAcc | AP25 | AP50 | AP | time per scene |
|---|---|---|---|---|---|---|
| OVO-SLAM (paper) | 27.0 | 39.1 | – | – | – | |
| OVO-SLAM (measured, our run) | 27.4 | 39.9 | 28.0 | 19.5 | 11.3 | 24.6 min (3.1-11.2 s per keyframe, SAM dominates) |
| OVO-SLAM, TextRegion + PE-Core-L14-336 (repo, Jan 2026; reported 35.7 / 57.0) | **33.4** | **55.0** | **38.7** | 23.8 | 12.4 | 13.8 min (2.0-4.3 s per keyframe) |
| OpenSNAP + image, SigLIP-L | 29.8 | 40.3 | 33.1 | 27.3 | 21.2 | 37 s |
| OpenSNAP + image, SigLIP-so400m | 31.6 | 44.1 | 34.8 | **30.6** | **24.7** | 37 s |

Both on one shared 3g.40gb A100 MIG slice; OVO is online and incremental (poses given, no geometry up front), so its time
is the cost of mapping a stream, not a like-for-like offline cost. The TextRegion variant (`OVO_tr/`, same code with
`embed_type: TextRegion`, `model_card: PE-Core-L14-336`, `mask_res: 336`; `results/compare/ovo_textregion_replica.json`)
lands 2.3 mIoU below its reported number.

Reading (honest): the current OVO **beats OpenSNAP on semantics** (+1.8 mIoU, +10.9 mAcc) and on loose AP25 (+3.9), with a
stronger encoder (PE-Core) and dense region-text features. OpenSNAP keeps a clear lead on strict instance quality (AP50
+6.8, AP 0.50:0.95 2x) from the promptable segmenter's tight masks, and is ~20x faster offline. Its image column uses
SigLIP; swapping in PE-Core / TextRegion features is untested.

## 3. Proposals vs image features vs semantic decoding (new ablation)

`scripts/ablate_evidence.py`; ground-truth masks come from `eval_standard.py --proposals gt` (one mask per GT instance,
token from one click at the instance centre, quality 1; an oracle, for this ablation only).

| Replica (8 scenes) | our proposals: mIoU / mAcc / AP25 / AP50 / AP | GT masks (oracle): mIoU / mAcc / AP25 / AP50 / AP |
|---|---|---|
| dense 3D expert only | 18.7 / 27.4 / 16.5 / 15.3 / 13.2 | 20.2 / 28.0 / 14.8 / 14.8 / 14.8 |
| segmenter token only | 14.9 / 21.8 / 13.9 / 12.8 / 10.6 | 14.4 / 21.4 / 11.7 / 11.7 / 11.6 |
| 3D experts (geometry only) | 17.5 / 27.0 / 18.0 / 16.0 / 13.8 | 17.8 / 27.0 / 18.0 / 18.0 / 18.0 |
| image only, argmax SigLIP (OVI-MAP-style decoding) | 26.6 / 40.5 / 29.6 / 24.7 / 19.0 | 37.5 / 54.1 / 47.0 / 47.0 / 46.6 |
| 3D + image (OpenSNAP) | **31.6 / 44.1 / 34.8 / 30.6 / 24.7** | 41.4 / 53.1 / 43.9 / 43.9 / 43.8 |

| ScanNet, 15 val scenes | our proposals | GT masks (oracle) |
|---|---|---|
| 3D experts (geometry only) | 16.3 / 22.9 / 19.8 / 16.4 / 11.5 | 15.8 / 22.3 / 17.3 / 17.3 / 17.2 |
| image only, argmax SigLIP | 25.2 / 41.5 / 35.9 / 28.9 / 19.5 | 32.2 / 47.7 / 40.7 / 40.7 / 40.4 |
| 3D + image (OpenSNAP) | **29.4 / 42.8 / 39.2 / 31.0 / 20.5** | 38.4 / 48.8 / 40.4 / 40.4 / 40.2 |

Reading:
- **Image features are the largest single factor for semantics**: +14 mIoU on Replica and +13 on ScanNet over geometry only.
- **Proposals are the largest factor for instance AP**: oracle masks raise image-only AP from 19.0 to 46.6 (Replica) and
  from 19.5 to 40.4 (ScanNet); without images, oracle masks add little (geometry-only semantics are the limit there).
- **The decoding (product of experts) adds +4 to +5 mIoU on top of the image evidence** with both proposal sets, and
  +5.7 AP with our proposals on Replica (+1.0 on ScanNet). With oracle masks it no longer helps AP (43.8 vs 46.6 Replica,
  40.2 vs 40.4 ScanNet): its AP gain comes from re-ranking imperfect masks, not from better labels.
- Strict AP with image-only decoding on our proposals already exceeds OVI-MAP (19.0 vs 8.5): the promptable segmenter's
  tight masks, not the fusion, explain the strict-AP margin.

## 4. Table IV with images

Add the existing image-column row (`results/compare/image_dev_b.json`, ScanNet200 val, 312 scenes, 198 classes):
OpenSNAP + image evidence **15.3 AP / 24.1 AP50 / 29.9 AP25; head 21.3, common 12.1, tail 12.1** (geometry only: 6.1).
Still below Search3D-SigLIP (23.0), Open3DIS 2D+3D (23.7) and Details Matter (ICCV 2025; 25.8 top-1 with Mask3D trained
on ScanNet200 train + Grounded-SAM + Alpha-CLIP). These all use a ScanNet200-supervised mask network; the gap is the
long tail and is stated as a limitation.

## 5. Stronger relational baseline (Table V)

`scripts/eval_relational.py` (`downloads/runs/lane_relational2.sh`; outputs `results/compare/relational_{replica,s3dis}[_allpairs].json`).
Every variant scores the same stored objects (Replica: c2f proposals + SigLIP-so400m object vectors; S3DIS: fps
proposals, no images) on the same queries: the NEAR queries of Table V replayed from `agg_<ds>.json` ("published"), and
separately every (a, b) pair with a true answer ("all pairs"). Variants:
- **OVQL**: Eq. (4) IS(c) with background B = 4 words (as in the paper's runs, see 6) or the 234-word vocabulary; Replica
  also with the image expert (product of the 3D and SigLIP distributions over [c] + B234).
- **vector search + the same predicates**: relu(cos(object, text)) for each argument, NEAR(a, b) = s_a x max_b(near x s_b),
  i.e. the reviewer's baseline with OVQL's spatial predicate and max-witness semantics but no calibration.
- **label then filter**: each object takes its argmax label (over B234 or over the dataset's class list), then SQL-style
  equality + spatial filter, ranked by cosine.
- **phrase embedding** (current Table V baseline): cosine with "a near a b".

NEAR, AP / AP50 (ECE for OVQL):

| | Replica published | Replica all pairs | S3DIS published | S3DIS all pairs |
|---|---|---|---|---|
| OVQL, B4, 3D only | 20.5 / 27.2 (0.022) | 20.8 / 26.4 (0.030) | 11.8 / 19.4 (0.030) | 10.2 / 16.0 (0.025) |
| OVQL, B234, 3D only | 21.9 / 30.7 (0.013) | 23.1 / 30.4 (0.012) | 13.8 / 22.6 (0.016) | 10.8 / 17.5 (0.013) |
| vector search + predicates, CLIP (3D) | 20.2 / 26.1 | 19.9 / 26.4 | **15.5 / 26.7** | **12.2 / 20.5** |
| label then filter, B234, CLIP | 0.0 / 0.0 | 2.3 / 3.0 | 0.0 / 0.0 | 0.0 / 0.3 |
| label then filter, closed set, CLIP | 2.6 / 3.2 | 4.5 / 5.4 | 7.2 / 9.4 | 5.4 / 7.8 |
| phrase embedding, CLIP | 17.4 / 21.0 | 15.8 / 20.4 | 10.7 / 17.7 | 7.9 / 13.0 |
| OVQL, B234 + image | 25.5 / 36.5 (0.012) | **31.6 / 43.3** (0.011) | | |
| vector search + predicates, SigLIP | **26.0 / 36.9** | 30.2 / 41.2 | | |
| label then filter, closed set, SigLIP | 7.8 / 10.9 | 11.0 / 14.6 | | |
| phrase embedding, SigLIP | 14.8 / 23.7 | 16.3 / 24.2 | | |

IS(c) alone, AP: Replica OVQL B234 + image 28.7 vs SigLIP cosine 27.8; OVQL B234 3D 18.8 vs CLIP cosine 17.1;
S3DIS OVQL B234 10.2 vs CLIP cosine 10.2 (ECE OVQL 0.006 / 0.015 / 0.021; cosine is not a probability).

Reading (honest):
- **The AP gain of Table V over the phrase baseline comes from evaluating the predicate structure**, not from calibration:
  vector search with the same NEAR predicate recovers almost all of it. It is ahead of OVQL on S3DIS (+1.7 / +1.4 AP) and
  on the Replica published queries with images (+0.5), and behind on Replica with images, all pairs (-1.4) and with 3D only
  (-1.7 / -3.2).
- **What calibration buys is a probability**: OVQL's scores have ECE 0.006-0.016, so thresholds ("p > 0.5"), expected
  counts and cross-scene ranking mean the same thing in every room; cosine scores do not (no fixed threshold, scale differs by
  encoder and by phrase). The claim to make is calibrated answers at the same ranking quality, not better ranking.
- **Label-then-filter fails** even with the dataset's own class list: one hard label per object loses every object whose
  argmax is a near-synonym (e.g. "closet door" for door) and a single mislabel removes the witness for NEAR.
- The 234-word background is better than the 4-word one for NEAR (+1.4 to +2.3 AP on Replica, +0.6 to +2.0 on S3DIS) and
  halves ECE; IS AP is slightly lower (-0.8 to -1.5). Table V should be regenerated with B234 to match the text.

## 6. Background vocabulary discrepancy (found while doing 5)

The paper states that IS(c) competes against the 234-word training vocabulary. The query engine's default background is
4 words (`opensnap/query/engine.py`, `BACKGROUND`), and eval_queries / eval_workload / eval_scale use that default, so
Tables V, VI and Fig. 6 were computed with |B| = 4. Fig. 4 and the ECE claims use the 234-word vocabulary
(eval_calibration). The relational rerun reports both.

## 7. Suggested related systems

- Beyond Bare Queries (ICRA 2025): scene graph + LLM deductive grounding; RGB-D + poses; Replica/ScanNet semseg on a
  different 8-scene protocol (Replica mIoU 0.27 for BBQ-CLIP), grounding on Sr3D+/Nr3D/ScanRefer. Related work and the
  qualitative relational comparison; a run needs its code and an LLM API.
- Details Matter (ICCV 2025): Table IV row (published, top-1 protocol).
- SAM 3 (ICLR 2026): lifted-2D baseline / proposal source; needs its released weights (gated) and code.
