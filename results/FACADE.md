# UnderOneFacade: zero-shot facade study (October 2026)

Out-of-domain stress test of OpenSNAP on laser-scanned urban facades, against published and constructed zero-shot
alternatives. Every number below is generated into `results/TABLES.md` (T2d, T2e, T3f, T5f, T5l) from the JSON files
listed there; this page records the setup and what we concluded.

## Data

- **UnderOneFacade** (Wang et al., ECCV 2026, arXiv 2607.02018; CC BY 4.0): TLS/MLS facades from the UK, Germany and
  Singapore, 15 fine (LoFG3) / 5 coarse (LoFG2) semantic classes, no instance labels, no camera images.
- Used: the full test split (17 facades, 254M points) and 22 of 29 validation files (6 Singapore files and one 9.5 GB
  German file were not downloaded; Drive quota). Validation was used only to choose the OpenSNAP configuration.
- The local copy was deleted after the study. Drive file ids, label definitions and the dataset README are kept in
  `docs/data/underonefacade/`; re-download with the ids, then `scripts/prep_facade.py` and
  `scripts/prep_facade_instances.py`.
- Preprocessing: 5 cm voxels (7.8 cm for one 36M-point facade, GPU memory); every raw point is scored with its voxel's
  prediction through per-voxel label histograms. Raw label 16 (Singapore) -> "other", as in the authors' loader. German
  .xyz files store the label colour in their rgb columns, so their colour is zeroed for every method.
- Pseudo-instances for clicks and search: 3D-connected components of window/door/balcony/molding/decoration/column/
  arch/stairs/blinds labels (>= 100 voxels): 1,606 objects in test. These are our construction, not official labels.

## Results (test split)

| task | OpenSNAP | best alternative | verdict |
|---|---|---|---|
| semantic segmentation, LoFG3 / LoFG2 mIoU (T2d, T2e) | 9.0 / 31.8 (mask pooling); 8.1 / 28.1 full | OpenScene 3D 8.4 / 29.8; Grounded-SAM render-then-lift 6.1 / 16.6; OpenShape 4.2 / 15.4 | **best zero-shot**, far below supervised in-domain (33.4 LoFG3) |
| vs supervised, other continent (published) | 31.8 LoFG2 | 25.7–40.8 LoFG2 | inside the range without facade training |
| click segmentation, IoU@5 / IoU@10 (T5f, T5l) | 37.4 / 49.4 (SNAP: 10.1 / 10.1) | Point-SAM 64.1 / 71.9; AGILE3D 57.6 / 70.1; InterObject3D 56.8 / 70.6 | **behind**; holds with colour zeroed (Point-SAM 59.9, AGILE3D 54.1) |
| search, concept / relational AP25 (T3f) | 0.3 / 0.4 (2.5 ms per query) | Grounded-SAM 3.3 / 2.8 (UK 18.6, Germany 0) | **every system fails** |

Diagnostic: an oracle that gives each SNAP proposal its true majority class reaches only 18–21 mIoU (3 validation
facades, `results/aux/facade_oracle_val_outdoor.json`), so the limit is the proposals, not the vocabulary.

## Conclusions

- **Positive.** Object-anchored pooling is the best zero-shot semantic method on facades, about 2x the image-based
  pipelines at LoFG2 while using XYZ only, and inside the supervised cross-continent range. Negative clicks more than
  triple a positive-only segmenter's IoU@5. Queries over the stored relation take ~2.5 ms.
- **Negative.** Facade elements are parts lying on one wall surface; SNAP's object-level masks do not separate them
  (wall 44 IoU; window, balcony, molding 0). Hence part-level search fails for every zero-shot system, and dedicated
  click segmenters, above all part-trained Point-SAM, are 17–27 IoU@5 ahead. Fusing the segmenter token costs 0.9 mIoU
  against pooling alone (the token is weak here), and cracking does not help.
- **For the paper.** Present facades as a stress test and limitation: object-level semantics transfer, part-level
  interaction and search need part-aware proposals.

## Systems compared (all released weights, no facade training)

| family | systems | code |
|---|---|---|
| OpenSNAP | segmenter token, mask pooling, full, + cracking; OVQL engine | `eval_semseg.py`, `eval_interactive.py`, `facade_search.py` |
| dense 3D CLIP | OpenScene 3D (nuScenes OpenSeg model), + connected components | `eval_semseg.py`, `facade_search.py` |
| 3D shape-text | OpenShape spconv-all (OpenCLIP ViT-bigG) on OpenSNAP's own proposals | `facade_openshape.py` |
| render-then-lift | Grounded-SAM (GroundingDINO-base + SAM-H); SAM-H + SigLIP-L; RGB and geometry renders | `facade_render_lift.py` |
| interactive | InterObject3D, AGILE3D (ScanNet weights), Point-SAM ViT-L; same crops, clicks and ground truth | `click_baselines.py` |

Configuration was chosen on validation only (Outdoor segmenter, nuScenes dense model, among Outdoor/Indoor/Aerial x
nuScenes/ScanNet/Matterport); baselines use their authors' defaults. Single runs; identical interactive runs differ by
about +-0.5 IoU. Run scripts: `downloads/runs/lane_facade.sh`, `downloads/runs/lane_search.sh`; the baselines run from
`downloads/competitors/click_env` (Point-SAM needs torkit3d; OpenShape needs open_clip).
