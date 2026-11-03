# Novelty notes: what the three suggested papers leave open (2026-10-30)

Reading notes on the papers Aruni suggested, and the idea they point to. No venue is assumed; this is an experiment log.

## What each paper actually does

**BBQ: Beyond Bare Queries (ICRA 2025, arXiv 2406.07113)**
- **Map:** MobileSAMv2 masks plus DINOv2 descriptors; per-frame association by DINO cosine; DBSCAN cleanup; periodic merging.
- **Nodes:** each object gets a best view (cluster the camera positions, raycast, take the largest projection). LLaVA-1.6 captions it and EVA2 embeds it.
- **Edges:** metric edges (centre distance) and semantic edges (left/right, front/behind, above/below). The view-dependent ones use a virtual camera at the room centre looking at the anchor (ZSVG3D heuristic).
- **Deductive reasoning, 2 LLM calls:**
  1. From the captions alone, the LLM picks target and anchor ids.
  2. Edges are built only between those, and the LLM picks one final object.
- **Grounding results:** Nr3D 28.3 / 19.0 Acc@0.1 / Acc@0.25 with GPT-4o; ScanRefer 19.4 / 11.6. Spatial edges help most with same-class distractors (view-dependent +18 R@1).
- **Semantic segmentation:** Replica BBQ-CLIP 27 mIoU on its own 8-scene protocol.
- **Gaps:**
  - It returns **one object**, with no confidence, so there is no notion of "all objects that satisfy" and no way to say "none".
  - Relations are rebuilt per query by an LLM: ~2 LLM calls per query, and nothing is indexed.
  - Evaluation is Recall@1 on queries whose answer exists.

**Details Matter (ICCV 2025, arXiv 2507.23134)**
- **Recipe, combining existing pieces:** Grounded-SAM masks with 2D overlap removal; superpoint lifting (Open3DIS visibility ratios); tracking with frame-wise superpoint IoU; multi-view consensus refinement; iterative merge (IoU > 0.3) and inclusion removal (> 0.99); plus Mask3D/ISBNet proposals with NMS 0.95.
- **Classification:**
  - Alpha-CLIP with SAM masks as the alpha channel (+3.0 AP over CLIP).
  - **SMS:** per query, z-score the similarity over all proposals of the scene. Each proposal keeps its top-1 query, and proposals whose standardized max similarity is below a threshold are dropped (+2.2 AP).
- **Results:**
  - ScanNet200 top-1: 25.8 AP (2D+3D), 21.5 AP for 2D-only proposals. Top-K: 32.7.
  - Replica 2D-only: 20.8 / 32.4 / 38.5 (AP / AP50 / AP25).
- **Gaps:**
  - SMS is a **heuristic normalization standing in for calibration**: one global threshold works only because scores were standardized per scene.
  - The paper notes top-1 and top-K protocols differ by about 7 AP. AP never penalizes asking for an absent class.

**SAM 3 (ICLR 2026, arXiv 2511.16719)**
- **Prompts:** concept prompts (noun phrases and/or exemplars), returning *all* instances.
- **Detector:** DETR-style on a Perception Encoder backbone.
- **Presence token:** score(object) = p(concept present in image) × p(object matches | present). It decouples recognition from localization (+1.5 cgF1, +0.05 IL_MCC). Hard-negative phrases are the biggest factor for image-level recognition (IL_MCC 0.44 → 0.68).
- **Metric:** SAM 3 argues AP ignores calibration. It scores only predictions with confidence > 0.5 using cgF1 = 100 × pmF1 × IL_MCC, i.e. localization F1 on positive pairs gated by image-level presence MCC, with hard negatives in the benchmark.
- **SAM 3 Agent:** an MLLM proposes noun phrases to SAM 3 for complex queries.
- **Gaps:**
  - It is 2D/video only. A scan observed from many views has no presence model: "is there a kettle in this room?" needs aggregation across views and objects.
  - The weights are gated.

## The idea they point to together

All three pieces of work rank or pick under the assumption that the queried thing exists:
- BBQ evaluates R@1 on existing referents;
- Details Matter reports AP over present classes;
- our own Table V uses AP over present concepts.

SAM 3 shows that, in 2D, the step that matters for real use is **deciding presence and returning a calibrated set**. It also shows that AP hides exactly that. Our own measurements say the same thing about our system:
- vector search with the same predicates ties OVQL on AP;
- OVQL's only advantage was calibration (ECE), which AP cannot see.

Proposed direction: **answer sets, not rankings, for open-vocabulary queries over 3D object stores.**
1. **Task and metric.** Every query (concept or relational) must return the set of objects that satisfy it, possibly empty. Concepts absent from the scan are asked too, as hard negatives. Score with cgF1 / IL_MCC / absent-query FP / count error, lifted from SAM 3 to 3D instances.
2. **Method.** Calibrated, presence-factored object probabilities:
   - p(o satisfies q) = p(q satisfiable in the scan) × p(o | satisfiable). Scene-level presence aggregates evidence across objects and views; our product of experts already gives per-object probabilities.
   - For relations, presence of a witness is already the max-witness term of NEAR.
   - Hard-negative contrast sets per query, in place of a fixed background vocabulary (SAM 3's strongest factor, and our label-then-filter failure mode: "closet door" vs "door").
3. **Baselines** (one each from the three papers, plus ours):
   - cosine with a tuned global threshold;
   - SMS (Details Matter);
   - label then filter;
   - an LLM single pick in the style of BBQ, for relations;
   - SAM 3 lifted to 3D, if the weights become available.

Prior art check (web search, 2026-10-30):
- OV3D-Bench (arXiv 2608.17110, Aug 2026) makes the absent-category argument for *monocular* 3D detection. It shows that target-aware protocols inflate scores 1.1-1.9×.
- OVIP-SG (arXiv 2608.17633) tests present and absent queries on HM3D retrieval.

No paper was found that calibrates per-query presence for 3D object stores or evaluates set-valued 3D queries. These two papers are the closest and must be cited.

## Decision protocol (fixed 2026-10-30, before any further idea is tested)

**Task.** Set-valued open-vocabulary object queries over a stored scan. For every scene, ask IS(c) for every class of the dataset, present or absent. Absent concepts are the hard negatives. Each method returns a set, possibly empty. Relational NEAR(a, b) sets come second, with the same rules.

**Splits.** Nothing is selected on a test scene.
- **Dev** (all choices: radii, ontology rules, weights, thresholds, which variant to carry forward):
  - S3DIS Area 1 (44 rooms, 3D only);
  - the 30 ScanNet *train* scenes of `image_dev_a` (train[::40], 3D and image).
- **Confirmatory test**, run once per idea after it is frozen on dev:
  - ScanNet200 val (312 scenes, 200 classes, most absent in any scene; never used in this study);
  - the ScanNet val scenes of OVI-MAP's 18, for the image setting.
- **Exploratory only:**
  - Replica (8) and S3DIS Area 5 (68) were used in the pilot, so they are already contaminated for this question;
  - they are reported, but they cannot decide anything.

**Metrics.** SAM 3 protocol lifted to 3D instances (`scripts/answer_sets.py`).
- **Primary:** cgF1 = 100 · pmF1 · IL_MCC.
  - pmF1: one-to-one matching at IoU ≥ 0.5 on present concepts.
  - IL_MCC: "answer non-empty" vs "concept present", over all queries.
- **Secondary:** IL_MCC, absent-query FP rate, pmF1, count MAE, and set-level calibration (reliability of p > 0.5).
- **Guard:** AP on present concepts (the old protocol) must not drop.

**Thresholds.**
- Probabilistic methods (ours) use p > 0.5 with no tuning; that is the claim being tested.
- Every baseline gets its threshold tuned on dev, so it is applied to test exactly as a user would.
- The leave-one-scene-out test-tuned threshold is reported as an upper bound for baselines, not as the comparison.

**Baselines.** Fixed now:
- CLIP and SigLIP cosine with a dev-tuned threshold;
- SMS (Details Matter) with a dev-tuned tau;
- closed-set label-then-filter;
- OVQL with the 234-word softmax;
- the best of these is the bar.

**Uncertainty.** Paired bootstrap over scenes (1000 resamples), 95% CI of cgF1(idea) - cgF1(best baseline).

**Go.** An idea proceeds only if all of these hold:
1. Dev: it beats the best baseline; the choice is made here.
2. ScanNet200 val: Δ cgF1 ≥ +2.0 with the 95% CI excluding 0.
3. It is not worse than the best baseline on the image setting beyond its CI, with no regression beyond the CI on the exploratory sets either.
4. Guard: AP on present concepts drops by no more than 1.0.
5. A mechanism check confirms the stated cause, e.g. the median p of true objects moves toward or above 0.5 and the absent-FP rate falls.

**Drop.** The CI includes 0 on ScanNet200 val; or the gain needs a tuned threshold; or it appears only on one dataset or only in one setting (3D or image).

**Landed.** An idea "lands" when it passes Go on ScanNet200 val in both settings (3D only, and 3D + image) and survives the relational sets. One dev loop and one test run per idea; every idea and outcome is logged here before the next starts, including failures.

## Pilot (exploratory; Replica + S3DIS Area 5)

`eval_relational.py --absent 1 --full 1` (IS for every present and absent class, every object's score kept) scored by
`scripts/answer_sets.py` (`results/compare/answer_sets.json`). Replica: 376 IS queries, 226 absent. S3DIS Area 5:
612 queries, 287 absent.

| | Replica cgF1 / IL_MCC / absent-FP | S3DIS cgF1 / IL_MCC / absent-FP |
|---|---|---|
| OVQL B234 + image, p > 0.5 (untuned) | 3.0 / 0.21 / 0.01 | – |
| OVQL B234 + image, tuned t (0.15) | 17.3 / 0.43 / 0.05 | – |
| SigLIP cosine, tuned t | **17.8** / 0.45 / 0.11 | – |
| SMS / SigLIP (Details Matter), tuned | 17.2 / **0.51** / 0.20 | – |
| label then filter (closed set), SigLIP | 15.7 / 0.44 / 0.17 | – |
| OVQL B234, p > 0.5 / tuned t (0.05) | 3.5 / 7.0 | 5.6 / 19.4 |
| OVQL B4, p > 0.5 / tuned t (0.95) | 4.6 / 6.5 | 17.1 / **21.7** |
| CLIP cosine, tuned t | 8.3 | 19.0 |
| SMS / CLIP, tuned | 9.3 | 14.5 |

**Hypothesis 1 is falsified.** "OVQL's calibrated probabilities already give correct answer sets at p > 0.5" is wrong:
- untuned, it is far worse than every tuned baseline;
- even with a tuned threshold, it only ties cosine.

The low ECE was an artefact of the many easy negatives.

**Diagnosis.** True objects get a median p of 0.27 (with images) and 0.07 (3D only) under the 234-word softmax. The mass splits over near-synonyms in the vocabulary (chair/armchair/office chair, lamp/light, bin/trash can, sofa/couch, pillow/cushion). In 31 cases even the SigLIP argmax is "chair" while the fused p(chair) is below 0.5. A softmax over overlapping words is not a probability of concept membership.

**Hypothesis 2 results** (`results/compare/answer_sets2.json`). The radius 0.92 was picked by inspecting the neighbourhoods *before* any results (see the paragraph below), not tuned on these scenes.

cgF1, untuned (p > 0.5) / threshold tuned leave-one-scene-out:

| | Replica, 3D only | S3DIS, 3D only | Replica + image |
|---|---|---|---|
| OVQL B234 (softmax over words) | 3.5 / 7.0 | 5.6 / 19.2 | 3.0 / 17.3 |
| **OVQL, concept-marginalized, d = 0.92** | **10.5 / 12.8** | **23.6 / 23.8** | 11.9 / 13.2 |
| marginalized d = 0.90 / 0.87 | 11.0 / 7.4 (untuned) | 22.8 / 25.2 (untuned) | 12.5 / 10.0 (untuned) |
| best tuned baseline | 9.3 (SMS / CLIP) | 21.7 (OVQL B4, t = 0.95) | 17.8 (SigLIP cosine) |

- **3D only, on both datasets:** marginalized OVQL *without any threshold* beats every baseline *with* a test-domain-tuned threshold.
  - The tuned threshold moves back near 0.5 (0.55 on S3DIS), so calibration is restored.
  - Absent-concept false positives drop to 0.7 % on S3DIS.
- **With images (SigLIP):** no gain over tuned baselines. The image distribution is already near argmax (tau 0.02), and the neighbourhood is defined in CLIP text space, not SigLIP's.
- **Caveats:**
  - The radius is a hyperparameter. 0.85 collapses, because hypernyms like "furniture" absorb everything.
  - It must be fixed on development scenes, or replaced by an ontology.

**Hypothesis 2 (setup).** Marginalize over the concept's neighbourhood: p(o is c) = sum of p(w | o) over words w with text-sim(c, w) >= d, for d in {0.999, 0.95, 0.92, 0.90, 0.87, 0.85} (`ovql_marg*` variants). The CLIP-L text neighbourhoods look right at about 0.92 (armchair, office chair, couch, trash can, cushion, closet door). Generic hypernyms ("furniture" 0.91 to chair) also leak in, so an asymmetric ontology (WordNet / LLM hyponyms) is the principled version if this works.

## Dev selection (2026-10-30/31; `scripts/sets_lab.py`, `results/compare/sets/dev_*.{json,log}`)

Dev = S3DIS Area_1 (44 rooms, 396 queries, 137 absent) and ScanNet train[::40] (31 scenes, 6107 queries, 5615 absent).
Baselines get a dev-tuned threshold; candidates use p > 0.5. Metric cgF1 over all queries.

| Setting | best tuned baseline | best candidate (p > 0.5) | Δ |
|---|---|---|---|
| S3DIS, 3D only | label-then-filter 15.7 (t 0.95) | text:clip_l:0.9 / 0.92 excl 16.6 | +0.9 |
| ScanNet, 3D only | OVQL B234 10.0 (t 0.40) | text:clip_l:0.92 excl 13.4; wn_hyp excl 13.4 | +3.4 |
| ScanNet, + SigLIP image | OVQL + image w 0.75, 21.7 (t 0.20) | wn_hyp excl, image_w 0.5: 15.6 | **−6.1** |

- `excl` (sum over the neighbourhood, excluding hypernyms broader than the query) beats `sum` everywhere.
- CLIP-L text radius 0.92 and the WordNet-hypernym neighbourhood agree (13.4 each on ScanNet), so the gain is not one lucky radius. SigLIP-text neighbourhoods and WordNet synonyms are worse.
- **Image setting fails on dev.** All candidates under-select (absent FP 0.006–0.02 vs 0.029 for the tuned baseline), and the tuned baseline's threshold is 0.20, far from 0.5. The fused distribution is not calibrated once the sharp SigLIP expert (tau 0.02) is in. `siglip_sig` (SigLIP's own sigmoid) 5.5 and `fuse_sig` 10.3–13.8 also fail.
- Status: 3D-only would be frozen as text:clip_l:0.92 excl. Under the protocol the idea cannot pass ("not worse with images") until the image setting is fixed **on dev**.

### Image-setting follow-ups on ScanNet dev (2026-10-31; test set untouched)

- **Speed.** sets_lab was slow because of thread oversubscription on the shared box (load 230 / 128 cores; ~190 torch threads per process). `torch.set_num_threads(2)` takes a full dev grid from ~1 h per candidate to ~20 s.
- **Diagnosis.** True objects (IoU ≥ 0.5) have median p 0.06 (3D) / 0.16 (image w 0.5); only ~30 % clear 0.5. In the image setting the mass mostly goes to hypernyms or siblings and WordNet gaps (office chair→chair, trash can→bucket/bin/recycling bin, door→doorframe, book→bookshelf/shelf).
- **H3, hierarchy-consistent contrast (`mode hier`: also drop WordNet ancestors from the contrast set): falsified.** It scores 12.4–12.8 (3D) and 14.4–14.6 (image), against excl's 13.4 / 15.6.
- **Mechanism check, equal dev-tuned thresholds** (`scratchpad/lab/tuned.py`):

| | OVQL B234 tuned | marginalized excl tuned | AP B234 → marg |
|---|---|---|---|
| S3DIS 3D | 15.0 | 16.7 (clip_l 0.92) | 14.6 → 17.9 |
| ScanNet 3D | 10.0 | 13.6 / 13.7 (clip_l / wn_hyp) | 15.9 → 16.5 |
| ScanNet + image w 0.75 | 21.7 | 20.7 / 21.0 | 31.4 → 31.4 |

  In 3D, marginalization improves separability, not only calibration. With SigLIP it adds nothing at equal tuning, so the image-setting gap at p > 0.5 is a calibration failure of the fusion.
- **H4, independent-experts product** (`fuse bayes`: p ∝ p_3D · p_img^b instead of the geometric mean with weights summing to 1). Parameter-free b = 1 lifts the best candidate from 15.6 to **17.3**. With b on dev: b = 3 → 20.0, b = 4–6 → **20.1–20.2**, b = 10 → 19.3. With the same fusion and no marginalization: 19.0–19.6.
  - The gap to the tuned baseline shrinks from −6.1 to **−1.5** but does not close.
  - Absent FP rises (0.05–0.07 vs 0.029), so the remaining loss is over-selection on absent classes.
- Status: the image setting still fails the "not worse" clause on dev. Not run on test.
- **H5, SAM 3-style presence factorization** (`m_presence`: score = presence(c, scene) × p, or hard gate). Falsified on dev.
  - Gate = max of SigLIP's own sigmoid over crops: 2.9–6.8. The crop-level sigmoid is far too low, so recall collapses.
  - Gate = noisy-OR or max of p: 20.0–20.3, against 20.1 without a gate.
  - Absent FP drops slightly (0.037), but recall drops with it.

### Frozen before the test run (2026-10-31)
Dev exploration is exhausted, and more dev search risks overfitting dev. Frozen, one candidate per setting, using the same neighbourhood in both:
- 3D only: `ovql {hood: text:clip_l:0.92, mode: excl}`, p > 0.5. Dev: 13.4 vs 10.0 on ScanNet, 16.6 vs 15.7 on S3DIS.
- Image: `ovql {hood: text:clip_l:0.92, mode: excl, fuse: bayes, image_w: 4}`, p > 0.5. Dev: 20.1 vs 21.7 (expected to fail the image clause unless dev was pessimistic).
- Baselines: the full baseline list with dev-tuned thresholds. The bar is the best on dev.
- Test: ScanNet200 val, 312 scenes, run once. Exploratory: S3DIS Area_1 → Area_5.

### Confirmatory test, run once (2026-10-31; `results/compare/sets/test_*.{json,log}`)

ScanNet200 val: 312 scenes, 61,464 queries, 57,286 of them absent. The bar is the best baseline on dev, with its dev-tuned threshold.

| | candidate (p > 0.5) | bar (dev-tuned t) | Δ cgF1 [95% CI] | ΔAP | absFP cand / bar |
|---|---|---|---|---|---|
| ScanNet200 val, 3D | 13.1 | OVQL B234 7.3 (t 0.40) | **+5.8 [+4.9, +6.7]** | +0.4 | 0.037 / 0.039 |
| ScanNet200 val, + image | 19.3 | OVQL+img w 0.75, 19.0 (t 0.20) | +0.3 [−0.6, +1.2] | **−1.1** | 0.058 / 0.028 |
| S3DIS A5, 3D (exploratory) | 23.1 | label-then-filter 17.0 (t 0.95) | +6.1 [+3.5, +8.6] | +1.1 | 0.003 / 0.143 |

S3DIS A5: the strongest baseline *on test* is OVQL B234 tuned (19.2), so the gap there is +3.9.

Mechanism on test, for true objects (IoU ≥ 0.5, n = 6326): median p / fraction with p > 0.5:
- 3D: 0.070 / 0.20 (B234) → 0.144 / 0.39 (marginalized);
- image: 0.160 / 0.11 → 0.864 / 0.58.

**Verdict against the protocol:**
1. Dev: 3D ✓; image ✗ (20.1 vs 21.7).
2. 3D on ScanNet200 val: Δ ≥ +2 with CI excluding 0 ✓.
3. Image not worse beyond its CI ✓; exploratory sets ✓.
4. AP guard: 3D ✓; **image ✗ (−1.1 vs the −1.0 limit)**.
5. Mechanism: 3D ✓ (median p up, absFP slightly down). Image partial: p moves above 0.5, but absFP doubles.

→ **The idea does not "land".** The gain appears only in the 3D-only setting, which is a Drop condition for the two-setting claim. As a 3D-only result it is strong and replicates (+5.8 on 312 unseen scenes, +6.1 exploratory). With images, the best fusion only matches the tuned baseline.

## Relational answer sets, NEAR(a, b) (2026-11-02; `scripts/near_sets.py`, dev only)

**Queries** (fixed before results):
- every ordered pair of present classes, so "both present, not near" is a hard negative;
- plus one absent partner per present class in each direction.

**Ground truth:** class-a objects within a 0.5 m bbox gap of a class-b object. GT boxes are rebuilt from the raw scans on the SNAP grid; 0 mismatches against the dumps' `gt_classes`.

**Scale:** S3DIS A1 2,052 queries (1,318 empty); ScanNet dev 10,546 (7,752 empty).

**Scores:** p_a(o) · R_b(o), where R_b is the max (or noisy-OR) over near objects of p_b; `min` = compose the IS decisions. NEAR in the store is binary.
- The baseline threshold grid was widened down to 0.002, because the tuned thresholds hit the 0.05 floor.

| dev | best tuned baseline | marginalized IS, p > 0.5 (best composition) |
|---|---|---|
| S3DIS 3D | label-then-filter 8.7 | 5.9 (min_noisyor) |
| ScanNet 3D | OVQL B234 max, t 0.05: 6.0 | 5.0 (min_noisyor) |
| ScanNet + image | OVQL+img max, t 0.03: 12.3 | 12.7 (noisyor) |

→ **Relational fails in 3D on dev.**
- The IS decision rule (p > 0.5) does not compose: requiring both IS(a) and IS(b) compounds the recall loss. On test, only 39 % of true objects clear 0.5 even after marginalization.
- Composing decisions (`min`) instead of products changes nothing (±0.2).

**H6, calibrate once and decide by expected F1: falsified.** Platt (A, B) is fitted on dev IS objects (label: IoU ≥ 0.5 with a GT of the class). The answer set is the top-k maximizing E[F1] under the calibrated probabilities, with F1(∅, ∅) = 1, so the empty answer is chosen by the same rule.
- The fit gives B < 0: at object level the model is over-confident (a correct class with a poor mask counts as negative).
- EF1 sets: S3DIS 14.3 (vs 16.6 at p > 0.5); ScanNet 3D 3.5 (vs 13.4); image 12.2 (vs 20.1). Far too conservative (absFP 0.004–0.008).
- With Platt, p > 0.5 is worse still. Not carried to NEAR.

**Error budget of the frozen IS candidate on ScanNet dev 3D** (`scratchpad/lab/errors.py`):
- GT 825 → TP 231; missed with no proposal at IoU ≥ 0.5: 106 (13 %); proposal exists but p ≤ 0.5: 488 (59 %).
- FP on present queries: background (IoU < 0.25 with every annotated object) 190, other class 99, bad mask of the right class 52, duplicate 5.
- FP on absent queries: background 397, other class 278.
- Duplicates are negligible: 0.93 proposals per GT at IoU ≥ 0.5, so redundancy does not explain the calibration.
- Background FPs are small (median 312 points vs 538 for all objects), and their argmax word is usually the query itself ("storage organizer" 72, "door" 33, "closet wall" 30). These look like fragments of real objects: a mask problem, not a semantic one.
- S3DIS: 34 % of GT has no proposal at IoU ≥ 0.5. The proposal ceiling bounds any answer-set method there.

# Round 2 (2026-11-02): the three next directions, explored together

1. **Trained membership head** (SAM 3-style match/presence, trained with absent and hard negatives).
2. **Evaluation study:** whole 3D query systems as answer sets.
3. **Mask-level fixes:** fragments and the proposal ceiling.

**Infrastructure:**
- `dump_evidence.py` now stores `contain` = |o ∩ o'| / |o| between proposals, and takes `--offset`.
- Re-dumping proposals is not bit-deterministic: 185 vs 184 objects on scene0000_00, probably CUDA. Containment therefore cannot be patched onto the old dumps, so everything is re-dumped to `downloads/runs/evidence2/` (lane `downloads/runs/lane_evidence2.sh`):
  - training split ScanNet train[5::10], 120 scenes, disjoint from dev train[::40];
  - dev (ScanNet, S3DIS A1), then ScanNet200 val, S3DIS A5, Replica.
- All round-2 numbers are on evidence2, so methods share proposals.

## Direction 2: systems as answer sets, Replica OVI-MAP benchmark (`scripts/system_sets.py`, `results/compare/system_sets_replica.json`)

**Setup:** 8 scenes × the 48 instance classes = 384 IS queries, 226 of them absent; the NICE-SLAM mesh GT shared by OVO and OVI-MAP. Each system answers with the instances it outputs for label c.
- native = OpenSNAP p > 0.5, or every instance OVO outputs.
- loso = a confidence threshold tuned on the other 7 scenes (an upper bound).
- OpenSNAP uses the headline setting: grid (0, 0.6, 'S', 1, 0, 1) of replica_joint_loso_image on dump_replica_ul4. Replica is exploratory under the protocol.

| system | mIoU (known) | cgF1 native | IL_MCC | pmF1 | absFP | cgF1 loso |
|---|---|---|---|---|---|---|
| OVO-SLAM TextRegion + PE-Core | 33.4 | **17.9** | 0.533 | 0.336 | **0.221** | **20.7** |
| OpenSNAP + image (so400m) | 31.6 | 17.2 | 0.449 | **0.383** | **0.035** | 19.3 |
| OVO-SLAM default | 27.4 | 13.7 | 0.460 | 0.299 | 0.106 | 12.5 |
| OpenSNAP 3D only | – | 9.7 | 0.308 | 0.315 | 0.040 | 10.6 |

- The answer-set ranking follows mIoU, but the failure modes differ sharply. The newest OVO returns objects for **22 % of absent classes**, 6× OpenSNAP's 3.5 %; its higher IL_MCC comes from recall on present classes.
- No system exceeds ~21 cgF1 even with a test-tuned threshold. Set-valued querying is far from solved for any of them.

## Direction 1: trained membership head (`scripts/head_lab.py`), prototype on the old dev dumps

**Head:** logistic regression (and HistGBM) over 21 class-agnostic features per (object, query class), 27 with images:
- OVQL p and marginalized p, and their logits;
- scene z-score and in-scene rank of p;
- scene presence (max and 2nd-max p), hood size, log object size, log #objects;
- per expert (dense, SNAP token, image): cosine, scene z-score, rank among all labels, margin over non-hood labels, best hood cosine.

Rows are every (object, class) of a training scene, including all absent classes. Label 1 = IoU ≥ 0.5 with a GT instance of c.
Decision rules: p > 0.5, expected-F1 (`ef1_select`), fixed t ∈ {0.3, 0.2, 0.1}.

**Prototype A:** train on the old ScanNet dev (31 scenes, 510k rows, 753 positives), evaluate on S3DIS Area_1 dev (unseen class names, other sensor).
- logreg: p > 0.5 11.2, **ef1 17.1**, t 0.3 19.4, **t 0.2 20.4**, t 0.1 19.5; absFP 0.000–0.022.
- GBM: 7.1–7.9; it overfits 753 positives.
- References: best tuned baseline on S3DIS dev 15.7; round-1 candidate 16.6.

**Prototype B:** 2-fold scene CV inside ScanNet dev (15/16 scenes, ~220k rows, ~350 positives). Baselines are tuned on the training fold and applied to the held-out fold (`sets_lab.py --dev fold --test fold`).

| held-out fold | best tuned baseline | round-1 candidate | head ef1 | head t 0.2 |
|---|---|---|---|---|
| 3D a→b | 8.7 (OVQL B234) | 11.7 | 17.3 | 20.1 |
| 3D b→a | 11.6 | 15.2 | 19.2 | 23.6 |
| image a→b | 19.0 (OVQL+img) | 18.2 | 22.6 | 27.6 |
| image b→a | 23.7 | 22.3 | 27.2 | 32.7 |

- First idea in this study that beats the tuned baselines **in both settings**, by a large margin, with absFP below every baseline.
- p > 0.5 is the wrong rule for rare positives. The F1-optimal threshold for a calibrated p is F*/2, which is why t ≈ 0.2 wins. ef1 is the parameter-free version and is about 3–5 points behind t 0.2.
- Caveats:
  - tiny training sets;
  - the head sees ScanNet200 names at training; S3DIS shows transfer to unseen names;
  - the rule (ef1 vs t) must be fixed on dev before the test.
- Next: train on train[5::10] (120 scenes) once evidence2 lands. Select model and rule on dev (ScanNet dev + S3DIS dev, v2). Test once on ScanNet200 val v2.

## Direction 1 on evidence2: dev selection (2026-11-02; `results/compare/round2/{bar_*,head_dev_*}.{json,log}`)

**Training:** ScanNet train[5::10], 120 scenes, 1.94M (object, class) rows, 2,718 positives; 24 features in 3D, 30 with images (now including the 3 containment features).
**Dev:** ScanNet train[::40] (31 scenes, 6,107 IS queries, 5,615 absent) and S3DIS Area_1 (44 rooms, 396 queries, 137 absent). Baselines are tuned on the same dev sets (`sets_lab.py`); the head is not tuned on dev except for the choice of model, rule and post.

| dev set | best tuned baseline | round-1 candidate (p > 0.5) | head logreg ef1 | head logreg t 0.2 | head logreg t 0.1 | head GBM best |
|---|---|---|---|---|---|---|
| ScanNet 3D | 10.0 (OVQL, t 0.40) | 13.4 | 18.8 | **22.5** | 22.3 | 16.8 |
| S3DIS 3D (unseen names) | 15.6 (label, t 0.95) | 16.5 | 18.0 | **19.9** | 19.8 | 15.3 |
| ScanNet image | 21.7 (OVQL+img w 0.75, t 0.20) | 20.1 | 26.0 | **30.5** | 30.4 | 16.3 |

- absFP of head t 0.2: 0.020 (ScanNet 3D), 0.015 (S3DIS), 0.024 (image), vs 0.034 / 0.058 / 0.029 for the bars.
- **GBM overfits** even with 2.7k positives. Logistic regression wins everywhere.
- **Direction 3 (fragments as a post-step): no effect.** On ScanNet, drop/host change cgF1 by −0.3 to −0.5 at t 0.2. On S3DIS, host gives +0.5 (20.4) and drop +0.1. On image, drop is ±0 and host −0.5. Selected answers are rarely nested fragments once the head scores them, which agrees with the round-1 error budget (0.93 proposals per GT). Not adopted.

### Frozen before the test run (2026-11-02, before `evidence2/scannet_val` finished dumping)

- **Head:** logistic regression (C = 1, standardized features), trained on evidence2/scannet_train, all feature groups.
- **Rule:** t = 0.2, no fragment post-step. The same in 3D and with images. The value was chosen on dev; t 0.2 and t 0.1 are within 0.2 points on every dev set.
- **Bars** (best tuned dev baseline, with its dev threshold):
  - 3D: `["ovql", {"image_w": 0.0}, 0.40]`;
  - image: `["ovql", {"image_w": 0.75}, 0.20]`.
- **Test:** ScanNet200 val v2 (312 scenes), run once per setting. `head_lab.py --test ... --freeze '{"model":"logreg","rule":0.2,"post":null}' --bar ...`.
- **Go:** Δ cgF1 ≥ +2 with a paired scene-bootstrap 95 % CI excluding 0 **in both settings**, and AP of the head ≥ AP of the bar − 1 point (the guard).
- **Exploratory, after the test:** S3DIS Area_5 v2 and Replica v2 with the same frozen head (trained on ScanNet only).

### Confirmatory test, run once (2026-11-03; `results/compare/round2/test_head_{3d,img}.{json,log}`): **GO in both settings**

ScanNet200 val v2, 312 scenes, frozen head vs frozen bar; Δ with paired scene-bootstrap 95 % CI.

| setting | head cgF1 (IL_MCC, pmF1, absFP) | bar cgF1 (absFP) | Δ cgF1 [95 % CI] | AP head vs bar |
|---|---|---|---|---|
| 3D | **21.8** (0.476, 0.458, 0.023) | 7.3 (0.039) | **+14.5 [+13.1, +15.8]** | 14.8 vs 9.1 |
| image | **28.9** (0.564, 0.513, 0.024) | 19.0 (0.028) | **+9.9 [+8.8, +11.0]** | 25.3 vs 21.8 |

- All three pre-registered conditions hold in both settings: Δ ≥ +2, the CI excludes 0, and the AP guard is met (the head's AP is higher, not just within 1 point).
- **Dev → test transfer is tight for the head** (22.5 → 21.8 in 3D, 30.5 → 28.9 with images). **The tuned bars lose more** (10.0 → 7.3, 21.7 → 19.0): their thresholds were tuned on dev, while the head's only dev choice was t.
- Round 1 (marginalized OVQL p > 0.5) was a Drop: +5.8 in 3D but +0.3 [−0.6, +1.2] with AP −1.1 in the image setting. Round 2 fixes exactly that setting: +9.9 with images, on the same bar (19.0). This is the first pre-registered Go in this project, and it more than doubles round 1's 3D gain (+14.5 vs +5.8, same bar 7.3). The claim it supports: **a learned, class-agnostic membership head over per-expert evidence, with an F1-appropriate threshold, beats the best tuned single scoring rule on answer-set quality**, by a wide margin, in 3D and with images.
- Still exploratory (no decision attached): S3DIS Area_5 v2 (dumping, 47/68) and Replica v2.

### Mechanism: leave-one-group-out ablation on dev (2026-11-02; `results/compare/round2/abl_*.{json,log}`; does not change the freeze)

Logreg at t 0.2; the zeroed group is removed at train and at dev. Full model: ScanNet 3D 22.5, S3DIS 19.9, ScanNet image 30.5.

| group removed | ScanNet 3D | S3DIS 3D | ScanNet image |
|---|---|---|---|
| prob (OVQL p, marginalized p, logits) | 22.5 (±0) | 20.6 (+0.7) | 29.5 (−1.0) |
| rank (scene z, in-scene rank) | 23.0 (+0.5) | 19.7 (−0.2) | |
| presence (scene max / 2nd p) | 22.6 (+0.1) | 19.9 (±0) | 30.2 (−0.3) |
| context (hood size, object size, #objects) | 20.8 (−1.7) | 17.8 (−2.1) | |
| dense expert | 22.8 (+0.3) | 19.8 (−0.1) | |
| SNAP-token expert | **18.8 (−3.7)** | 18.5 (−1.4) | |
| image expert | | | 27.9 (−2.6) |
| containment | 21.6 (−0.9) | 20.8 (+0.9) | |

- **No single group carries the gain.** The features are largely redundant. Even without any OVQL probability the head reaches 22.5 / 20.6, so it does not depend on the hand-built OVQL scoring.
- **The largest single contributions:**
  - per-expert evidence of the SNAP token in 3D (−3.7) and of the image in the image setting (−2.6);
  - object-level context (−1.7 / −2.1): object size and how many confusable names exist.
- **Explicit presence features are redundant** (±0.3). Presence is recoverable from the per-object features, consistent with H5 (presence gating) failing in round 1.
- Containment helps ScanNet (−0.9 without) but hurts S3DIS (+0.9 without): it does not transfer. This matches direction 3's null result.
- The head without image features (27.9) still beats the image-setting bar (21.7). Much of the image-setting gain comes from re-weighting the 3D experts and the fused OVQL p, not from image cosines alone.
- **Reading:** the gain is a learned, class-agnostic **combination** of experts plus an F1-appropriate decision rule (t ≈ F*/2). It is not one new signal. That is SAM 3's recipe (learned match with hard negatives), not BBQ's or Details Matter's.

## Related work check: Chorus (Li, Ma et al., CVPR 2026 oral, arXiv 2512.17817v3), read 2026-11-03

**What it is:** a feed-forward 3DGS scene encoder (Sonata-style 5-stage transformer, 131M params). It distills three 2D teachers into one per-Gaussian embedding via per-teacher MLP projectors: SigLIP2-so400m (language), DINOv3-L (generalist) and PE-Spatial (object-aware).
- Training uses cosine + SmoothL1 matching losses, plus optional class- and instance-level InfoNCE.
- There is a point-cloud variant (centres, colours, normals) and render-and-distill adaptation.

**What it evaluates:**
- zero-shot semantic segmentation (f-mIoU; ScanNet200 22.4 trained on ScanNet only, 24.6 trained jointly);
- open-vocabulary instance segmentation on ScanNet200 with Mask3D proposals (18.0 mAP50, 13.0 tail mAP);
- linear/decoder probing and finetuning, data efficiency, and 3D VQA (ScanQA, Nr3D via GaussianVLM).

**Scope overlap with round 2:**
- None on the question: no absent-class queries, no presence or "is c here at all", no set-valued answers or thresholds, no cgF1/IL_MCC-style metrics, and no relational queries.
- Its open-vocabulary instance protocol is ranked AP over a closed list of present classes, the exact framing direction 2 argues is incomplete.
- It is a stronger **feature source**, i.e. a competitor to our per-object experts (dense CLIP, SNAP token, SigLIP image), not to the membership head.

**Release:** github.com/GaussianWorld/Chorus (CC-BY-SA-4.0), checkpoints at huggingface.co/SceneSplatPro/Chorus.
- `chorus_pts` takes coord/color/normal and was pretrained on ScanNet++ v2 only, so there is no ScanNet leakage.
- `chorus_3dgs` was pretrained on ScanNet + ScanNet++ + MP3D.
- Output `lang` is 1152-d SigLIP2-aligned per point. The released checkpoints are lang + dino (no PE head).
- About 22 GB for 500k points, which fits a 40 GB MIG slice for ScanNet scenes.

**Implication:** the trained head is encoder-agnostic (the ablation shows the gain is a learned combination of experts plus an F1-appropriate rule). The decisive follow-up is to add Chorus-pts `lang` per object as a fourth expert and test two things:
- (a) whether a tuned-threshold Chorus-cosine baseline closes the gap to the head;
- (b) whether the head still adds on top of it.

Not run yet; it needs a separate env (spconv, flash-attn) and the SigLIP2 text tower.

### Learning curve of the head (2026-11-03; `results/compare/round2/curve_3d_*.{json,log}`; dev only)

Logreg at t 0.2, trained on random subsets of evidence2/scannet_train (2 seeds each).

| training scenes | positives | ScanNet dev 3D | S3DIS dev 3D |
|---|---|---|---|
| 15 | ~270 | 21.4 / 21.9 | 19.3 / 19.5 |
| 30 | ~650 | 22.0 / 20.2 | 19.9 / 20.5 |
| 60 | ~1,300 | 22.0 / 22.1 | 19.8 / 19.7 |
| 120 | 2,718 | 22.5 | 19.9 |

- **Saturated at about 15–30 scenes.** The linear head over 24 evidence features is limited by its evidence, not its data.
- More training scenes (the SceneSplat-49K lesson: 280 → 3.5k scenes lifts their encoder) will **not** lift the head as it is. Lift has to come from better per-object evidence or better proposals.
- GBM stays unstable (10–17) at every size.

## Ideas from the SceneSplat line that could lift the head (2026-11-03; read: ShapeSplat 3DV'25, SceneSplat ICCV'25, SceneSplat++ / 49K NeurIPS'25 D&B, Chorus CVPR'26)

Mapped onto the round-1 error budget: 59 % of GT covered by a proposal but scored low; 13 % (ScanNet) / 34 % (S3DIS) with no proposal; FPs from small fragments.

1. **A feed-forward 3D language encoder as a fourth expert** (SceneSplat / Chorus `lang`, SigLIP2-aligned, per point). Targets the 59 % "covered but scored low" bucket.
   - Evidence: Chorus beats Mosaic3D on ScanNet200 tail classes (13.0 vs 5.4 tail mAP), and both papers show the encoder beats its own lifted pseudo-labels (SceneSplat Tab. 5; SceneSplat++ in 75 % of settings), i.e. cross-scene training denoises per-scene evidence.
   - Our ablation says the SNAP-token expert alone is worth 3.7 points, so a stronger 3D expert is the most direct lever.
   - Released: `chorus_pts` (coord/color/normal, trained on ScanNet++ v2 only) and `chorus_3dgs` (needs 3DGS scenes; SceneSplat-7K hosts ScanNet 3DGS under the ScanNet terms).
2. **SceneSplat's three-crop dynamic weighting** for image evidence: global frame + crop with background + masked crop without background, weighted by their agreement. It adapts context to integrated vs isolated objects. Plus SigLIP2 in place of SigLIP.
   - Targets the image expert (worth 2.6 points in the ablation). Cost: an image re-dump only.
3. **Object-aware features (Chorus DINO / PE heads, ShapeSplat's attribute-aware grouping) for proposal affinity.** Targets the no-proposal bucket (13 % / 34 %), which the head cannot touch. The largest change; touches c2f.
4. **A richer pooled descriptor per object**: ShapeSplat's learned splats pooling, or simply max/percentile pooling of per-point similarity rather than a mean. Small objects inside large proposals are then not washed out. Needs per-point similarities at dump time.
5. **Not useful to us as is:** MAE/self-distillation pretraining and 3DGS-aware augmentations (encoder-training tricks); outdoor scaling. More data for the head (see learning curve).

Any of these is a new idea under the protocol: dev selection, a new freeze, and a test once.

# Round 3 (planned 2026-11-03): better evidence for the membership head

**Motivation:** the head saturates at ~30 training scenes, so the lever is evidence, not data. Ideas from the SceneSplat line, each targeting one bucket of the error budget.

**Hypotheses (each is an added expert block for the head, 5 class-agnostic features like the existing experts):**
- **R3a, pooling** (ShapeSplat's point): `dense_max`, `dense_p90`, the max and 90th percentile of per-voxel OpenScene cosine inside the object, next to the mean-pooled `dense`. Target: small or partial objects whose mean is washed out.
- **R3b, three-crop image evidence** (SceneSplat label collection): `image3c_so400m` (same encoder as `image`, new crops) and `image3c_siglip2` (SigLIP2-so400m-p16-512, the Chorus teacher). This separates the crop strategy from the encoder. Target: image-expert errors (2.6 points in the ablation).
- **R3c, Chorus** (`chorus_pts`, ScanNet++ v2 pretraining only): `chorus` (mean-pooled SigLIP2-aligned feature) and `chorus_p90`, via `scripts/chorus_expert.py` on the stored masks. Target: the "covered but scored low" bucket (59 %). Needs a separate env (pending: installing external packages needs the user's approval).

**Data:** v3 dumps (`downloads/runs/evidence3/`), with proposal masks stored so later experts reuse exactly the same objects. Splits:
- train = the first 60 scenes of ScanNet train[5::10] (the curve shows 60 suffice);
- dev = ScanNet train[::40] (31) and S3DIS Area_1 (44);
- test = ScanNet200 val (312), dumped only after the dev choice is frozen.

**Fair baselines:** every new expert also enters `sets_lab.py` as tuned-threshold cos and sms baselines (`--extra`). A strong encoder alone with a tuned threshold must not close the gap.

**Selection on dev:** forward selection over {R3a, R3b, R3c} blocks on top of the round-2 head (logreg, t 0.2). A block is kept only if it adds ≥ +1.0 cgF1 on ScanNet dev without losing > 0.5 on S3DIS dev. Then the frozen test, the same Go/Drop as round 2, against both the best tuned baseline (now including the new experts) and the frozen round-2 head.

## Round 3 log

**2026-11-03, v3 smoke test** (ScanNet scene0000_00, 183 objects, 47 matched to GT at IoU ≥ 0.5). Top-1 of the GT label among the 234 labels:

| evidence | top-1 | median rank |
|---|---|---|
| dense (mean-pooled) | 0.06 | 18 |
| dense_max | 0.04 | 15 |
| dense_p90 | 0.13 | 13 |
| image so400m, current multi-scale crops | 0.55 | 0 |

The p90 pooling looks useful for the dense expert. One scene only; the head on dev decides.

**Crop diagnostic** (same scene; `views.object_embeddings_3crop(parts=True)`, scratch script `diag/crops.py`):

| crops | so400m top-1 / top-5 | SigLIP2 top-1 / top-5 |
|---|---|---|
| current multi-scale (0 / 10 / 20 % pad) | **0.55** / 0.70 | 0.49 / **0.79** |
| crop with background (20 % pad) | 0.51 / 0.72 | 0.43 / 0.74 |
| masked crop, no background | 0.06 / 0.23 | 0.26 / 0.47 |
| SceneSplat three-crop fusion | 0.38 / 0.68 | 0.40 / 0.72 |
| local + masked | 0.43 / 0.70 | 0.49 / 0.77 |

- **SceneSplat's three-crop fusion loses here.** Its masked crop is near chance at ScanNet's 320×240 frames: masks come from sparse projected points, not SAM2 masks on full-resolution frames as in SceneSplat. The fusion drags the plain crops down.
- So R3b is reduced to:
  - an encoder swap with the crops fixed (`image_siglip2_ms`);
  - local + masked (`image_*_lm`) as a complementary expert.
- Both are added after the dump from the stored masks (`scripts/image_posthoc.py`).

**v3 dump settings:** v2 settings + `--pool_stats 1 --masks 1` (lane `downloads/runs/lane_evidence3.sh`; train = the first 60 of train[5::10], dev as before). The post-hoc image lane is `downloads/runs/lane_round3_dev.sh`.

**R3c status:**
- `env_chorus` is installed (torch 2.5.1 + cu124, spconv, flash-attn 2.8.3, chorus 0.1.0).
- The checkpoints at huggingface.co/SceneSplatPro/Chorus are **gated**. The account behind `.env`'s hf_key is not on the authorized list (401). The user has to request access on that page; R3c waits on it.

**2026-11-03, R3c unblocked.** The user was granted access to SceneSplatPro/Chorus; `lang-dino-enc-pretrain-ppv2-mcmc-from-pts-params.pth` is downloaded.

Smoke test on scene0000_00 (the same 47 matched objects as above):

| evidence | top-1 / top-5 | median rank |
|---|---|---|
| dense_p90 | 0.13 / 0.30 | 13 |
| **chorus** (mean-pooled) | **0.28 / 0.49** | 5 |
| **chorus_p90** | **0.36 / 0.60** | 1 |
| image so400m | 0.55 / 0.70 | 0 |

- Chorus is the strongest 3D-only evidence we have, 3–6× our dense expert at top-1. Its typical error is the generic label "object" (stool, guitar, backpack → "object"), which the head's margin and hood features can discount.

**Incident:** the first v3 training dump died on CUDA OOM (the shared slice also held other users' jobs and the v2 val dump). It wrote 0 scenes; the log is kept as `evidence3/scannet_train.oom1.log`. Fixes:
- `dump_evidence.py` now retries OOM 10× with 3 min waits;
- the training dump restarts after the v2 val dump finishes (`lane_evidence3b.sh`);
- DEV_TRAIN_DONE is written only when all 60 training scenes exist.

Chorus and the dev runs are queued in `lane_round3_chorus.sh`.

**Incident 2 (2026-11-03):** the v3 dev dumps, which ran under the old 3-retry code, also died on OOM: ScanNet dev reached 7/31 and S3DIS dev 9/44. They still logged EVIDENCE_DONE, so lane 3b would have marked the data complete with partial dev sets. Fix:
- `lane_evidence3c.sh` replaces lane 3b's tail. It resumes both dev dumps on the second slice (finished scenes are skipped) while the train dump runs.
- DEV_TRAIN_DONE is written only when train 60/60, ScanNet dev 31/31 and S3DIS dev 44/44 all exist; otherwise it writes INCOMPLETE.

The round-2 test came in meanwhile (Go, see above). The exploratory S3DIS A5 / Replica runs of the frozen round-2 head are queued in `lane_head_explore.sh`.

**Incident 3 (2026-11-03, about 04:50):** `image_posthoc.py` crashed on the first record (StopIteration). It took the scene split from `eval_queries.DATA`, which lists only ScanNet `val`, but the v3 dumps are train scenes. The lane still wrote POSTHOC_DONE. The Chorus lane was stopped while still waiting, so it never used the incomplete records. Fixes:
- The ScanNet lookup now searches train and val.
- The marker is now POSTHOC_OK, written only if both processes print IMAGE_POSTHOC_DONE.
- The failed logs are kept as `round3/posthoc_*.fail1.log`.

**Round-2 exploratory runs on partial data:** the v2 S3DIS A5 and Replica dumps also died of OOM (61/68 rooms, 4/8 scenes). The first runs of the frozen head on them are kept as `round2/explore_*.partial.*`:

| set (partial) | head cgF1 | bar cgF1 | Δ [95 % CI] | AP head vs bar |
|---|---|---|---|---|
| S3DIS A5, 3D (61/68 rooms) | 26.6 | 16.8 (label t 0.95) | +9.8 [+6.6, +13.1] | 14.0 vs 10.1 |
| Replica, 3D (4/8 scenes) | 9.7 | 3.6 (ScanNet bar) | +6.1 [+1.1, +8.7] | 25.0 vs 19.6 |
| Replica, image (4/8 scenes) | 13.2 | 14.9 (ScanNet bar) | −1.7 [−6.7, +2.6] | 32.4 vs 32.1 |

- Replica with images is the first setting where the head does not win. It is 4 scenes, the CI is wide, and the bar's threshold was tuned on ScanNet, not Replica. Not interpretable until the full set is in.
- `lane_explore2.sh` resumes both dumps once the round-3 dev runs finish, then reruns `lane_head_explore.sh` on the full sets.

### Round 3 dev results, ScanNet dev (2026-11-03 08:13; `results/compare/round3/{bar,head}_*.{json,log}`)

Training is evidence3/scannet_train, 60 scenes. The learning curve showed saturation by 30 scenes. Logreg; cgF1 at t 0.2, with t 0.1 in brackets.

| block added to the round-2 features | ScanNet 3D | Δ vs base | ScanNet image | Δ vs base |
|---|---|---|---|---|
| none (base; reproduces round 2's 22.5 in 3D) | 22.5 (22.6) | | 29.0 (30.1) | |
| sharper pooling (dense_max, dense_p90) | 22.6 (22.6) | +0.1 | | |
| Chorus (chorus, chorus_p90) | **24.2** (24.8) | **+1.7** | 29.2 (30.0) | +0.2 |
| pooling + Chorus | **24.6** (25.1) | **+2.1** | | |
| SigLIP2 multiscale crops | | | 30.1 (31.0) | +1.1 |
| all image variants (SigLIP2 ms + lm, so400m lm) | | | 30.4 (31.8) | +1.4 |
| everything (dense_p90, Chorus, all image) | | | **30.7** (31.2) | **+1.7** |

**Tuned baselines with every new expert:**
- **3D:** the best is now a Chorus rule, `sms chorus_p90` at 11.3 (`sms chorus` 10.3, the old OVQL bar 10.1). Chorus alone as a scoring rule is the strongest 3D baseline we have, but it is still half the head's score.
- **Image:** unchanged. The OVQL+img bar stays best at 21.8; the best new-crop rule, `sms image_siglip2_lm`, scores 17.6.

**Reading (ScanNet only; the S3DIS gate is still pending):**
- **Sharper pooling is null** (+0.1 on ScanNet, −0.1 on S3DIS): **dropped**.
- **Chorus passes the ScanNet gate in 3D** (+1.7 alone, +2.1 with pooling). With images it adds nothing (+0.2): the image expert already carries what Chorus distills from SigLIP2. This is consistent with Chorus being 2D-teacher distillation.
- **The image variants pass the ScanNet gate:** +1.1 for SigLIP2 ms alone, +1.4 for all three. They are ScanNet/Replica only (S3DIS has no posed frames), so the S3DIS gate does not apply to them.
- The image base (29.0) is lower than round 2's 30.5 at 120 training scenes. The image setting may not saturate as early as the 3D curve did. Not tested.

**Incident 4:** Chorus on S3DIS dev hit OOM on the first room, which had over 1M raw points (S3DIS rooms are 0.4–3.8M points; ScanNet scenes about 80–150k). It wrote nothing. As a result, `bar_s3dis_3d` crashed (KeyError `chorus`), and the S3DIS rows of the Chorus heads were scored with zero features (17.5 / 18.7, an artifact). These are kept as `*.fail1.*`.
- Fix: `chorus_expert.py --voxel_input 1` encodes one averaged point per dump voxel; above 250k voxels it encodes a random 250k subset and copies features by nearest neighbour. It also retries OOM 10×.
- `lane_round3_s3dis.sh` reruns the S3DIS bar and the two Chorus heads.
- Note: S3DIS Chorus input is therefore voxel-averaged, unlike ScanNet's raw points. This deviation is forced by memory.

**S3DIS gate (2026-11-03 08:34; Chorus on voxel input; the fail1 rows are superseded).** Head at t 0.2 on S3DIS dev:
- base 20.0;
- + pooling 19.9 (−0.1);
- **+ Chorus 20.2 (+0.2)**;
- + pooling + Chorus 20.2 (+0.2).

S3DIS tuned baselines with the new experts: the best is now `sms chorus` at 16.7 (`sms chorus_p90` 16.3, label t 0.95 15.7). Chorus is again the best single scoring rule, but 3.5 below the head.

**Forward selection, as pre-registered:** keep a block if it adds ≥ +1.0 on ScanNet dev and loses ≤ 0.5 on S3DIS dev.
- **3D:**
  - step 1: R3c Chorus +1.7 (S3DIS +0.2) → **keep**; R3a pooling +0.1 → drop.
  - step 2: R3a on top of R3c, +0.4 → drop.
- **Image (S3DIS gate not applicable):**
  - step 1: R3b image variants +1.4 → **keep**; R3c +0.2.
  - step 2: everything else on top of R3b (dense_p90 + Chorus), +0.3 → drop.

### Frozen before the round-3 test (2026-11-03 08:40, before `evidence3/scannet_val` exists)

- **Heads:** logreg (C = 1, standardized), rule t 0.2, no post, trained on evidence3/scannet_train (60 scenes).
  - 3D: round-2 features + `--extra chorus chorus_p90`.
  - Image: round-2 image features + `--extra image_siglip2_ms image_siglip2_lm image_so400m_lm`.
- **Bars, the best tuned dev baseline including the new experts:**
  - 3D: `["sms", {"feat": "chorus_p90"}, 1.54]` (ScanNet dev 11.3);
  - image: `["ovql", {"image_w": 0.75}, 0.20]` (21.8, unchanged).
- **Reference:** the frozen round-2 head (logreg t 0.2, round-2 features, trained on evidence2/scannet_train, 120 scenes), applied to the same v3 test records. Δ uses a paired scene bootstrap.
- **Test:** ScanNet200 val v3 (312 scenes; masks, then image post-hoc, then Chorus), run once per setting.
- **Go for round 3 (the new evidence):**
  - Δ vs the round-2 head ≥ +1.0 with a 95 % CI excluding 0 in **at least one** setting, and the other setting not significantly worse;
  - plus the round-2 conditions against the bar (Δ ≥ +2 with the CI excluding 0, AP ≥ AP_bar − 1) in both settings.
  - The round-3 dev gains are +1.7 (3D) and +1.4 (image), so the +1.0 test threshold is not a gift.
- **Expected from dev, stated before the test:**
  - **3D:** vs the reference (round-2 head at 120 scenes, 22.5 on ScanNet dev), the round-3 3D head at 24.2 should show about **+1.7**.
  - **Image:** the round-2 image head at 120 scenes scored 30.5 on dev, and the round-3 image head scores 30.4 at 60 scenes. So the image comparison should come out at about **0**: the new crops buy what the extra 60 training scenes bought.
  - In practice round 3 rides on 3D/Chorus.

### Round-2 exploratory runs on the full sets (2026-11-03 08:24; `round2/explore_*.{json,log}`; supersedes `*.partial.*`; no decision attached)

Frozen round-2 head (ScanNet-trained, logreg t 0.2):

| set | head cgF1 | bar cgF1 | Δ [95 % CI] | AP head vs bar |
|---|---|---|---|---|
| S3DIS A5, 3D (68 rooms) | **26.8** | 17.0 (label t 0.95, S3DIS-dev tuned) | **+9.8 [+6.6, +13.0]** | 15.4 vs 10.4 |
| Replica, 3D (8 scenes) | **10.2** | 4.7 (ScanNet-tuned OVQL t 0.40) | **+5.5 [+1.1, +9.5]** | 22.4 vs 18.8 |
| Replica, image (8 scenes) | 14.4 | 16.1 (ScanNet-tuned OVQL+img w 0.75, t 0.20) | −1.8 [−6.4, +3.0] | 29.9 vs 28.7 |

- **In 3D the head transfers to both unseen datasets.** On S3DIS A5 this is with unseen class names.
- **Replica with images is a no-win (not a loss; the CI spans 0).** It is the one place the head does not beat the tuned rule. Possible reasons, not tested:
  - Replica's rendered frames make the image expert much stronger relative to the 3D experts than on ScanNet, and a head trained on ScanNet's noisy 320×240 frames under-weights it;
  - 8 scenes give a wide CI.

  A candidate for the round-3 image variants (the SigLIP2 crops) is Replica, after the round-3 test.

**`--ref` smoke test on dev** (ScanNet dev v3 as the test spec; not the test):
- the reference round-2 head scores 22.8 (22.5 on the v2 dev records);
- the round-3 3D head scores 24.2, Δ **+1.4 [+0.0, +2.7]** on 31 scenes;
- vs the new Chorus bar, +12.8 [+9.4, +16.7].

So the round-3 3D gain over round 2 is real but small on dev; the 312-scene test decides. `lane_round3_test.sh` started 08:45: it dumps val v3, then runs the image post-hoc, then Chorus, then the two tests.

## Where the head hits the wall: error budget of the round-3 heads on dev (2026-11-03; `scripts/head_errors.py`)

GT instances of present queries, logreg t 0.2:

| bucket | ScanNet 3D (+Chorus) | ScanNet image (+crops) | S3DIS 3D (+Chorus) |
|---|---|---|---|
| hit | 36 % | 43 % | 34 % |
| **covered, another class wins the object** | **37 %** (34 other + 3 hood) | **30 %** | 15 % |
| covered, low for every class ("weak") | 14 % | 15 % | 16 % |
| **no proposal at IoU ≥ 0.5** | 13 % | 13 % | **34 %** |

- **FPs on present queries:** background/fragment plus partial mask = 66 % (ScanNet 3D), 72 % (image), 93 % (S3DIS). Wrong-name FPs are 29 / 25 / 7 %.
- **FPs on absent queries** are mostly real objects of another class: 73 % on ScanNet 3D, 73 % with images.
- **Top confusions** (query → winning class):
  - ScanNet 3D: bottle→object 15, office chair→chair 14, file cabinet→cabinet 7, end table→coffee table 7, box→object 6, mirror→window 5, bookshelf→shelf 5, basket→object 5, dresser→cabinet 4.
  - S3DIS: beam→door 16, bookcase→door 12, column→door 11, bookcase→table 10.
  - The clip_l 0.92 hood is narrow, so most hypernym/sibling confusions land in "other" (office chair→chair is not in the hood).

**Reading:**
- On ScanNet the wall is **naming**: specific names lose to hypernyms, siblings and the generic "object".
- On S3DIS it is the **proposal ceiling** (34 %).
- Adding experts adds little because every expert inherits CLIP/SigLIP-family hypernym confusion.
- Chorus's lang head is trained only to regress per-point SigLIP2 *image* features (cosine + L1, plus a 0.02-weight segment contrastive term; `configs/chorus/lang-dino-enc-pretrain-ppv2-*.py`). No text enters its training, so nothing in it can learn "office chair beats chair on an office chair".

## Round 4 (planned 2026-11-03): modify Chorus, don't just plug it in

- **M1 (first): a set-trained language head on the frozen Chorus trunk.**
  - Replace the image-regression lang projector with one trained on proposal objects against the text of the whole vocabulary.
  - Sigmoid membership loss (SAM 3-style), with hard negatives from the measured confusion structure (WordNet hypernyms, siblings, "object") and with the scene's absent classes.
  - Initialized from Chorus's own projector.
  - Open-vocabulary hygiene:
    - split the ScanNet200 names into seen and held-out halves; M1 trains on seen names only;
    - report on held-out names of ScanNet dev/val, and on S3DIS and Replica (all unseen);
    - M1's features enter the class-agnostic head like any expert.
  - Targets the 37 % "another class wins" bucket and the absent-query FPs.
- **M2 (second): Chorus trunk/DINO features for proposal grouping.** Targets the S3DIS 34 % ceiling and the fragment FPs. A larger change, inside c2f.
- **Step 0 (shared by both):** pool Chorus `backbone_upcast` (trunk) features over the stored proposal masks of the v3 train/dev dumps.

### Round 4 M1 setup (2026-11-03; `scripts/{chorus_trunk,chorus_gt,m1_lang}.py`, lanes `lane_round4_{trunk,gt,m1}.sh`)

- **Data finding:** the 60 head-training scenes hold only 1,293 matched objects over 139 classes (about 650 on the seen half). That is too few to retrain a 10M-parameter projector. Note: "object" is itself an annotated ScanNet200 class (62 of them), so bottle→object is a confusion between two annotated classes.
- **M1 training data:**
  - Chorus trunk (`backbone_upcast`, 1232-d) mean-pooled over **GT instances**, one scan per ScanNet train room.
  - Every room of the head's training scenes and the dev scenes is excluded (91 rooms excluded, 474 eligible), so no room, and no rescan of one, is seen by both M1 and the head or dev.
- **M1 model and loss:**
  - The projector is initialized from Chorus's lang projector (LN → 4928 → 1152).
  - Sigmoid membership loss over all training names, with SigLIP2 text (the Chorus teacher).
  - Negatives up-weighted 4× for WordNet ancestors, siblings, and names with clip_l text cos ≥ 0.85 to the positive.
- **Class split** (`round4/split.json`, seed 0): the 197 queried ScanNet200 classes are halved into seen and held-out.
  - Held-out names and instances never enter M1, as positive or negative.
  - Held-out queries are also dropped from head training (`head_lab.py --class_split`).
  - Dev cgF1 is reported on seen, held-out and novel names (S3DIS names that are not ScanNet classes).
  - M1 model selection uses seen-class loss on 10 % held-apart GT rooms.
- **Ablation built in:** `m1_frozen` = Chorus's own projector applied to the pooled trunk, without training. It separates "pool then project" from the set training.
- **Success on dev (stated now):**
  - M1 must beat the Chorus expert on **held-out** names, in head cgF1 on ScanNet dev, by ≥ +1.0, and must not lose more than 0.5 on S3DIS.
  - A gain on seen names only would mean M1 memorized the vocabulary, not that the modification works.
- **Split content (seed 0, 99 seen / 98 held):**
  - seen includes chair, office chair, door, window, pillow, lamp, bed, file cabinet, coffee table;
  - held includes table, object, cabinet, couch, desk, shelf, box, bottle, bookshelf, end table.
  - So the measured confusions fall into three kinds:
    - learnable directly (office chair vs chair, both seen);
    - half-seen (file cabinet seen, cabinet held; end table held, coffee table seen);
    - fully zero-shot (bottle/object, bookshelf/shelf).
  - Because "object" is held, it is not among M1's hard negatives.
- **Engineering incidents:**
  - Encoding the vocabulary on the shared CPU took minutes per call. The SigLIP2 text is now cached once, on GPU (`trunk3/text_siglip2.pt`, 258 names).
  - The first sibling rule made about 43 names "hard" per positive. It is now restricted to WordNet subtrees of ≤ 15 labels: about 4.5 per positive, e.g. office chair → {chair, armchair, seat, sofa chair, folded chair, furniture}.
  - The trunk lane hit persistent OOM next to the val dump (scene0126_00, WC_1). It resumes after the GT extraction (`lane_round4_trunk2.sh`), and M1 waits for both.
- M1 smoke test (6 scenes, 2 epochs) passed end to end.

**Incident 5 (2026-11-03, found at 18:17): silent stalls for about 6 h.**
- **Slice A is shared with other users' jobs** (about 18 GB of 40 GB taken), and every Chorus process gave up after 10 OOM retries:
  - Chorus on val died at scene0207_02;
  - both GT-extraction shards died at 61/474;
  - the trunk resume died again.
- **The image post-hoc on val ran at about 3 min/scene** (110/312 after 5.8 h).
- **The lanes waited on markers that could no longer come**, and my waiters had no liveness or failure check. A 7-day-old waiter for an OpenShape / XYZ-only rerun (from before round 2) could never fire either: its `pgrep` matched its own command line. It was stopped; the rerun was not done.
- **Recovery:**
  - Chorus scripts take `OOM_RETRIES` (40 = 2 h).
  - `image_posthoc.py --reverse` lets a second process walk the val records backwards on slice B.
  - `lane_round3_test2.sh`: both walkers → a final post-hoc pass → Chorus on val → the frozen test.
  - `lane_round4_m1b.sh`: GT extraction → trunk features → M1 → dev, all Chorus work serial on slice B.
  - The waiter now also fires on any failure marker, and after 2 h regardless.

### Incident 6 (2026-11-03 22:50): one GT room too large for Chorus
GT extraction for M1 stopped at 334/474 rooms: `scene0497_00` (518k points, ~5x a typical room) OOMed on all 40
retries on slice B, which had room to spare. It is size, not contention. Fix: `chorus_gt.py --max_points 250000`
(a first try at 300k still OOMed);
larger rooms are encoded on a random 250k-point subset and every point takes its nearest encoded point's features
(same scheme as S3DIS `--voxel_input`). The 334 rooms already done were encoded uncapped and are kept; only
remaining rooms above 250k points are subsampled.
Lane relaunched at 22:55 (old logs kept as `round4/gt_all.fail{1,2}.log`, `round4/m1b.fail1.log`). Round-3 test lane
unaffected: forward image post-hoc finished, the reverse walker is closing the last scenes.
