# OpenSNAP: An Open-Vocabulary Object Database for 3D Scans

*An explained version of the work: what the system is, why each piece exists, how every operator works, and what the
experiments show. All numbers come from [`results/TABLES.md`](results/TABLES.md) (table IDs in brackets, e.g. [T5f]);
the analysis notes are in [`results/RESULTS.md`](results/RESULTS.md); implementation details in
[`docs/TECHNICAL_NOTES.md`](docs/TECHNICAL_NOTES.md).*

---

## Abstract

3D scans of rooms, buildings and streets are collected at scale, but they remain raw point sets: every semantic
question — *where are the chairs? which monitor is next to the bottle?* — today means re-running a segmentation network
over the whole scan with a fixed list of class names. We present **OpenSNAP**, which turns a scan into an
**open-vocabulary object database**. A scan is ingested once into an **object relation** — one tuple per object with its
geometry, a learned segmenter token and mask-pooled 3D vision–language evidence. Open-vocabulary predicates such as
`IS('whiteboard')` receive **calibrated probabilistic semantics**, and an **open-vocabulary query language (OVQL)**
combines them with spatial (`NEAR`, `ABOVE`, `ON`) and boolean operators. Queries run on the stored relation in under a
millisecond per room (vs. 8.8 s to re-process the scan), scale to a 68-room building in 0.9–3.4 ms, and are refined
lazily by **semantic cracking**, which materialises objects the first query could not explain. The operators that
populate the relation come from SNAP, a promptable point-cloud segmenter, which we extend in four ways: object-anchored
open-vocabulary evidence (58.2 vs. 41.5 mIoU on ScanNet), click-to-name with a vision–language model (40.1% vs. 30.7%),
**negative clicks** for a model trained only on positive ones (+4 to +15 IoU after five clicks across five datasets),
and **phone-video ingest** with a guard against out-of-distribution features. We evaluate on seven datasets — ScanNet,
ScanNet200, Matterport3D-160, S3DIS, Replica, nuScenes and SemanticKITTI — in 32 tables.

---

## 1. Introduction

### 1.1 The problem
Point clouds are everywhere: indoor RGB-D scans (ScanNet, Matterport), laser scans of buildings (S3DIS), and LiDAR
sweeps from cars (nuScenes, SemanticKITTI). What people want from them are **objects and their relations**: "find every
fire extinguisher on this floor", "which chairs are near a whiteboard", "what is this thing I clicked on". Two things
make this hard.

1. **Vocabulary.** Classical 3D segmentation predicts a fixed label set (20 classes for ScanNet, 13 for S3DIS). A user's
   question uses arbitrary words — *whiteboard*, *couch*, *thermos* — most of which were never training labels.
2. **Cost.** A network forward pass over a scan takes seconds. If every question re-runs the network, a building with
   dozens of rooms and a workload of hundreds of questions is far from interactive.

### 1.2 Our view: a scan is a database
We treat a scan the way a database treats a document collection: **ingest once, query many times**. Ingest extracts
*objects* (not points, not classes) and stores, for each object, everything needed to answer open-vocabulary questions
later. Queries are then cheap algebra over this relation. This raises database questions that vision papers do not
ask: What is the schema? What does an open-vocabulary predicate *mean*, as a probability? How are boolean and spatial
operators composed? What happens when the stored objects cannot answer a query? How does it scale beyond one room?

### 1.3 Starting point: SNAP
SNAP (Gupta et al., 3DV 2026) is a *promptable* segmenter for point clouds, the 3D analogue of Segment Anything: given
one or more clicks on a scan, it returns a mask for the clicked object. It works across indoor, outdoor and aerial
scans, which makes it an excellent object extractor. But it has three gaps for our purpose:
* its **text head is closed-vocabulary** (trained on the class names of its training sets);
* it accepts **positive clicks only**, so it cannot be told "not this part";
* it has **no notion of storage or queries** — every use is a fresh forward pass.

### 1.4 Contributions
1. **Data model.** An *object relation* materialised once per scan (and a building-level union with a room partition),
   with provenance and a record of query-time refinements (§3).
2. **Semantics.** A calibrated probability for open-vocabulary predicates that fuses the segmenter's token with
   object-pooled 3D CLIP evidence, plus a *vocabulary relay* for words outside the segmenter's vocabulary (§4).
3. **Query language and processing.** OVQL with concept, spatial and boolean operators under tuple-independent
   probabilistic semantics; a two-phase plan; room-partitioned batched spatial joins; and *semantic cracking*, an
   adaptive-indexing scheme that materialises missing objects on demand (§5).
4. **Better operators.** Click-to-name by *3D-propose / VLM-choose*; a text-prompt decoder; **negative clicks** for a
   positive-only segmenter via test-time token tuning, a corrective decoder and backbone adapters (§6).
5. **In-the-wild ingest.** Phone video → metric, gravity-aligned point cloud; a *dense-feature collapse guard*; and
   *photo-grounded naming* that names objects from the real video frames (§7).
6. **Evaluation** on seven datasets: query accuracy, latency, storage, building scale, interactive segmentation and
   naming (§8).

---

## 2. Background

**Point Transformer V3 (PTv3)** is the backbone inside SNAP: it voxelises the cloud (2 cm indoor, 5 cm outdoor),
serialises voxels along space-filling curves and applies attention within patches, producing one feature vector per
voxel.

**SNAP's decoder** takes the voxel features and a set of click prompts. Each click becomes a *prompt token* (a positional
encoding of the click plus the backbone feature at the nearest voxel). Learned output tokens (a mask token, an IoU
token and a text token) attend to the prompts and to all voxel features through a two-way transformer. The final mask is
a dot product between refined voxel embeddings $E \in \mathbb{R}^{N\times D}$ and the mask token $m \in \mathbb{R}^D$:
$z_i = E_i^\top m$, voxel $i$ is in the mask iff $z_i > 0$. The text token is projected into CLIP space and compared to
class-name embeddings; this is SNAP's (closed) text head. Important for us: SNAP is trained on XYZ only — colour and
normals are zeroed before normalisation — so we feed XYZ only; with colour, IoU@1 drops from 53.3 to 48.6.

**OpenScene** distils 2D vision–language features (LSeg or OpenSeg, i.e. CLIP-aligned) into a 3D sparse-convolution
network, giving every point a CLIP-space feature from geometry alone. It is strong at "how much does this point look
like a chair" but has no notion of objects.

**CLIP** maps images and text into one embedding space; a class name is embedded with a prompt template ("a {} in a
scene") and compared by cosine similarity.

**Qwen2.5-VL** (3B / 7B) is a vision–language model we use to name objects from images.

**VGGT** is a feed-forward model that predicts per-pixel depth, camera poses and 3D point maps from a set of video frames
in one pass; we use it to turn phone videos into point clouds.

---

## 3. Data model

### 3.1 The object relation
Ingesting a scan $S$ produces a relation

$$\text{Obj}(\underline{oid},\ \text{points},\ \text{bbox}_{\min},\ \text{bbox}_{\max},\ \text{centroid},\ \text{conf},\ \text{token},\ \text{pooled},\ \text{provenance},\ \text{room})$$

| attribute | type | meaning |
|---|---|---|
| `points` | sorted voxel ids | the object's mask (voxel level; a voxel→point inverse map recovers points) |
| `bbox_min/max`, `centroid` | 3 floats each | geometry for spatial predicates |
| `conf` | float | SNAP's predicted mask IoU — how much to trust the object |
| `token` | 512-d | SNAP's text token in CLIP space (what the segmenter thinks the object is) |
| `pooled` | 512/768-d | mask-pooled dense 3D CLIP feature (what the scene features say), may be empty |
| `provenance` | string | `auto` (ingest) or `query-decoder:<text>` / `query-snap:<text>` (created by cracking) |
| `room` | int | scene id when several scans share one relation (building) |

A side table `cracked: text → oid | ⊥` records which concepts have already triggered refinement (a negative cache).

**Why objects and not points?** Open-vocabulary point features answer "where does chair-ness appear"; users ask "which
objects are chairs". Object-level storage gives (i) instance answers, (ii) far less data — the relation is 1.98 MB per
S3DIS room vs. 13.3 MB of raw XYZ [T11] — and (iii) better semantics, because evidence pooled over a whole object is less
noisy than per-point evidence (§4, [T1/T2]).

### 3.2 Populating the relation (ingest)
1. **Encode** the scan once with SNAP's backbone (2.3 s per S3DIS room [T11]) and, if available, compute OpenScene
   features per voxel.
2. **Propose objects** automatically: prompt SNAP at farthest-point-sampled locations (400 prompts) or with a
   coarse-to-fine grid that adds finer prompts only where nothing is covered yet; keep masks with predicted IoU > 0.3,
   stability > 0.5 and ≥ 30 voxels; remove duplicates by mask NMS. Coarse-to-fine reaches 69.7% recall@0.5 [T10].
3. **Materialise** one tuple per proposal: mask → points, bbox, centroid; SNAP token; and pooled dense evidence
   $\text{pooled} = \text{normalize}\big(\sum_i w_i f_i\big)$ with soft weights $w_i = \mathbb{1}[z_i>0]\,\sigma(z_i)$, so
   confident interior voxels count more than boundary voxels (6.6 s per room [T11]).

### 3.3 Building scale
A building is the union of its rooms' relations, each tuple tagged with `room`. Spatial predicates never cross rooms,
which we exploit for a partitioned join (§5.3). S3DIS Area 5 becomes one relation of 68 rooms and 5,149 objects, stored
in 171 MB vs. 945 MB of raw XYZ, loaded in 0.29 s [T16].

---

## 4. Semantics of open-vocabulary predicates

### 4.1 What should `IS('chair')` mean?
We want a number in $[0,1]$ per object that behaves like a probability, so that it can be thresholded, ranked and
combined with other predicates. A raw cosine similarity is not that: it has no calibration and no notion of
alternatives. We therefore define it **relative to a background vocabulary** $B$ = {wall, floor, ceiling, object}: the
query word competes against generic alternatives, and the probability is a softmax over this small open world.

### 4.2 Two experts
For object $o$ and candidate labels $L = [c, b_1, \dots]$:
* **Dense expert** (scene features): $p_{\text{dense}}(\ell \mid o) = \text{softmax}_\ell\big(\langle \text{pooled}_o,\ \text{CLIP}(\ell)\rangle / \tau\big)$
  with the template "a {} in a scene" and $\tau = 0.01$.
* **Segmenter expert** (SNAP's token): $p_{\text{snap}}(\ell \mid o) = \text{softmax}_\ell\big(\langle \text{token}_o,\ \text{CLIP}(\ell)\rangle / \tau\big)$
  with the template "segment {}" — but see the relay below.

They are fused as a **product of experts** (a weighted geometric mean, renormalised):

$$p(\ell \mid o) \propto p_{\text{dense}}(\ell \mid o)^{\alpha}\; p_{\text{snap}}(\ell \mid o)^{1-\alpha},\qquad \alpha = 0.7$$

and $P(\text{IS}(c) \mid o) = p(c \mid o)$. A product of experts is high only when *both* experts agree, which suppresses
the typical failure of each: the dense expert alone bleeds across neighbouring objects; the segmenter alone is
confidently wrong on words it never saw.

### 4.3 Vocabulary relay
SNAP's text token was trained against a fixed set of class names. For a word outside that set (e.g. *whiteboard*), the
direct similarity is unreliable. The relay routes the query through the training vocabulary $A$ (anchors):

$$p_{\text{relay}}(\ell \mid o) = \text{softmax}_\ell\Big(\tfrac{1}{\tau}\, \textstyle\sum_{a\in A} \text{softmax}_a\big(\langle \text{token}_o, \text{CLIP}(a)\rangle/\tau_r\big)\, \langle \text{CLIP}(a), \text{CLIP}(\ell)\rangle\Big)$$

i.e. first ask "which *known* class is this object?" (a posterior over anchors, $\tau_r = 0.05$), then "how similar is
that known class to the query word?". The final segmenter expert is
$p_{\text{snap}} = p_{\text{direct}}^{1-\rho}\, p_{\text{relay}}^{\rho}$ with $\rho = 0.8$. Relay adds +2.0 mIoU on ScanNet
for the full system (56.2 → 58.2) and helps unseen Matterport classes [T1/T2].

### 4.4 From objects to points (semantic segmentation)
To label every point with one of $C$ classes, each voxel takes a vote of the objects covering it, weighted by soft mask
and object confidence, $w_{o,i} = \mathbb{1}[z_{o,i}>0]\,\sigma(z_{o,i})\,\text{conf}_o$, and blends it with the point-wise
dense prediction: $p_i = \beta\, p^{\text{dense}}_i + (1-\beta)\, \frac{\sum_o w_{o,i}\, p(\cdot\mid o)}{\sum_o w_{o,i}}$ with
$\beta = 0.3$ (uncovered voxels use the dense prediction only).

### 4.5 Composite predicates
Predicates on different objects or different concepts are treated as independent (tuple independence, as in
probabilistic databases):
* $P(\varphi \wedge \psi) = P(\varphi)\,P(\psi)$, $P(\varphi \vee \psi) = 1 - (1-P(\varphi))(1-P(\psi))$, $P(\neg\varphi) = 1 - P(\varphi)$.
* Spatial predicates are *existential*: $P(\text{NEAR}(\varphi, d) \mid o) = \max_{o' \ne o,\ \text{gap}(o,o') \le d} P(\varphi \mid o')$
  — "there is some object near $o$ that satisfies $\varphi$", scored by the most probable witness. `ABOVE`, `BELOW`, `ON`
  use footprint overlap in $xy$ and vertical ordering; the gap is the distance between bounding boxes.

Independence is an assumption (a chair and its seat are not independent); we state it, as probabilistic databases do.

---

## 5. OVQL: query language and processing

### 5.1 Syntax
```
SELECT [WHERE <expr>] [LIMIT <k>] [THRESHOLD <p>]
expr   := term (OR term)*
term   := factor (AND factor)*
factor := NOT factor | '(' expr ')' | pred
pred   := IS('<text>') | NEAR(<expr>, <meters>) | ABOVE(<expr>) | BELOW(<expr>) | ON(<expr>) | CONTAINS(x, y, z)
```
Examples:
```sql
SELECT WHERE IS('chair') LIMIT 10
SELECT WHERE IS('chair') AND NEAR(IS('whiteboard'), 0.5)
SELECT WHERE IS('monitor') AND NOT ON(IS('desk'))
SELECT WHERE IS('couch') OR IS('sofa') THRESHOLD 0.3
```
The result is a ranked list of `(oid, probability)`.

### 5.2 Two-phase plan
**Phase 1 (refinement planning).** Walk the parse tree; for each `IS(c)` not yet in the `cracked` cache, compute
$\max_o P(\text{IS}(c)\mid o)$. If no stored object explains $c$ well (max < θ = 0.3), call a *refinement operator* that
creates the missing object (§5.4) and record the outcome (even ⊥) in `cracked`.
**Phase 2 (scoring).** Evaluate the tree bottom-up over the — possibly extended — relation with the semantics of §4.5,
rank, apply THRESHOLD and LIMIT.
Splitting the phases matters: refining *during* scoring changes the table size mid-expression, which corrupts the
vectorised evaluation.

### 5.3 Room-partitioned spatial join
A naive spatial predicate builds an $M \times M$ relation between all objects. In a building, objects in different
rooms are never near each other. We sort tuples by room and lay them out in a padded tensor of shape $(R, P)$ ($R$
rooms, $P$ = max objects per room), compute the relation as one batched $(R, P, P)$ tensor operation, mask padding and the
diagonal, and scatter results back. This is exact — identical to evaluating each room separately on all 5,149 objects —
and removes the Python loop over rooms: NEAR over 68 rooms takes 3.4 ms [T16].

### 5.4 Semantic cracking (query-driven refinement)
*Database cracking* reorganises an index as a side effect of queries, so the work is paid only for what is actually
asked. We apply the idea to semantics. Automatic proposals miss some objects (recall ≈ 70% [T10]); if a query asks for
one of them, phase 1 detects that nothing explains the word and invokes a refinement operator:
* **training-free**: *language as the click oracle* — compute per-voxel relevance of the word, click its densest
  high-relevance voxel, and keep clicking where relevance is high but SNAP's mask is missing (up to 3 clicks), choosing
  the click set whose mask best agrees with the relevance region;
* **trained**: the v4 text decoder (§6.3) segments "the {word}" directly.
The new object is appended to the relation with provenance `query-*:<word>`, unless it overlaps an existing object by
IoU > 0.5 (dedup), and `cracked[word]` remembers the outcome. The next query with that word (or one whose objects now
exist) runs at stored-relation speed.

Trade-off [T12/T14, T12c]: with the training-free refiner, repeated queries become cheap (1.7 ms second pass vs. 37 ms
refining every time) but AP50 drops 38.9 → 35.2 because the refined masks are imperfect. With the trained refiner plus
dedup, the second pass costs 3.7 ms vs. 26.8 ms and AP50 is 38.7 vs. 38.9 — cracking becomes nearly free in accuracy.
Outdoors it even improves accuracy: +1.2 AP on nuScenes, +2.6 on SemanticKITTI [T4c, T4d].

---

## 6. Operators: better segmentation, naming and prompting

### 6.1 Click-to-name: 3D-propose / VLM-choose
When a user clicks an object, we want its name in free vocabulary. Two sources exist: the 3D evidence of §4 (knows the
geometry, limited recognition) and a VLM looking at a rendering (strong recognition, but point-cloud renders are hard
to read). We combine them:
1. **3D proposes**: score a large prior vocabulary (SNAP's training names ∪ 1,198 LVIS names, 1,339 total) with
   $p(\ell \mid o)$ and keep the top-5.
2. **VLM chooses**: render the object highlighted in red from several viewpoints and ask Qwen2.5-VL to answer with a
   letter A–E; we read the letter log-probabilities instead of free text, so the answer is always in the list.
3. **Cycle consistency**: each candidate name is used as a text query; names whose text-driven mask agrees with the
   clicked mask are preferred.
On Matterport-160 (212 clicks), this reaches 40.1% top-1 vs. 30.7% for SNAP's text head and 18.9% for the VLM naming
freely; on unseen classes 17.2% vs. 12.2% and 6.5% [T6/T7].

### 6.2 Text → mask
"Segment the X" needs a mask from text. The training-free route is language-as-click-oracle (§5.4): 11.8 mIoU on
ScanNet200 [T8].

### 6.3 A trained text decoder (v4)
SNAP's decoder is good at clicks; teaching it text by adding tokens to the same decoder failed (v3: 1.1 mIoU [T8]) and
fine-tuning it hurt clicks (v1/v2 [T5]). v4 therefore keeps the click path frozen and adds a **separate trainable copy**
of SNAP's two-way decoder and mask head for text prompts: the prompt token is the CLIP text embedding, and per-voxel 3D
CLIP features enter only the attention keys. Trained on ScanNet200 head + common classes (tail held out): 18.2 mIoU
(head 32.8 / common 13.8 / tail 2.7), and 27.2 vs. 20.0 on S3DIS, which it never saw [T8, T8b].

### 6.4 Negative clicks for a positive-only segmenter
**Why it matters.** In interactive segmentation the user (or a simulated user) clicks where the current mask is wrong.
If the mask *misses* part of the object, a positive click fixes it. If the mask *spills* onto a neighbour, you need a
**negative** click. SNAP has no such input, so once it spills it plateaus: on Replica, SNAP goes from 59.5 IoU after 5
clicks to only 60.1 after 10, because every corrective negative click is wasted [T5f].

**(a) Test-time token tuning (training-free).** SNAP's mask is $z = E\,m$. Decode the positive clicks once with frozen
SNAP, keep the refined voxel embeddings $E$ fixed, and optimise only the mask vector $m$ (and a scalar bias $b$) so that
all clicks are honoured:

$$\min_{m,b}\ \frac{1}{|K|}\sum_{k\in K} \text{softplus}\big(-y_k\,(E_k^\top m + b - 2y_k)\big)\;+\;\lambda\,\frac{\|m - m_0\|^2}{\|m_0\|^2}$$

with $y_k = +1$ for positive and $-1$ for negative clicks (a margin of 2), $m_0$ SNAP's own token and $\lambda = 10$; 30
Adam steps. Only a $D$-dimensional vector moves over cached embeddings, so it costs ~43 ms per click on a 267k-voxel room
[T5h]. The anchor keeps the mask close to SNAP's where clicks give no evidence. All hyper-parameters were chosen on
ScanNet val only [T5g].

**(b) Corrective decoder (trained, v5).** A trainable copy of SNAP's decoder that knows *click types*: learned
positive/negative embeddings are added to the prompt tokens, and the previous mask enters as a dense prompt
($\text{pe}_i + \sigma(z^{\text{prev}}_i)\, e_{\text{prev}}$). Routing guarantees parity: if all clicks are positive, the
exact frozen SNAP decoder is used; the corrective decoder takes over only once a negative click exists. Trained with
simulated corrective clicking (each new click in the larger error region) on ScanNet, nuScenes-train and
SemanticKITTI-train in equal proportion. (a) and (b) compose: token tuning on the corrective decoder's $(E, m)$.

**(c) Backbone adapters (v6).** The only change to SNAP's *features*: rank-16 LoRA adapters on the attention projections
of every PTv3 layer, initialised so that the network is exactly SNAP at step 0, trained jointly with the corrective
decoder with a feature-distillation term $\|f - f_0\|^2/\|f_0\|^2$ that keeps features close to the original.

**(d) Click-jitter averaging.** The first click is the hardest. We decode the click set 8 more times with each click
moved to a random voxel within 10 cm and average the logits — a cheap way to reduce sensitivity to the exact click
position.

---

## 7. In-the-wild: from a phone video to a queryable scene

### 7.1 Video → point cloud
1. **Frames**: sample 3× the target number of frames and keep the sharpest of each consecutive triple (variance of the
   Laplacian), 48 frames for an 83 s video.
2. **Geometry**: VGGT predicts per-frame depth, confidence and camera poses in one pass; depth is unprojected to 3D.
3. **Metric scale**: VGGT's scale is arbitrary, but SNAP's 2 cm voxels need metres. On 8 frames we compare VGGT depth
   with Depth-Anything-V2 (metric, indoor) on confident pixels and take the median ratio (per-frame spread 13.9% on our
   video).
4. **Cleanup**: keep the top 35% most confident points within 2.5 m, voxel-downsample to 5 mm.
5. **Gravity**: RANSAC-fit the largest plane whose normal is within ~35° of the first camera's up axis (a desk or the
   floor, not a wall) and rotate it to $+z$, the orientation SNAP's indoor training data has.
Cameras, depth and frame paths are saved next to the cloud.

### 7.2 The dense features can silently fail — and how we detect it
On our phone reconstruction, OpenScene's 3D CLIP model labelled 80% of the object voxels "stair rail": the streaky
edges of a monocular reconstruction look like thin rails to a model trained on clean scans. Because the dense expert has
weight $\alpha = 0.7$ in the product of experts, this single failure turned every name into "stair rail". Denoising and
other OpenScene checkpoints did not fix it — they predicted "floor"/"wall" for everything instead.

**Collapse guard.** At ingest, label every voxel with the dense expert over SNAP's training vocabulary and measure the
share of the most frequent non-structural label (anything not wall/floor/ceiling-like) among object voxels. On 38
benchmark scans (ScanNet, S3DIS, Replica, Matterport) this share is 0.14–0.66; on the phone video 0.80 [T18]. Above 0.75
we drop the dense features for that scene and compute text relevance from the segmenter's own proposals: each voxel
gets the soft-mask-weighted vote of the proposals' $p_{\text{snap}}$ — the same object-anchored idea, without the dense
expert.

### 7.3 Photo-grounded naming
Point-cloud renders of a noisy reconstruction are hard to read even for a 7B VLM (it answered "gun", "glove", "cloud").
But for a video we have the real photos. For a selected 3D mask we project its points into every frame using the VGGT
cameras, keep points whose projected depth agrees with VGGT's depth map (occlusion test), pick the two frames where the
largest fraction of the object is visible near the image centre, and crop the original high-resolution frame around the
object with a red box. The VLM names the **photo**; we use its ranked answer directly, because here the visual evidence
is far stronger than the weakened 3D evidence. Result on the lab desk: chair, monitor, bottle (alternatives: thermos,
flask), printer, desk, the side of a PC tower ("vent"); errors: "browser" for a laptop screen (it reads the screen
content), "screwdriver" for a PC-case box.

---

## 8. Experiments

### 8.1 Setup
| dataset | evaluation split | tasks |
|---|---|---|
| ScanNet (20 classes) | val, 312 scenes | semantic segmentation |
| ScanNet200 | val, 156 / 104 scenes | queries; text → mask (head / common / tail) |
| Matterport3D-160 | test, 406 regions | semantic segmentation (63 classes unseen by SNAP); naming (212 clicks) |
| S3DIS | Area 5, 68 rooms | queries, instance AP, interactive (rooms and ±1.5 m object crops), text → mask, systems, building |
| Replica | 8 scenes | instance AP, interactive |
| nuScenes | official val, 6,019 keyframes | LiDAR semantic seg, instance AP, interactive |
| SemanticKITTI | sequence 08, 4,071 scans | LiDAR semantic seg, instance AP, interactive |
| UnderOneFacade | test, 17 façades (UK/DE/SG, TLS + MLS, 254M points) | zero-shot façade semantic seg (15/5 classes), interactive on façade parts |

*Metrics*: mIoU / mAcc; instance AP over IoU 0.50:0.95 (also AP50, AP25); IoU after $k$ clicks (IoU@k) and number of
clicks to reach 80/85/90% IoU (NoC); naming top-1 accuracy (free names mapped to labels by CLIP text similarity); latency
and storage. *Fairness*: every method is run under one protocol by us; published numbers are shown separately with their
protocol differences (§8.9). *Noise*: identical interactive runs differ by ≈ ±0.5 IoU; naming by ≈ ±3 points.

### 8.2 Open-vocabulary semantic segmentation [T1/T2, T15, T2b, T2c]
| method | ScanNet | Matterport-160 (unseen) | nuScenes | SemanticKITTI |
|---|---|---|---|---|
| SNAP text token | 41.5 | 6.9 (1.9) | 23.8 | **32.8** |
| OpenScene 3D (point-wise) | 52.0 | 9.5 (5.5) | 26.9 | 20.8 |
| mask-anchored pooling | 57.2 | 10.0 (5.5) | 27.2 | 21.5 |
| **OpenSNAP (full)** | **58.2** | **11.6 (5.7)** | **32.0** | 31.5 |

*Reading the table.* Pooling the same OpenScene features over objects instead of points is worth +5.2 on ScanNet: the
object is the right unit of evidence. Fusing with SNAP's token and relaying unseen words adds another +1.0. The one
loss, SemanticKITTI, is instructive: no OpenScene model exists for KITTI, so we used the nuScenes model cross-dataset; it
fails on ground classes (terrain 0.0 IoU), and fusing it costs mIoU even though OpenSNAP still wins on objects
(other-vehicle 2.9 → 11.9) and on mAcc (45.8 vs. 43.7). A mismatched dense model is also why §7.2's guard exists. With
a Matterport-trained dense model on ScanNet, OpenScene drops to 37.3 while OpenSNAP keeps 47.6 [T15] — the segmenter
token makes the system robust to dense-model mismatch.

**Out of domain: urban façades** [T2d]. UnderOneFacade (Wang et al., ECCV 2026) labels terrestrial and mobile laser
scans of 366 façades in the UK, Germany and Singapore with 15 architectural classes (wall, window, door, balcony,
molding, …) and 5 coarse groups. Neither SNAP nor any OpenScene model saw façade data; we chose the configuration
(Outdoor segmenter, nuScenes dense model) on 22 validation façades and ran the 17-façade test split once, scoring every
raw point (254M).

| method | LoFG3 mIoU | LoFG3 OA | LoFG2 mIoU | LoFG2 OA |
|---|---|---|---|---|
| OpenScene 3D (point-wise) | 8.4 | 38.5 | 29.8 | 55.5 |
| mask-anchored pooling | **9.0** | **42.1** | **31.8** | **59.4** |
| OpenSNAP (full) | 8.1 | 39.8 | 28.1 | 56.5 |
| *supervised, trained on the other continent (published)* | 6.6–23.6 | 32.1–51.9 | 25.7–40.8 | 48.6–65.8 |
| *supervised, trained on all façades (published, best)* | 33.4 | 62.7 | – | – |

Zero-shot, the system lands at the low end of what façade-trained networks reach when moved to another continent, but
far below in-domain training. The failure is structural, not lexical: windows, mouldings and balconies are parts lying
on one wall surface, and SNAP's object-level masks do not separate them (wall 44 IoU; window, balcony, molding 0). An
oracle that gives every proposal its true majority class reaches only 18–21 mIoU on three validation façades
(`results/aux/facade_oracle_val_outdoor.json`), so better
naming cannot close the gap; part-level proposals would. Here the segmenter token is the weakest expert (4.6), the
relay helps it (+0.7), and fusing it with the dense branch costs 0.9 against pooling alone.

*Against other zero-shot ideas* [T2e]: pooling also beats a 3D shape-text model labelling the very same masks (OpenShape:
4.2 / 15.4 LoFG3 / LoFG2 mIoU) and render-then-lift pipelines that render the scan from virtual cameras and lift 2D
open-vocabulary masks (Grounded-SAM 6.1 / 16.6; SAM + SigLIP 5.1 / 15.2), although those see real colour and we use XYZ
only. *Search* [T3f]: concept and relational queries for façade parts fail for every zero-shot system — OpenSNAP 0.3
AP25, the best alternative (Grounded-SAM) 3.3, all of it on photo-like UK renders and 0 on German scans. Record:
`results/FACADE.md`.

### 8.3 Query accuracy and instance segmentation [T3, T4, T4b–d]
| dataset (classes) | SNAP token | **OpenSNAP** | + cracking | published references |
|---|---|---|---|---|
| S3DIS (13, c2f proposals) | 10.5 | 15.4 | 15.5 | SNAP@auto 16.1 (own protocol) |
| Replica (all objects, c2f) | 14.8 | **19.9** | 18.9 | SNAP@auto 10.1 (OpenMask3D's closed-set protocol: see §8.10) |
| nuScenes (10 things) | 7.9 | 12.7 | **13.9** | — |
| SemanticKITTI (8 things) | 13.9 | 18.0 | **20.6** | — |

Relational queries improve as well: `IS(a) AND NEAR(IS(b), 0.5)` AP on S3DIS 14.5 vs. 12.2, nuScenes 17.0 vs. 13.3. The
exception is the ScanNet200 *tail*, where SNAP's text head — trained on all 200 classes — is in-domain and wins (4.4 vs.
2.5 AP) [T3].

### 8.4 Systems: latency, storage, cracking, building scale [T11, T12/T14, T12b, T12c, T16]
| measurement | value |
|---|---|
| ingest per S3DIS room | 2.3 s encode + 6.6 s materialise (once) |
| query on the stored relation | **0.8 ms** |
| re-processing the scan per query | 8.8 s (10,936× slower) |
| relation size vs. raw XYZ | 1.98 MB vs. 13.3 MB (6.7×) |
| building: 68 rooms / 5,149 objects | catalog 171 MB vs. 945 MB raw; loads in 0.29 s |
| building query latency | IS 1.41 ms (0.87 with LIMIT 10), NEAR 3.40 ms, ABOVE 2.95 ms, OR/NOT 2.08 ms |
| partitioned join vs. per-room evaluation | identical (5,149 / 5,149) |
| cracking, trained refiner | 2nd pass 3.7 ms vs. 26.8 ms; AP50 38.7 vs. 38.9 |
| cracking, training-free refiner | 2nd pass 1.7 ms vs. 37.1 ms; AP50 35.2 vs. 38.9 |

The calibration sweep [T13] shows how the refinement trigger θ and the confidence weight γ trade the number of
refinements against AP, and that dedup (merging refined masks with IoU > 0.3–0.7 into existing objects) recovers AP50
lost to duplicates.

### 8.5 Naming [T6/T7]
| method (Qwen2.5-VL-7B) | all classes | unseen classes |
|---|---|---|
| SNAP text head | 30.7 | 12.2 |
| VLM free naming | 18.9 | 6.5 |
| OpenSNAP 3D evidence | 35.4 | 16.4 |
| Render-Ask-Verify | 38.2 | 16.4 |
| **3D-propose / VLM-choose** | **40.1** | **17.2** |
| closed-vocabulary OpenSNAP (reference) | 39.2 | 18.3 |

Open-vocabulary naming reaches the closed-vocabulary level. The VLM alone is worst: point-cloud renders are hard to
recognise without the 3D side narrowing the options.

### 8.6 Text → mask [T8, T8b]
v4 text decoder: ScanNet200 18.2 mIoU (head 32.8 / common 13.8 / tail 2.7, tail held out) vs. 11.8 training-free; on
S3DIS (never seen) 27.2 vs. 20.0 — chair 79.0, table 51.4; flat classes (board, beam, column) stay at 0 with XYZ-only
input.

### 8.7 Interactive segmentation [T5, T5c–T5k]
**Parity first.** Our click path reproduces SNAP: S3DIS rooms 53.3 / 69.1 / 74.0 (published 53.6 / 71.1 / 77.6),
object crops 56.1 / 72.0 / 76.8 (published 56.6 / 73.8 / 80.9). Positive-only clicks always go through SNAP's exact
decoder.

**Negative clicks** (corrective protocol, identical click sequences; IoU@5 / IoU@10):
| dataset | SNAP (negatives ignored) | token tuning | multi-domain decoder + tuning |
|---|---|---|---|
| S3DIS crops | 73.1 / 74.6 | **81.8 / 88.8** | 81.3 / 87.8 |
| S3DIS rooms | 69.0 / 70.5 | **78.5 / 84.7** | 77.5 / 83.5 |
| Replica | 59.5 / 60.1 | **74.8 / 81.9** | 72.6 / 80.3 |
| nuScenes | 63.8 / 65.4 | 72.3 / 78.4 | **73.4 / 79.2** |
| SemanticKITTI | 67.6 / 70.4 | 71.0 / 76.5 | **71.9 / 76.9** |
| UnderOneFacade parts (zero-shot) | 10.1 / 10.1 | 26.2 / 46.2 | 37.4 / 49.4 |

Token tuning is the most robust choice indoors and on held-out data; the trained corrective decoder adds most on the
outdoor domains it was trained on. A decoder trained on ScanNet only overfits indoors (e.g. Replica IoU@10 75.8 vs.
81.9 for token tuning), which is why the shipped decoder is multi-domain. Clicks to 90% IoU drop everywhere (e.g.
nuScenes 6.18 → 4.78).

**Façade parts** (UnderOneFacade test, 1,603 window/door/molding/… instances = connected components of the part labels,
3 m crops). A click on a window returns the wall: SNAP reaches 11.0 IoU@5 and positive clicks barely help. Negative
clicks are what such a part needs, and they more than triple IoU@5 (37.4) and quadruple IoU@10 (49.4); per class, doors
29 → 65, columns 30 → 59, windows 5 → 25 IoU@5. Clicks to 90% stay high (9.5 of 10). Dedicated click segmenters
are well ahead under identical crops and clicks [T5l]: InterObject3D 56.8, AGILE3D 57.6, Point-SAM 64.1 IoU@5, and with
colour zeroed still 54.1 (AGILE3D) and 59.9 (Point-SAM). Point-SAM, trained on object parts, leads most on the
colourless German scans (48.7 vs 17.2). Façade parts are where object-level promptable segmentation stops transferring.

**Backbone adapters** (positive clicks only, IoU@1) [T5i]: nuScenes 53.5 → **57.0** (+3.5, and +3.3–4.0 at every click
count), SemanticKITTI 56.7 → 57.5, S3DIS rooms 53.3 → 53.3 (later clicks +0.5), S3DIS crops 56.1 → 54.2 (later clicks
+0.8–1.2). Adapters help in the domains whose training split they saw and cost a little on held-out indoor first clicks;
since SNAP has an explicit domain input, they should be enabled for outdoor scans only.

**Click-jitter averaging** [T5k]: +1.1 IoU@1 on ScanNet (selection set), +1.5 on held-out S3DIS crops; no effect on
nuScenes (10 cm is small relative to sparse LiDAR). Best S3DIS-crop result: 57.6 / 74.4 / 82.2 / 89.0.

**Against published interactive methods** (S3DIS crops, IoU@1/3/5): AGILE3D 58.7 / 77.4 / 83.6, Point-SAM 45.9 / 77.6 /
84.6, OpenSNAP 57.6 / 74.4 / 82.2. We close most of SNAP's gap and come within 1.1–1.4 points of AGILE3D at 1 and 5
clicks, but are still 3.0 behind at 3 clicks. OpenSNAP remains a single promptable model for indoor and outdoor data.

**Latency per click** [T5h]: SNAP 14 ms, corrective decoder 16 ms, token tuning 43 ms (median; 267k-voxel room).

### 8.8 Phone video [T18, `assets/demo_lab_desk/`]
An 83 s phone video of a lab desk becomes a 1.09 M-point metric scene. The collapse guard fires (0.80 vs. ≤ 0.66 on all
benchmark scans), photo-grounded naming names clean objects correctly, and the viewer supports click-to-name,
segment-everything and open-vocabulary labelling. Text queries are weaker on this scan (office chair found; keyboard,
printer and water bottle missed) because text relevance must come from the segmenter alone. This is a qualitative,
single-video result.

### 8.9 Protocol notes for comparisons with published numbers
* SNAP's published S3DIS click numbers exceed our reproduction at 10 clicks (83.2 vs. 77.4): click placement differs;
  every comparison we make uses one protocol.
* SNAP@auto's 16.1 S3DIS AP uses the authors' evaluation; under ours SNAP's text token gives 10.5 and OpenSNAP 15.4.
* OpenMask3D's 13.1 Replica AP is a closed-set 51-class protocol and is not comparable with the query workload above; the
  closed-set comparison is §8.10 (OpenSNAP 12.9 XYZ-only, 24.7 with the image column).
* OpenScene's published nuScenes 36.7 uses its own processed sweeps and script; ours, on single keyframes, is 26.9
  (31.1 on OpenScene's processed sweeps).

### 8.10 Comparative study under the protocols of Search3D, OVI-MAP and HouseLayout3D [C1–C13]
The full study is in `results/COMPARISON.md`. In brief:
* **OVI-MAP protocol** (same 8 Replica and 18 ScanNet scenes, same 51 / ScanNet200 classes, same evaluator):
  * Every setting is chosen on ScanNet *train* development scenes and applied once to the test scenes.
  * Replica with an image column (SigLIP so400m crops): 31.6 mIoU / 44.1 mAcc / 34.8 AP25 / 30.6 AP50 / 24.7 AP, against OVI-MAP's 26.5 / 32.2 / 34.5 / 21.2 / 8.5.
  * ScanNet-18 with an image column: 28.8 / 42.8 / 38.1 / 30.1 / 20.0 (mIoU / mAcc / AP25 / AP50 / AP), against 17.5 / 27.6 / 23.4 / 15.7 / 7.2.
  * From XYZ only: Replica AP 12.9, ScanNet AP 9.8 vs 7.2.
* **Search3D / OpenMask3D on ScanNet200 (312 val scenes):** with the image column, OpenSNAP reaches 15.3 AP / 24.1 AP50 /
  29.9 AP25. That is level with OpenMask3D on AP (15.4) and ahead on AP50 / AP25 (19.9 / 23.1). It is still behind
  Search3D-SigLIP (23.0), Open3DIS (23.7) and RAZER (24.7). From XYZ only, OpenSNAP gets 6.1 AP. Semantic segmentation
  on the same scenes reaches 22.7 mIoU vs OVO-mapping's 17.4.
* **OpenScan attributes:** 2.7 vs 9.9 mean AP.
* **HouseLayout3D buildings:** strict room partitioning loses 15–32% of NEAR answers, room + halo h ≥ d is exact, and about
  half of the global ABOVE / BELOW answers have their only witness on another floor, so vertical predicates are
  level-scoped.
* **Relational queries:** OVQL beats phrase-embedding retrieval (S3DIS NEAR AP50 21.7 vs 14.3).
* **Calibration:** ECE is 0.015–0.022 with the 234-word background (0.004–0.010 after Platt scaling fitted on ScanNet
  train).
* **Long cracking workloads** need admission control: without it probe AP50 drops from 21.7 to 8.6 after 400 queries; with
  verify-before-admit it stays at 21.7.
* **Campus scale:** a cost-based pushdown join answers NEAR over 1.22 M objects in 2.6 ms.

---

## 9. What we learned (design lessons)
1. **The object is the right unit** for open-vocabulary evidence, for storage and for answers.
2. **Do not fine-tune the segmenter on one dataset**: it loses the cross-dataset generalisation that makes it valuable
   (v1/v2). Add capacity *beside* it (v4 text decoder, v5 corrective decoder) or *gently inside* it with
   distillation (v6 adapters), and keep the original path exact.
3. **Negative clicks were the missing input**, and they can be added at test time with a 30-step optimisation of one
   vector; training helps most where training data exist.
4. **Dense features are a liability out of distribution** — detect collapse instead of trusting them blindly.
5. **When real images exist, name from images**: photo-grounded naming beats any point-cloud rendering.
6. **Cracking is only as good as its refiner**: a trained refiner makes lazy refinement nearly free in accuracy.

## 10. Limitations
* Positive-only clicking is SNAP's; our segmentation gains come from negative clicks, test-time methods and adapters.
* Flat objects on walls and floors (boards, rugs, paintings) are hard with XYZ-only input. The same holds for façade
  parts: zero-shot façade segmentation reaches 9.0 LoFG3 mIoU against 33.4 for supervised training (T2d), because the
  object-level proposals do not split a wall into its windows and mouldings; for the same reason façade-part search
  fails (0.3 AP25, T3f) and part-trained click segmenters are 17–27 IoU@5 ahead (T5l).
* Truly novel concepts remain hard; most unseen-class gains come from near-synonyms of seen classes.
* Tuple independence is an assumption for composite predicates; in practice max-witness, noisy-OR and clipped-count
  semantics differ by ≤ 0.5 AP50 (C9).
* Geometry-only evidence is weak on long-tail classes and attributes (ScanNet200 common/tail ≈ 2 AP, OpenScan 2.7 mean AP).
  The image column lifts ScanNet200 common/tail to 12.1 AP, still below the strongest image-based systems (21–24).
* RGB-only video ingest is limited by monocular metric scale (20–42% error on Replica).
* No scene-level collapse guard we tried separates collapse from clean scans (C11); per-object gating is future work.
* Several baselines are in-domain for SNAP (nuScenes, SemanticKITTI, ScanNet200); S3DIS, Replica and Matterport-160
  unseen classes are the fairest tests.
* The collapse-guard threshold was set from 38 benchmark scans and one phone video; at scale it fires on 4.5–5.9% of clean
  scans and misses partial collapse (C11).

## 11. Related work (pointers)
*Promptable 3D segmentation*: SNAP, AGILE3D, Point-SAM, Interactive4D, InterObject3D. *Segment Anything* (multi-mask
tokens, IoU head). *Open-vocabulary 3D*: OpenScene, OpenMask3D, OpenIns3D, SAI3D; 2D features LSeg, OpenSeg, CLIP.
*Vision–language models*: Qwen2.5-VL. *Reconstruction*: VGGT; Depth-Anything-V2. *Backbones*: Point Transformer V3,
MinkowskiEngine. *Data management*: database cracking (Idreos, Kersten, Manegold), probabilistic databases and tuple
independence (Suciu, Olteanu, Ré, Koch), spatial query processing, vector databases, multimedia retrieval.

## 12. Conclusion
OpenSNAP shows that a promptable 3D segmenter can be the ingest operator of an open-vocabulary object database: scans
are materialised once into an object relation with calibrated open-vocabulary semantics, queried in milliseconds with a
small probabilistic query language, refined lazily where the stored objects fall short, and scaled to whole buildings.
The same object-centric view improves the segmenter's own capabilities — open-vocabulary labels, names, text prompts and
negative clicks — and extends to scenes captured with nothing more than a phone.

---

### Reproducing
See [`results/RESULTS.md` §11](results/RESULTS.md) for the exact commands per table, and
[`downloads/checkpoints/README.md`] (local) for the trained decoders and adapters (SHA256SUMS provided).
