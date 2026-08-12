# HyperAlign

Hypergraph-refined multimodal alignment on top of [GRAM](https://github.com/ispamm/GRAM)
(Gramian Multimodal Representation Learning, ICLR 2025), with per-clip missing-modality
handling. Text, video, audio, subtitle and depth are aligned by the Gramian volume of their
embedding vectors; a gated hypergraph refines the non-text embeddings during training.

---

## 1. Storage layout on Leonardo

Home has a 50 GB quota, so the project is split across three locations:

| What | Where | Env var |
|---|---|---|
| Code (this repo) | `/leonardo/home/userexternal/amehrish/HyperAlign` | `HA_CODE` |
| Big folders: `pretrained_weights`, `datasets`, `data`, `workdir*` | `/leonardo_work/AIFAC_S07_041/HyperAlign` | `HA_WORK` |
| Raw dataset (videos / audios / annotations) | `/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset` | `HA_DATA` |

Model code and some configs use paths **relative to the repo root** (`./pretrained_weights/…`,
`datasets/annotations/…`); `setup_paths.sh` symlinks the work-area folders in so those resolve.
All training output goes to `$HA_WORK`, never to home.

**Always run from the home checkout.** A stale copy of the code also lives in the work area
with the old (broken) paths; submitting from there fails confusingly.

## 2. One-time setup

```bash
cd /leonardo/home/userexternal/amehrish/HyperAlign
bash setup_env.sh      # conda env in $HA_WORK/envs (caches redirected off home quota)
bash setup_paths.sh    # symlink big folders, create log dirs, sanity-check key files
```

`setup_env.sh` installs Python 3.10, CUDA torch 2.1.2 (cu121), ffmpeg and `requirements.txt`.
`setup_paths.sh` reports anything `MISSING` — fix those before submitting. The VAST foundation
checkpoint is expected at
`$HA_WORK/pretrained_weights/VAST_foundation/pretrain_vast/ckpt/model_step_204994.pt`.

Job scripts source `slurm_scripts/env.sh`, which locates conda and activates the environment.
Override with `CONDA_SH=…` or `HA_CONDA_ENV=…` (name or full prefix path) if yours differs.

---

## 3. The algorithm

**Notation.** A shard holds `B` documents (per GPU — the graph is built *before* the cross-GPU
gather, so documents on different GPUs are never neighbours). `M ⊆ {V,A,S,D}` are the non-text
modalities the task uses, `k₁ = |M|`, embeddings are `d = 512`-dimensional.

### Step 1 — Encode and project

Each modality goes through its encoder (EVA-CLIP-giant / BEATs / BERT), a pooling step and a
linear contrastive head, then L2 normalisation:

```
u_j^m = W_m · pool(Enc_m(x_j^m))          (pre-norm, kept for L_reg)
z_j^m = u_j^m / ‖u_j^m‖                   m ∈ M
c_j   = u_j^T / ‖u_j^T‖                   caption / query text
```

### Step 2 — Presence detection

A modality the loader could not read is zero-filled. Real features are unit-norm, so:

```
p_{j,m} = 1[ ‖z_j^m‖ > 0.5 ]  ∈ {0,1}
```

### Step 3 — Semantic graph over documents  *(training only, stage B)*

Built from **caption** similarity, under `torch.no_grad()` and on detached text features — the
topology is structure, not something learned, and text is never a graph vertex.

```
S = C Cᵀ                     S_ij = cos(c_i, c_j)   (C has unit rows)
S_ii = −∞                    a document cannot select itself
k_eff = min(k, max(2, ⌊B/4⌋))
N(i) = top-k_eff of row i    (rank-based)
A_ij = 1[ j ∈ N(i) ]         directed, binary
```

Three filters, all multiplicative masks, applied while the adjacency is still **binary**:

```
mutual     M ← A ⊙ Aᵀ                                    (symmetric; kills hubs)
floor      M ← M ⊙ 1[ S ≥ μ_S + σ·s_S ]                  μ_S, s_S over off-diagonal S
dropout    M ← M ⊙ R,  R = min(R′, R′ᵀ),  R′_ij ~ Bern(1−p)
```

Only then are edge **strengths** applied — doing it earlier would square them and flip
mutually-negative pairs positive:

```
W = M ⊙ max(S, 0)            w_ij = cos(c_i, c_j) on surviving edges
```

### Step 4 — Incidence matrices

Vertices are (document, modality) pairs, indexed doc-major as `j·k₁ + m`.

```
H_doc[(j,m), e] = p_{j,m} · 1[e = j]                     block-diagonal
H_sem[(j,m), e] = p_{j,m} · (W + I)_{j,e}                cross-document
H = [ H_doc | H_sem ]                                    (B·k₁) × (2B)
```

The `p_{j,m}` factor appears in **both**: an absent modality has degree 0, so it neither sends
nor receives messages and stays exactly zero.

### Step 5 — Degree normalisation

```
D_E = diag(1ᵀH) ∨ 1        edge degree   (how many vertices an edge holds)
D_V = diag(H1)  ∨ 1        vertex degree (how many edges a vertex is in)
H_e = H D_E⁻¹              column-normalised  → V→E is an average
H_v = D_V⁻¹ H              row-normalised     → E→V is an average
```

Averaging rather than summing keeps a large hyperedge from shouting louder than a small one.
The `∨ 1` (clamp) makes a degree-0 row divide by 1 instead of 0.

### Step 6 — Gated hypergraph message passing  (L ≤ 2 layers)

```
F⁰ = [ z_j^m ]                                       (B·k₁) × d
Fᴱ_ℓ  = GELU( H_eᵀ Fℓ⁻¹ W_V^ℓ )                      vertices → edges
Fᴺ_ℓ  = H_v Fᴱ_ℓ W_E^ℓ        (GELU if ℓ < L)        edges → vertices
Fℓ    = Fℓ⁻¹ + tanh(g_ℓ) · Fᴺ_ℓ                      gated residual
```

`W_V, W_E` are bias-free, so a zero vertex contributes exactly zero. Vertices inside one
hyperedge are **not** wired pairwise — they all read the same edge summary, which is what makes
this a hypergraph rather than a graph. Run in fp32 (autocast disabled).

### Step 7 — Refined embeddings and document embedding

```
ẑ_j^m = Fᴸ[j,m] / ‖Fᴸ[j,m]‖
h_j   = W_h ( Σ_m p_{j,m} Fᴸ[j,m] ) / max(Σ_m p_{j,m}, 1)        mean over PRESENT only
ĥ_j   = h_j / ‖h_j‖
```

`c_j` is left untouched — refining the text anchor with its own document would leak.

### Step 8 — Masked Gramian volume

For query `i` and gallery document `j` over modalities `m₁…m_L`, build the Gram matrix of
`[ c_i, ẑ_j^{m₁}, …, ẑ_j^{m_L} ]` and let `p̃_j = [1, p_{j,m₁}, …, p_{j,m_L}]` (text always
present):

```
G ← G ⊙ (p̃ p̃ᵀ) + diag(1 − p̃)          zero the missing row/col, put 1 on its diagonal
vol(i,j) = √( |det G| + ε )
```

A missing modality becomes an **orthonormal phantom axis** contributing a determinant factor of
exactly 1, so `det G` reduces to the determinant of the present sub-Gram — the clip is scored at
its own arity. With all modalities present this is byte-for-byte the unmasked volume; without the
masking a zero row makes `G` singular and `vol = 0` for **every** query, so incomplete clips all
tie and become unrankable.

For two vectors the volume is `√(1 − cos²) = |sin θ|`, i.e. the parallelogram area — the same
geometry at arity 2.

### Step 9 — Losses

With learnable temperature `τ`, in-batch targets, label smoothing 0.1:

```
L_area = ½[ CE(−vol/τ, y) + CE(−volᵀ/τ, y) ]              GRAM's volume contrastive loss
L_doc  = ½[ CE(−vol₂(c, ĥ)/τ, y) + CE(−vol₂(ĥ, c)/τ, y) ]  arity-2, robust by construction
L_reg  = Σ_m relu(1 − std(u^m)) + relu(1 − std(h))         VICReg-style variance hinge
L_itm  = image-text matching cross-entropy

L = L_area + w_doc · L_doc + w_reg · L_reg + L_itm
```

`L_area` is computed on the **refined** `ẑ`, so the graph is on the main path, not a side branch.
`L_doc` folds a document into a single vector, so its 2×2 Gram can never collapse regardless of
how many modalities are missing.

### Step 10 — Inference

**The hypergraph is not used at inference.** Evaluation is GRAM-faithful: raw `z` (no
refinement), masked Gramian volume, then ITM re-ranking of the top 50. The graph's whole
contribution is the encoder weights it shaped during training; the masked volume, by contrast,
is active in training *and* evaluation.

---

## 4. Configuration flags

Set in `model_cfg` of a pretrain/finetune config.

| Flag | Default | Effect |
|---|---|---|
| `stage` | `"A"` | `A` = plain GRAM (no hypergraph, `hgnn` not constructed). `B` = hypergraph on. |
| `masked_volume` | `true` | Step 8 masking. `false` → vanilla GRAM: missing modality ⇒ singular Gram ⇒ volume 0. |
| `semantic_edges` | `false` | Enable cross-document edges (Step 3). Doc edges are always on. |
| `sem_edge_presence_mask` | `true` | Apply `p_{j,m}` to `H_sem` too. `false` → the graph imputes missing modalities. |
| `sem_edge_weighted` | `true` | `w_ij = cos(c_i,c_j)`. `false` → binary edges (all neighbours count equally). |
| `sem_sim_std` | unset | Similarity floor `σ` (Step 3). Unset ⇒ no floor, selection is purely rank-based. |
| `knn_k` | 4 | Neighbour budget per document, before the adaptive clamp. |
| `edge_dropout` | 0.3 | Nominal rate; symmetrisation makes the **effective** rate `1−(1−p)² = 51%`. |
| `hgnn_layers` | 2 | Message-passing layers (hard-capped at 2 — more over-smooths). |
| `global_graph` | `false` | Build ONE hypergraph over the whole DDP batch instead of one per GPU shard. See below. |
| `w_doc`, `w_reg` | 0, 0 | Weights of `L_doc`, `L_reg`. |

Run-level: `keep_last_n_ckpt` (default 1) retains a rolling window of model checkpoints so the
best can be chosen post-hoc; optimizer states always prune to the newest. `bf16: true` now
selects bf16 autocast correctly (it previously fell through to full fp32).

### Global vs per-shard graph

By default the graph is built **per GPU shard** — `_hg_refine` runs before the cross-GPU gather,
so 4 GPUs give 4 disjoint graphs of `B/4` documents and clips on different GPUs can never be
neighbours. `global_graph: true` gathers the 512-d contrastive embeddings (~1.5 MB/step for
T+V+A over 4 ranks — never encoder features), builds one graph over the full batch, refines, and
slices each rank's rows back out. The contrastive loss was already global; this makes the *graph*
global too, so neighbours are drawn from `world_size`x more candidates.

Two correctness requirements, both handled:

- Gathering uses `all_gather_with_grad`, not `concat_all_gather`. A refined embedding depends on
  every document in the graph, so a no-grad gather would silently drop the cross-rank terms of
  `∂ẑ_i/∂z_j`.
- Edge dropout draws from a generator seeded identically on every rank and advanced per step.
  Each rank rebuilds the same global adjacency independently, so an unsynchronised draw would
  give each rank a different graph and DDP would average gradients of different functions.

**`knn_k` must be recalibrated with it, and the two settings are coupled.** Density is
`k_eff/(B−1)` where `k_eff = min(knn_k, max(2, ⌊B/4⌋))`:

| | `knn_k=8` | `knn_k=32` |
|---|---|---|
| per-shard, B=64 | k_eff 8 → **12.7%** | k_eff 16 (clamped by ⌊B/4⌋) → 25.4% |
| global, B=256 | k_eff 8 → 3.1% | k_eff 32 → **12.5%** |

`hyperalign.json` ships `global_graph: true` with `knn_k: 32`, which holds the original 12.7%
density at the larger batch.

**`sem_sim_std` must be set against measured statistics, not assumed ones.** Real caption
embeddings are far more spread than a Gaussian intuition suggests — the first `[EDGES]` line on
VAST data reported `batch_cos=0.259 (sd 0.513)`, with retained edges at `edge_cos=0.730`, i.e.
z=+0.92. Since the threshold is `mean + sigma*sd` and cosine is bounded by 1.0, any
`sem_sim_std` above ~1.4 exceeds the bound and silently kills **every** semantic edge (the
adjacency sums to zero, `semantic_incidence` returns `None`, and only doc edges remain). The
config ships 0.5 (threshold ~0.516) accordingly. Re-read `batch_cos`/`sd` from `[EDGES]` after
any change to the text encoder or batch size before touching this value. Note `knn_k=32` in *per-shard* mode would be clamped to 16 by the
⌊B/4⌋ term and double the density instead — so the two flags must be changed together, and
`gram_base*.json` (stage A, no graph) are unaffected. `sem_sim_std` improves for free: its
mean/std are estimated over ~65k pairs instead of ~4k.

### Ablation matrix

| Config | stage | masked_volume | Isolates |
|---|---|---|---|
| `gram_base.json` | A | false | vanilla GRAM — the published baseline |
| `gram_base_maskedvol.json` | A | true | the masked volume alone |
| `hyperalign.json` | B | true | the full model |

**The masked volume is not yet measured on real data.** `gram_base_maskedvol` matched `gram_base`
on the training-time validation metric — but that metric is MSR-VTT only, and MSR-VTT has 100%
audio coverage, so no zero rows ever reach `volume_computation_masked` there and the mask cannot
fire. That agreement is therefore uninformative, not evidence of a no-op. No zero-shot eval of
`gram_base_maskedvol` has been run.

Where it *should* be measurable: `data/audio_mapper.py` returns `torch.zeros(...)` when it finds no
audio file, so audio genuinely is zero-filled, and ActivityNet is missing `.wav` for 232 of its
4917 test clips (4.7%). Masked and unmasked volumes must differ on `activitynet_tva`. Coverage by
benchmark:

| benchmark | annotated | has video | + has audio |
|---|---|---|---|
| msrvtt | 1000 | 1000 | 1000 |
| didemo | 1003 | 1003 | 1003 |
| activitynet | 4917 | 4917 | **4685** |
| vatex | 1500 | 1358 | 1358 |

So the ablation to run is a zero-shot eval of both stage-A configs on ActivityNet, checking the
`[VOLUME]` log line reports `masked_volume=` as expected in each. Until then the masked volume is
supported only by the synthetic harness (`HA_DROP_MOD`, §5) — 99–101% retention at any drop rate
versus collapse to ~4% — which is a real result but on injected rather than naturally missing
modalities.

---

## 5. Running

```bash
# smoke tests (debug QOS, ~30 min)
sbatch slurm_scripts/smoke_train.sh
sbatch slurm_scripts/ft_smoke.sh msrvtt

# pretraining (24 h; auto-resumes from $HA_WORK/workdir_*/4model on resubmit)
sbatch slurm_scripts/run_pretrain.sh            # HyperAlign, stage B
sbatch slurm_scripts/run_gram_base.sh           # ablation, stage A
sbatch slurm_scripts/run_gram_base_maskedvol.sh # ablation, stage A + masked volume

# finetuning: all 5 benchmarks (msrvtt_depth chains after msrvtt)
bash slurm_scripts/finetune_all.sh

# evaluation
export GRAM_CKPT=$HA_WORK/workdir_pretrain/4model/ckpt/best_*.pt
sbatch benchmark_eval/eval_zeroshot.sh          # 12 benchmark/mode configs + summary table
sbatch benchmark_eval/eval_finetune.sh msrvtt

# the control: zero-shot score of the VAST checkpoint we initialise FROM, before any of our
# training. Every "we gained +X over the baseline" claim is measured against this number, so it
# has to be measured, not assumed. GRAM_TRAIN_CFG points model_cfg at stage A because the VAST
# checkpoint carries no `hgnn` weights.
EVAL_RES_DIR=$HA_WORK/eval_results_vastinit \
GRAM_CKPT=$HA_WORK/pretrained_weights/VAST_foundation/pretrain_vast/ckpt/model_step_204994.pt \
GRAM_TRAIN_CFG=config/gram/pretrain_cfg/gram_base.json \
  sbatch benchmark_eval/eval_zeroshot.sh

# missing-modality robustness: drop a modality from a growing share of gallery clips,
# with the masked volume ON and OFF, on identical inputs
sbatch benchmark_eval/eval_missing_modality.sh msrvtt_tva a
```

`benchmark_eval/gram_paper_baselines.json` holds the published GRAM numbers plus ~15 competitor
methods from GRAM's own tables, so results can be compared without re-reading the paper.
`eval_summary.py` prints ours-vs-paper automatically after `eval_zeroshot.sh`.

## 6. Training diagnostics

Two lines are logged every 50 steps on rank 0:

```
[GATE]  step~51: [0.83, 0.71]
[EDGES] step~51: B=64 k=8 edges=107 deg=3.34 isolated=0/64 | edge_cos=0.412 batch_cos=0.180(sd 0.140) z=+1.66
```

- **`[GATE]`** — the residual gates. Trending to 0 means the model is switching the graph off.
- **`[EDGES]`** — graph health. `isolated` climbing toward `B` means `sem_sim_std` is too
  aggressive for the data; lower it. The **z-score** (how far retained-edge similarity sits above
  the batch mean, in batch std units) is the quality signal: near 0 means the semantic wiring is
  selecting no better than chance, since edge *count* alone cannot distinguish a graph wiring
  genuine topic-mates from one wiring arbitrary pairs.

### The recipe, read off GRAM's released checkpoint

GRAM publishes its pretrained checkpoint, and the training directory recorded inside it names the
recipe outright: `finetuneVolume256batchlossonlyvolume4Mod120k/ckpt/model_step_459.pt`. Batch 256,
volume-only loss, 4 modalities, 120k samples — so `120000/256 = 468` steps, and the released
checkpoint is step 459. **GRAM pretrains for one epoch.**

This repo ran five, and the consequence is not just length. `utils/sched.py` parameterises the LR
schedule by `num_train_steps`, so stretching the run stretches the decay:

| | `num_train_steps` | LR at step 459 |
|---|---|---|
| GRAM | 468 | 2% of peak — annealed |
| this repo, `epoch: 5` | 2649 | 92% of peak — mid-flight |

Every checkpoint we compared against the paper was an un-annealed model that had seen a comparable
number of samples. `epoch` is now `1`.

The same arithmetic explains the "early peak" that several runs showed. Warmup ends at
`warmup_ratio × num_train_steps = 0.1 × 2649 = 265`, and the first validation fired at
`num_train_steps // valid_freq - 1 = 264`, because `warmup_ratio` and `1/valid_freq` were both
0.1. The first measurement of every run landed exactly on peak learning rate. It was a config
coincidence, not collapse, saturation, or over-training.

### Validation resolution

`valid_freq` is a *count*, not an interval: `valid_steps = num_train_steps // valid_freq - 1`.
At the original `valid_freq: 10` a 2,649-step run validated only ten times, the first at step
263. With 530 steps to one pass over the 135.7k clips actually on disk, that is two measurements
per epoch — and three of the four runs recorded their best score at the *very first* one. A peak
at the first sample is not evidence of a peak; it is the absence of any earlier sample. It is
equally consistent with the model having peaked at step 30, or with it never having improved on
its initialisation at all. `valid_freq` is now `20`, which over the corrected 530-step run means a
validation every ~25 steps — about twenty points inside the single epoch, and enough resolution for
`save_best` to catch the maximum rather than the first sample it happens to see.

## 7. Results

### The paper's table is not a reachable target here

GRAM's released checkpoint (`GRAM_pretrained_TVAS/ckpt/model_step_459.pt`), scored by this
harness, lands **3.6 R@1 below GRAM's own published table**. That is GRAM's weights, so nothing
about our method is involved. Scoring the same checkpoint with GRAM's *unmodified*
`evaluation_mm.py` (`GRAM_UPSTREAM_EVAL=1`) reproduces our numbers to within 0.2 everywhere, so
the shortfall is not our evaluation code either:

| | ours | GRAM's eval | paper |
|---|---|---|---|
| msrvtt tva | 53.2 | 53.4 | 54.2 |
| msrvtt tvas | 52.5 | 52.5 | 54.8 |
| didemo tva | 50.8 | 50.7 | 54.2 |
| activitynet tva | 56.3 | 56.3 | 59.0 |
| vatex tva | 77.2 | 77.3 | 83.9 |
| vatex tvas | 76.3 | 76.3 | 83.5 |

With code, metric, galleries, configs and weights all eliminated, the residual is the video data,
and it orders the way provenance predicts — smallest where the files are identical for everyone,
largest where they must be rebuilt from YouTube:

| benchmark | gap (GRAM's weights) | video source |
|---|---|---|
| msrvtt | −1.0 / −2.3 | fixed archive |
| didemo | −3.4 | Flickr / YFCC100M |
| activitynet | −2.7 | YouTube |
| vatex | −6.7 / −7.2 | YouTube, 10-second Kinetics segments |

**Consequence for reporting:** comparing our runs against the published column charges them ~3.6
points they did not cause. Use `compare_runs.py --ref=gram_official`, which re-baselines onto
GRAM's checkpoint measured here, and show the published table separately with this discrepancy
stated.

### Recipe vs hypergraph (mean delta vs `gram_official`, same harness)

|  | 5 epoch | 1 epoch |
|---|---|---|
| stage A (plain GRAM) | −1.0 | **+0.3** |
| stage B (hypergraph) | −0.5 | −0.3 |

Split by whether the benchmark drove checkpoint selection (`save_best` selects on MSR-VTT `tvas`):

| | MSR-VTT (selected on) | held out (didemo/anet/vatex) |
|---|---|---|
| base_5ep | −0.9 | −1.0 |
| ha_5ep | −1.0 | −0.3 |
| base_1ep | −0.6 | **+0.7** |
| ha_1ep | 0.0 | −0.5 |

Two findings:

1. **The corrected recipe is worth +1.3 on stage A**, and stage A at one epoch is the only
   configuration that beats GRAM's released checkpoint on data it was not selected against.
2. **The hypergraph does not help, and under the correct recipe it hurts.** Its sign flips with
   the schedule: +0.5 at five epochs, −1.2 at one (held-out). A term that helps only under an
   un-annealed schedule was compensating for that schedule, not adding signal. This is consistent
   with `_hg_refine` being unreachable at inference (`model/gram.py:539`, inside `if
   compute_loss:`) — the graph's only channel is the encoder weights it leaves behind.

One seed per cell. Run-to-run spread measured across three matched runs is ~1.2 R@1, so the recipe
effect clears it and the hypergraph effect sits at it; a second seed for the two one-epoch cells
would separate "neutral" from "harmful".

### The `tv` column is a different protocol from the paper's

Upstream builds the volume from every modality the loader supplied and ignores the task string, so
its `tv`, `tva` and `tvas` rows share one volume and differ only in ITM conditioning. Ours makes
the task string select the arity, so `ret%tv` is a genuine 2-modal volume — the paper's own text
("with two modalities the volume computation degenerates to the area of the triangle") supports
this reading, but it means our `tv` rows do not measure what the published `tv` rows measure. It
affects `tv` only: for `tva`/`tvas` both versions use the same modalities, which is why the
upstream-eval control above agrees to 0.2.

## 8. Known gaps

- **`L_reg` does not mask missing modalities.** Zero rows enter the batch variance, inflating it
  and weakening the hinge exactly where data is sparsest.
- **ITM re-ranking is not presence-masked** — it conditions on full `condition_feats_*` even when
  the volume saw fewer modalities.
- **The `tv` column uses a different metric path** than `tva`/`tvas` (pairwise ITM vs
  Gramian-volume ITM; see the comment in `eval_summary.py`), so within-row comparisons across
  modality settings are not strictly like-for-like.
- **VATEX gallery** is the 1500-clip test split filtered to clips on disk (~1358); the gallery
  size is printed by `make_configs.py` and must be footnoted, since a smaller gallery inflates
  recall.
- **Whether a genuinely missing modality yields a zero feature** is unverified — presence
  detection assumes it, but the zero-filling happens in `data/`, which is not part of this repo.
