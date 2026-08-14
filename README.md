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

# pretraining. HA_RUN_NAME picks the workdir, HA_CFG picks the config, so concurrent runs never
# share an output directory -- the launcher refuses to start if a live job already holds one.
sbatch slurm_scripts/run_pretrain.sh                                   # -> workdir_pretrain
HA_RUN_NAME=distill sbatch slurm_scripts/run_pretrain.sh               # -> workdir_distill
HA_RUN_NAME=base HA_CFG=config/gram/pretrain_cfg/gram_base.json \
  sbatch slurm_scripts/run_pretrain.sh                                 # -> workdir_base
# (run_gram_base.sh / run_gram_base_maskedvol.sh predate HA_CFG and hardcode workdir_gram_base*;
#  prefer the HA_CFG form above, which cannot collide.)

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

The paper states the same thing directly, and one more value we had wrong:

> "We set the batch size to 256 and a single epoch pretraining on 4 NVIDIA A100 cards."
> "We pretrain the GRAM-based model on a subset of the VAST27M dataset comprising 150k random
> samples with a learning rate of 1e-4 using the AdamW optimizer with weight decay and batch
> size of 256."

`config/gram/default_run_cfg.json` — GRAM's own file, unmodified — agrees: `learning_rate: 1e-4`,
`optim: adamw`. But every pretrain config here overrode it to `2e-5`, so all runs up to and
including `ha_wraw` and `distill` trained at **one fifth of the paper's learning rate**. That was an
inherited guess of the same kind as the `epoch5/bs128` header, and it was never checked against the
source. Corrected to `1e-4`; every result predating that change is at the wrong LR and needs
rerunning before it can be compared with GRAM's checkpoint.

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
validation every ~25 steps — about twenty points inside the single epoch.

That resolution is now purely diagnostic. `save_best` is `false`, matching GRAM's own
`default_run_cfg` and their released `model_step_459.pt`, so the evaluated checkpoint is always the
final step regardless of what validation did. Selecting the best of ~20 validations on MSR-VTT —
which is also a reported benchmark — was worth roughly 3 R@1 on that benchmark relative to held-out
ones, an advantage GRAM does not take.

## 7. Results

All numbers below share one protocol, verified value-by-value against a primary source:

| | value | verified by |
|---|---|---|
| batch size | 256 | checkpoint dir name + paper |
| epochs | 1 | paper + `model_step_459` ≈ 120k/256 |
| learning rate | 1e-4 | paper + GRAM's own `default_run_cfg` |
| optimizer | AdamW | paper + their default |
| vision frames | 2 | **GRAM checkpoint's `vision_frame_embedding` is (1, 2, 768)** |
| audio segments | 1 | **`audio_frame_embedding` is (1, 1, 768)** |
| modalities | T+V+A+S | `subtitle` present in 150,154/150,154 annotations |
| checkpoint | final step, `save_best: false` | their default; their release is `model_step_*` |

Earlier results in this repo were produced at `lr 2e-5` with `save_best` selection on MSR-VTT.
Both differ from GRAM, so those numbers are not comparable to GRAM's checkpoint and have been
quarantined (`tools/cleanup_results.sh`). Nothing below depends on them.

### Reproduction: our pipeline against GRAM's own checkpoint

Zero-shot T2V R@1, same harness, same galleries, same protocol.

| bench | mode | GRAM ckpt | our repro | paper |
|---|---|---|---|---|
| msrvtt | tv | 51.9 | **52.6** | 52.8 |
| msrvtt | tva | 53.2 | 52.2 | 54.2 |
| msrvtt | tvas | 52.5 | 51.9 | 54.8 |
| didemo | tv | 50.9 | 49.2 | 54.0 |
| didemo | tva | 50.8 | 49.4 | 54.2 |
| activitynet | tv | 55.2 | 53.8 | 58.9 |
| activitynet | tva | 56.3 | 52.2 | 59.0 |
| vatex | tv | 76.2 | 76.2 | 81.1 |
| vatex | tva | 77.2 | 77.5 | 83.9 |
| vatex | tvas | 76.3 | 76.8 | 83.5 |
| | **mean vs paper** | **−3.6** | **−4.5** | 0.0 |

Our reproduction is **−0.9** from GRAM's checkpoint, against a measured run-to-run spread of ~1.2 —
i.e. indistinguishable. The pipeline reproduces GRAM.

The remaining **−3.6 for GRAM's own weights against GRAM's own table** is not ours: scoring that
checkpoint with GRAM's unmodified `evaluation_mm.py` (`GRAM_UPSTREAM_EVAL=1`) reproduces our
numbers to within 0.2 on every shared setting. Annotations are md5-identical to upstream,
`default_model_cfg.json` is identical, `utils/volume.py` is upstream plus additions, and the
eval-path feature extraction is unchanged. The residual tracks how much each benchmark's video
data must be rebuilt from third-party sources (msrvtt −1.0, didemo −3.4, activitynet −2.7,
vatex −6.7), but that ordering is suggestive, not proven.

### The hypergraph does not work

Same recipe, same harness, final checkpoints, one variable.

| bench | mode | hypergraph − plain GRAM |
|---|---|---|
| msrvtt | tv / tva / tvas | −0.6 / 0.0 / −2.7 |
| didemo | tv / tva | −3.3 / −4.6 |
| activitynet | tv / tva | −3.3 / −3.1 |
| vatex | tv / tva / tvas | −0.9 / −2.2 / −4.5 |
| | **mean** | **−2.5** |

Negative on 9 of 10 settings, at twice the noise floor. The validation curves show the mechanism:

```
step    plain GRAM   hypergraph
  24        55.6         54.8
  74        51.7         50.7
 324        51.0         49.3
 524        51.9         49.2    <- evaluated checkpoint
```

Both dip as the volume objective disrupts the VAST representation. **Plain GRAM re-converges and
plateaus near 52; the hypergraph does not recover.** The residual gate climbs 0.174 → 0.201 across
exactly that window, so the graph's influence grows during the period the model needs to
re-stabilise.

This is consistent with the structural argument: the doc hyperedge duplicates the cross-modal
coupling the Gramian volume already performs, and the semantic hyperedge — the genuinely novel
part — is built from caption similarity and so cannot run at inference without leaking the query.
The novel half is unavailable at test time and the available half is redundant.

An earlier round at `lr 2e-5` measured the effect as neutral rather than negative. The wrong
learning rate compressed the difference: the model barely adapted, so neither did the gap.

### What remains untested

`impute_missing` (see §4). Previously a missing modality was disconnected from its hyperedges in
*both* message directions, so the graph could never reconstruct it and the masked volume did all
the work. With an asymmetric incidence the vertex receives its document's edge summary and is
reconstructed from the modalities the clip does have — per-clip, inductive, valid at inference.
That is a different claim, in a regime where GRAM's volume is undefined rather than merely
suboptimal, and nothing measured here speaks to it. It also needs training-time modality dropout
to be a fair test: `finetune_area` filters to clips with both audio and video (136,674 of 136,694),
so the HGNN has never seen a zero vertex.

The missing-modality robustness figure previously quoted (99–101% retention vs ~4% collapse) is
**withdrawn**. It came from a sweep that read a stale smoke config — a 22-step checkpoint on an
80-clip gallery — compounded by a presence bug that disabled the mask entirely. Both are fixed;
the number has to be re-earned.


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
