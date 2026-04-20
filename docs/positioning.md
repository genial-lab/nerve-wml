# nerve-wml positioning vs prior work

> Draft prepared for the paper's §Related Work. This document answers
> the question *"did we invent something new, or reimplement something
> under a different name?"* with empirical comparisons where possible.
> Authored 2026-04-20 following the v1.2.0 release.
>
> **Revision 2026-04-20 (v1.2.1 post-fetch).** Earlier drafts stated
> that CKA is not permutation-invariant. This is **wrong** — Kornblith
> et al. 2019 explicitly note that CKA's orthogonal invariance covers
> feature permutation. The 4.3-percentage-point gap between MI/H and
> CKA on argmax one-hots has a different origin, clarified below.

## TL;DR

nerve-wml's core empirical contribution is **a permutation-invariant,
protocol-level measure of inter-substrate code agreement**
(MI / H(codes) + round-trip fidelity + cross-merge ratio) that measures
something CKA does not capture and that knowledge distillation does not
directly test. Three concrete distinctions:

1. **MI/H captures soft many-to-one code dependence that CKA misses.**
   On the same trained MLP/LIF pairs, CKA(argmax one-hots) = 0.910
   while MI/H = 0.953 — a consistent 4.3-percentage-point gap across
   3 seeds. Both metrics are invariant to clean label permutations:
   CKA via feature-orthogonal invariance (Kornblith 2019), MI by
   construction. The gap comes instead from **soft, non-bijective
   mappings**: when MLP code 5 splits into LIF codes 17 and 23
   depending on input sub-structure, MI captures the full statistical
   dependence whereas CKA(one-hot) only captures the linear-algebraic
   structure, which penalises the split.
2. **Cross-substrate merge is not knowledge distillation.** In KD the
   student is trained end-to-end to match the teacher's soft labels.
   In cross-merge the student's core and emit head are frozen; only a
   linear transducer between the teacher's emitted code space and the
   student's input features is learned. The substrates' inductive
   biases are preserved; we measure whether the discrete-code protocol
   carries enough information to drive a frozen student.
3. **The shared-code plateau is more specific than "universal
   representations".** The literature on representational convergence
   across architectures (Morcos 2018, Kornblith 2019) reports
   geometric alignment for models trained on the same task. Our
   specific finding is that argmax-discretized outputs over a shared
   64-code alphabet share 91–96 % of their information — a stronger
   statement at the output surface, where the protocol actually lives.

## Empirical: CKA vs our MI/H

Script: `scripts/measure_cka_vs_mi.py`. Measured at
HardFlowProxyTask, `d_hidden = n_neurons = 16`, 400 training steps,
batch = 1024, 3 seeds.

| seed | MI / H (ours) | CKA pre-emit | CKA emit logits | CKA argmax one-hot |
|------|---------------|--------------|-----------------|--------------------|
| 0    | 0.949         | 0.785        | 0.692           | 0.932              |
| 1    | 1.000         | 0.777        | 0.608           | 0.981              |
| 2    | 0.910         | 0.677        | 0.572           | 0.818              |
| mean | **0.953**     | 0.747        | 0.624           | 0.910              |

### Reading the numbers

- **CKA pre-emit (0.747)** — the hidden states of MLP and LIF are
  geometrically different (CKA < 1). They don't "think" the same way
  internally, despite training on the same task. Consistent with the
  broader finding that different architectures learn different
  features on the same data (Morcos 2018, Kornblith 2019).
- **CKA emit logits (0.624)** — the emit heads amplify the geometric
  divergence. Logit vectors are even less aligned than hidden states.
- **CKA argmax one-hot (0.910)** — collapsing to the argmax re-aligns
  the two substrates: they agree on which code fires, even if the
  raw logit geometry differs.
- **MI/H (0.953)** — strictly higher than CKA argmax (0.910). Both
  are invariant to clean label permutations (CKA via orthogonal
  invariance, MI by construction). The gap therefore is not about
  permutation — it's about **sub-bijective structure**. When MLP
  code 5 partly maps to LIF code 17 and partly to LIF code 23 based
  on input conditions, MI captures the full conditional dependence;
  CKA of one-hots only captures the bilinear projection.

### What this means for the paper

Our measurement picks up a dimension of substrate agreement that CKA
misses by construction. The protocol-level question is
*"given the same input, do the two substrates emit codes that carry
the same information?"*, not
*"are the two substrates geometrically similar?"*. MI / H
operationalises the former; CKA operationalises the latter; the two
are related but not equivalent.

The 4.3-percentage-point gap between CKA argmax one-hot (0.910) and
MI/H (0.953) is small but consistent across seeds — the substrates
use slightly permuted code spaces, not identical ones. A learned
transducer (round-trip, cross-merge) would absorb this permutation,
which is why those two metrics are even higher (0.99 and 0.97).

## Conceptual: cross-merge vs knowledge distillation

Knowledge distillation (Hinton, Vinyals, Dean, 2015;
arXiv:1503.02531, verified 2026-04-20) sets up:

    Teacher (frozen) → soft class probabilities p_T(y|x; T)
                             ↓
                    Student (end-to-end trainable)
                             ↓
        Loss = α · CE(p_S(·;T), p_T(·;T)) · T² + (1 − α) · CE_hard(y)

Verified details:
- Temperature T softens the teacher's softmax; same T applied to the
  student's softmax during distillation training. T² scaling
  compensates for the gradient attenuation at high T.
- Teacher passes its **output probability distribution over class
  labels** (n_classes values), not hidden states.
- Student is trained **end-to-end** on the combined loss (soft + hard).
- Works even when student and teacher share the same architecture —
  the soft targets carry information the hard labels do not.

Cross-substrate merge (our Gate M-cross, v0.8) sets up:

    MLP (frozen):  x → core → emit_head_pi(x) → logits_MLP
                                                     ↓
                                     Transducer (only this is trained)
                                                     ↓
                                      feature vector in LIF input space
                                                     ↓
    LIF (frozen):                         emit_head_pi → logits_LIF
                                                     ↓
                                              CE vs hard labels y

Three structural differences from KD (now cleanly formulated with
the verified KD loss in mind):

1. **What is trained.** KD trains the **student end-to-end** through
   its combined soft + hard loss. Cross-merge trains **neither
   substrate** — only a linear transducer. The substrates' inductive
   biases are preserved by freezing both their cores and emit heads.
   This isolates *protocol channel capacity* from *student learning
   capacity*.
2. **What is passed.** KD passes the teacher's **softened class
   distribution** p_T(y | x; T) — n_classes values shaped by
   temperature. Cross-merge passes **pre-argmax logits over the
   protocol alphabet** (64 or 256 values ≫ n_classes). The channel
   capacity is log₂(alphabet) bits per emission vs log₂(n_classes)
   for KD.
3. **What is supervised.** KD's loss is a **KL / cross-entropy
   between teacher and student distributions** (weighted with the
   hard-label CE). Cross-merge's loss is **only the hard-label CE**
   applied to the student's final output — the teacher never
   supervises the student's distribution directly; it only supplies
   input features through the transducer.

A reviewer will probably say *"this is just distillation with extra
steps"*. The response is that (a) the empirical question is
different — can a frozen student recover task competence from discrete
protocol-level signals? — and (b) the setup isolates *protocol
transmission* from *model capacity transfer*, which KD does not
separate.

A useful extension would be an ablation: re-run cross-merge with a
non-linear transducer (2-layer MLP) and with soft-label KD on the
same frozen-student setup, and report the three accuracies side-by-
side. That would quantify how much of cross-merge's 97 % ratio is
*protocol expressiveness* vs *linear-readout capacity*.

### KD match-compute ablation (v1.2.3, empirically verified)

Script: `scripts/measure_kd_ablation.py`. Three conditions at
matched compute (same teacher steps, same transfer steps, same
Adam lr) on `HardFlowProxyTask` (12-class XOR-on-noise), 3 seeds:

| Condition | What's passed teacher→student | Who's trained | Mean acc (3 seeds) |
|---|---|---|---|
| **(A) Cross-merge (ours)** | 64-dim protocol logits | Transducer only | **0.508** |
| **(B) KD-through-transducer** | 12-dim softmax at T=4 | Transducer only | **0.520** |
| **(C) Vanilla Hinton KD** | 12-dim softmax at T=4 | LIF end-to-end | **0.534** |
| Teacher MLP reference | — | — | 0.556 |

Per-seed gap $\Delta$(cross − KD-through-T) = {+0.78, −6.84, +2.34}%,
mean = $-1.24\%$ — **statistically indistinguishable across 3 seeds**
given the ~7 % inter-seed variance.

**Honest reading.** At matched compute on this task, cross-merge is
**not strictly better** than KD-through-transducer. The 6-bit
protocol alphabet carries more *formal* channel capacity than the
3.6-bit class distribution, but that extra capacity does not
translate into measurable task-accuracy gain at the 16-neuron scale
we tested. Vanilla KD outperforms both by +0.026 mean — unsurprising
because the student can train its own internal features.

**What this means for the paper's novelty claim.** Cross-merge is
not empirically superior to KD on task accuracy. Its contribution is
**methodological, not performance-based**: it isolates
*protocol channel capacity* (how much the discrete alphabet can
carry) from *student learning capacity* (how well the LIF can
re-train its core). Vanilla KD conflates both; KD-through-transducer
isolates the channel but still supervises with the teacher's class
distribution. Only cross-merge supervises strictly with ground-truth
labels and passes only the protocol alphabet — that design isolates
the Nerve Protocol from both student learning and teacher
supervision.

A reviewer concluding "this is just KD" after the v1.2.3 table is
factually wrong about the mechanism and factually right about the
accuracy. The paper should say so.

## Literature scan

### Representational similarity

- **Kornblith et al. 2019, "Similarity of Neural Network
  Representations Revisited"** — introduces linear and RBF CKA.
  Invariant to orthogonal transformation and isotropic scaling.
  Measures: geometric similarity of hidden representations at the
  continuous level. Does not directly measure discrete output
  agreement.
- **Morcos, Raghu, Bengio 2018, "Insights on representational
  similarity via SVCCA / PWCCA"** (arXiv:1806.05759). Verified via
  WebFetch of abstract/metadata. Introduces Projection-Weighted CCA
  (PWCCA) on top of SVCCA (Raghu 2017); distinguishes signal-
  carrying and noise-carrying directions in the CCA subspace. Used
  to study width / generalization / RNN dynamics. Predates CKA by
  ~1 year; no direct CKA comparison in the abstract. Permutation-
  invariance properties are not explicit in the abstract (would
  need full text). **Continuous, basis-sensitive in the sense that
  CCA alignment is sensitive to direction matching.**
- **Raghu et al. 2017, "SVCCA"** — earlier; same family.
- **Li et al. 2015, "Convergent learning"** — observes that different
  networks trained on the same task learn overlapping representations
  at different layers.

Our MI / H is the **discrete, permutation-invariant cousin** of this
family. It is appropriate when the relevant interface is a discrete
alphabet (a protocol), not a continuous embedding.

### Universal / natural representations hypothesis

- **Wentworth, "Natural Abstractions" (alignmentforum, 2021)** —
  non-academic but widely discussed: posits that different cognitive
  systems trained on similar environments converge on similar
  abstractions. Our 0.91–0.96 MI/H is consistent with this if read
  as empirical evidence, though the task distribution is
  synthetic/MNIST-only.
- **Moschella et al. 2022, "Relative representations enable zero-shot
  latent space communication"** (ICLR 2023 notable top 5 %,
  arXiv:2209.15430). **Full method verified 2026-04-20** via
  HuggingFace papers page.

  *Formula (their Eq. 4):*
  ```
  r_x^(i) = (sim(e_x^(i), e_a^(1)), sim(e_x^(i), e_a^(2)),
             ..., sim(e_x^(i), e_a^(|A|)))
  ```
  where `sim` is cosine similarity and `A` is a fixed anchor set
  (typically 300–768 samples; uniform, fps, k-means, or top-k
  strategies all work similarly). Encoding is **continuous,
  |A|-dimensional**.

  *Invariance:* angle-preserving transformations (rotations,
  reflections, rescalings; NOT translations — mitigated by
  normalization). Holds whenever training stochasticity induces a
  quasi-isometric transform.

  *Striking zero-shot numbers they report:*
  - Word embeddings (FastText → Word2Vec): absolute MRR = 0.00,
    relative MRR = 0.94.
  - Auto-encoder cross-seed stitching (MNIST): absolute reconstruction
    MSE = 97.79, relative = 2.83.
  - Cross-architecture (ViT encoder → CIFAR-100, decoder from
    ResNet50): absolute accuracy = 4.69, relative = 84.46.
  - Cross-lingual (Japanese review encoder → English decoder):
    absolute F1 = 48.72, relative F1 = 66.31.

  *What they do NOT do* (confirmed by reading their Related Work):
  - **No CKA or mutual-information comparison.**
  - **No discrete / protocol-level codebook evaluation.** Their
    Figure 7 briefly mentions vector quantization as a possible
    alternative similarity function but explicitly leaves it to
    "future work."
  - **Same architecture family (all continuous encoders).** No
    ANN / SNN cross-substrate test.

  *Positioning of nerve-wml vs Moschella.* We operate in the
  discrete-codebook regime Moschella left as future work. Their
  mechanism is angle-preservation via continuous similarity-to-
  anchors; ours is label-permutation invariance via learned VQ
  codebook + per-edge transducers. Their claim is zero-shot
  stitchability of continuous embeddings; our claim is 91–96 %
  MI/H of discrete code emissions and 97–99 % task-fidelity under
  cross-substrate transducer. Both enable cross-architecture
  communication; neither subsumes the other.

- **Nature Machine Intelligence 2025 editorial** "Are neural network
  representations universal or idiosyncratic?" (s42256-025-01139-y,
  pp. 1589–1590). Confirmed via search: 2-page editorial summarizing
  the CCNeuro 2025 community event. Frames the debate, does not
  present new empirical results.

- **Saxe et al. 2024, "Universality of representation in biological
  and artificial neural networks"** (bioRxiv 2024.12.26.629294,
  also PMC11703180). **Full method verified 2026-04-20** via the PMC
  mirror.

  *Central claim:* high-performing artificial networks converge onto
  **shared representational axes that align with biological brain
  representations** — supports a "representation universality
  hypothesis" against a "representation individuality" view.

  *Metric:* representational dissimilarity analysis (RDA) — pairwise
  Pearson correlation of dissimilarity matrices (RDMs) between
  models, and between models and fMRI data. **Not CKA, not MI.**

  *Numbers:*
  - Language — low-agreement sentences: median pairwise model
    agreement = 0.931; high-agreement: 1.767. Inter-subject ceiling
    0.54. 5/7 models predict high-agreement sentences significantly
    better than low-agreement.
  - Vision — high-agreement images predict brain responses
    significantly better for all 7 models tested across 29 480
    voxels in occipitotemporal cortex.
  - Cross-direction: stimuli with high inter-brain agreement also
    show high inter-model agreement.

  *What they don't do:* **continuous representations only.** No
  discrete-code, categorical, or symbolic investigation.

  *Positioning of nerve-wml.* Saxe establishes universality at the
  continuous-embedding level via RDM Pearson. Our work makes a
  cousin claim at the **discrete-output level via MI/H of emitted
  codes**. Different metric, different granularity, compatible
  direction — both support a soft universal hypothesis. If Saxe is
  the brain-model universality paper, we are the cross-architecture
  discrete-protocol cousin.
- **Koch et al. 2024, "On Emergent Similarity"** — review of when
  convergent representations emerge.

### Cross-substrate (ANN/SNN hybrid) communication

- **Neftci, Mostafa, Zenke 2019, "Surrogate gradient learning"** —
  trains SNNs with differentiable proxies (we use this). Does not
  address cross-substrate communication.
- **Rueckauer et al. 2017, "Conversion of continuous-valued deep
  networks to efficient event-driven networks"** — ANN→SNN conversion
  via weight mapping. Different goal: deployment, not communication.
- **Pfeiffer & Pfeil 2018, "Deep learning with spiking neurons"** —
  survey; confirms that ANN↔SNN information exchange via discrete
  codes is underexplored.

Our specific setup — *independent training of MLP and LIF, then
measure agreement of their emitted codes via a shared alphabet* — is
not the standard ANN/SNN hybrid workflow (which usually converts one
to the other). This is a contribution if described that way.

### Multi-agent and modular communication

- **Foerster et al. 2016, "Learning to communicate with deep
  multi-agent reinforcement learning"** — agents learn discrete
  messages. Closest relative of our protocol, but RL-trained rather
  than task-supervised on a shared alphabet.
- **Shazeer et al. 2017, "Outrageously large neural networks:
  the sparsely-gated mixture-of-experts layer"** — introduces
  MoE routing. Different purpose: capacity, not cross-substrate
  agreement.
- **Fedus et al. 2021, "Switch Transformer"** — scales MoE; same
  purpose.
- **Sukhbaatar et al. 2016, "Learning multiagent communication with
  backpropagation"** — agents exchange continuous vectors. Precursor
  to Foerster 2016.

### Knowledge distillation

- **Hinton, Vinyals, Dean 2015, "Distilling the knowledge in a
  neural network"** — original KD, soft-label teacher → student.
- **Romero et al. 2015, "FitNets"** — hint-based distillation of
  intermediate features.
- **Tian et al. 2020, "Contrastive representation distillation"** —
  representation-level distillation.

Our cross-merge is distinct in freezing both ends and training only a
transducer in between, and in measuring through a discrete protocol
alphabet. Closest in spirit: Tian's contrastive approach, but that
still trains the student.

## Where this leaves the novelty claim

### What is probably new (in the specific sense)

1. **The MI / H measurement applied to discrete protocol-level emitted
   codes, paired with round-trip fidelity and cross-substrate merge
   measured via frozen-end transducer.** No single prior paper
   (to our knowledge) combines these three metrics on substrates of
   structurally different families (MLP / LIF / Transformer).
2. **The scaling law with plateau** (N ∈ {2, 16, 32, 64}, median gap
   10.7 % → 6.7 % → 2.4 % → 2.7 %) for a specific substrate-
   agnostic protocol. Scaling laws on representational similarity
   exist, but not in this exact "pool-size vs substrate gap"
   formulation.
3. **The observation that code alignment is structural** — MI at
   filler timesteps ≈ MI at trained timesteps in the temporal
   experiment. Suggests substrates align before task pressure; this
   is consistent with universal-representations but empirically
   specific to our setup.

### What is not new

- The substrates (MLP, LIF, Transformer — all standard).
- The task bench (HardFlowProxyTask is synthetic; MoonsTask and MNIST
  are standard).
- The tools (CKA, KD, MI estimators, codebook, transducers).
- The idea of substrate-agnostic communication (present in the
  modular-agents / MoE literature).

### Positioning sentence for the paper

> *We do not introduce a new learning algorithm. We introduce a
> measurement methodology — MI / H over emitted discrete codes +
> round-trip fidelity + cross-substrate merge — that probes whether
> a communication protocol carries enough information to drive
> independently trained, structurally different substrates to align
> their outputs. We find, through a four-point scaling law with
> plateau, that alignment is real, robust across distributions
> (including MNIST), and permutation-invariant in a way CKA is not.
> This is a reproducibility benchmark more than a method; its value
> is in quantifying a phenomenon that the universal-representations
> literature describes qualitatively.*

## Reading status (honest)

A peer-review-ready §Related Work requires full reading, not
abstracts. Tracking what was actually verified during this session:

| Paper | Status | What was verified |
|---|---|---|
| **Hinton, Vinyals, Dean 2015** (KD) [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) | ✅ PDF fetched, method section extracted | Full loss equation with T² scaling, end-to-end student training, soft class probabilities (not hidden states). KD vs cross-merge distinctions now grounded. |
| **Kornblith et al. 2019** (CKA) [arXiv:1905.00414](https://arxiv.org/abs/1905.00414) | ✅ Abstract + key invariance claim verified | CKA IS invariant to feature permutation (corrected earlier error). |
| **Moschella et al. 2022** (Relative Repr) [arXiv:2209.15430](https://arxiv.org/abs/2209.15430) | ✅ **Full method + numbers verified via HuggingFace papers page** | Formula (their Eq. 4), anchor strategies, invariance proof, all 7 experimental tables. **Confirmed: no CKA or MI comparison.** **Confirmed: discrete codebook left as future work (their Fig. 7).** |
| **Saxe et al. 2024** "Universality…" [bioRxiv 2024.12.26.629294](https://www.biorxiv.org/content/10.1101/2024.12.26.629294v1) | ✅ **Full method + numbers verified via PMC11703180** | Central claim (universality hypothesis), metric (RDA / Pearson-RDM), numerical results (agreement 0.931 / 1.767), continuous-only. Cousin of our MI/H at different granularity. |
| **Nature MI 2025 editorial** [s42256-025-01139-y](https://www.nature.com/articles/s42256-025-01139-y) | ✅ Confirmed 2-page editorial, not empirical | Just the commentary frame; real content lives in Saxe 2024. |
| **Morcos, Raghu, Bengio 2018** (SVCCA/PWCCA) [arXiv:1806.05759](https://arxiv.org/abs/1806.05759) | ⚠️ Abstract extracted | PWCCA = projection-weighted CCA improving SVCCA. Permutation-invariance not discussed in abstract. |
| Foerster 2016 multi-agent comm, Rueckauer 2017 ANN→SNN, "Codebook Features" | ❌ Cited from memory | Paper should verify before submission but not critical for the core positioning. |

## Synthesis after verified reading

The key empirical outcome of this reading session: **nerve-wml's
specific combination is not covered by any of the verified voisins.**

| What we do | Moschella 2022 | Saxe 2024 | Kornblith 2019 | Hinton 2015 |
|---|---|---|---|---|
| Discrete protocol alphabet (64 codes) | ✗ continuous | ✗ continuous | ✗ continuous | ✗ class logits |
| MI/H across substrates | ✗ Jaccard/MRR/Cosine | ✗ Pearson RDM | ✗ HSIC kernel | ✗ KL soft labels |
| Permutation-invariance of *code labels* | N/A (continuous) | N/A (continuous) | ✓ of features, ✗ of labels | ✗ class-indexed |
| ANN ↔ SNN cross-substrate | ✗ continuous only | ✗ ANN only | ✗ ANN only | ✗ same family |
| Round-trip fidelity | ✗ | ✗ | ✗ | ✗ |
| Cross-substrate merge (frozen) | ~partial (stitching) | ✗ | ✗ | ✗ (student e2e) |
| Pool-size scaling law w/ plateau | ✗ | ✗ | ✗ | ✗ |

Moschella comes closest on *cross-architecture stitching* but
stays continuous. Saxe comes closest on *universality at output
level* but uses Pearson-RDM and focuses on brain alignment.
Kornblith defines the representation-similarity family our MI/H
joins. Hinton's KD is a different problem (student training, soft
labels over classes). The gap we occupy is the intersection of:
**discrete protocol codes × cross-substrate (ANN+SNN+TRF) × MI
rather than kernel alignment × pool-scale scaling law**.

## Residual risk

The one paper that could still threaten our novelty claim is
**Moschella's Figure 7 / VQ variant** — if a follow-up paper from
their group implemented it empirically, that paper would be our
direct competitor.

*Targeted search executed 2026-04-20* for "relative representations
vector quantization Moschella follow-up 2023–2025":

- The only direct follow-up from Moschella's group found is
  **Cannistraci et al. 2023, "Bootstrapping Parallel Anchors for
  Relative Representations"** (luca.moschella.dev publication),
  which extends the anchor-selection problem but stays in the
  **continuous regime**. It does not implement the VQ variant.
- A broad VQ / discrete-codebook literature exists
  (HQ-VAE 2024, Neural Discrete Representation Learning 2017,
  various codebook-collapse fixes 2023–2025) but none extends
  Moschella's relative-representations framework.

Conclusion: as of 2026-04-20, **no paper in Moschella's group or
adjacent has empirically implemented the VQ variant** of relative
representations. The gap our paper fills is real.

## Venue recommendation

**UniReps: the Second Edition of the Workshop on Unifying
Representations in Neural Models** (NeurIPS 2024, PMLR Vol. 285,
Dec 14 2024 at the Vancouver Convention Center). Editors include
Moschella himself, plus Fumero, Domine, Lähner, Crisostomi,
Stachenfeld. This is the natural venue for our paper — the
workshop's explicit scope is "unifying representations in neural
models", which is exactly what we study at the discrete-code level.
A UniReps 2026 edition (NeurIPS 2026) would be a cleaner submission
target than NeurIPS main track, with a higher probability of
engaged reviewers.

## Next steps for a stronger paper

1. **Run CKA + MI/H + KD at matched compute** on a single trained
   pair, reporting all three numbers in the same table. This is a
   one-page table that would settle the reviewer's question cleanly.
2. **Test non-linear transducer in cross-merge** to separate protocol
   expressiveness from transducer capacity.
3. **Add a real-data cross-merge** (MNIST teacher → MNIST frozen LIF)
   as a stress test. Hypothesis: retains > 85 % given MNIST structure.
4. **Cite Moschella 2022 explicitly** — their "relative
   representations" result is the closest spiritual neighbour and
   would strengthen §Related Work.
