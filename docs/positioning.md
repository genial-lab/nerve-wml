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

Knowledge distillation (Hinton, Vinyals, Dean, 2015) sets up:

    Teacher (frozen) → soft labels p_T(y|x)
                             ↓
                    Student (end-to-end trainable)
                             ↓
                    Loss = KL(p_T || p_S)  (+ optional CE on hard labels)

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

Three structural differences from KD:

1. **What is trained.** KD trains the student. Cross-merge trains
   neither substrate — only a linear transducer. Both substrates'
   inductive biases are preserved by construction.
2. **What is passed.** KD passes the teacher's full softmax
   distribution over labels. Cross-merge passes logits over the 64-
   code *protocol alphabet* (typically ≫ n_classes), so the message
   carries more bits than the label entropy. This is the protocol
   channel's capacity.
3. **What is supervised.** KD's loss compares the student's
   distribution to the teacher's. Cross-merge's loss compares the
   student's *final label prediction* to the ground-truth label. The
   teacher provides features through the transducer but never
   supervises the student's output directly.

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

## Literature scan

### Representational similarity

- **Kornblith et al. 2019, "Similarity of Neural Network
  Representations Revisited"** — introduces linear and RBF CKA.
  Invariant to orthogonal transformation and isotropic scaling.
  Measures: geometric similarity of hidden representations at the
  continuous level. Does not directly measure discrete output
  agreement.
- **Morcos, Raghu, Bengio 2018, "Insights on representational
  similarity via SVCCA / PWCCA"** — canonical correlation variants.
  Same limitation: continuous, basis-sensitive.
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
  latent space communication"** (ICLR 2023 notable top 5%,
  arXiv:2209.15430). Verified by fetching the abstract: representation
  is pairwise *cosine similarity to a fixed set of anchors* —
  **continuous**, not discrete. Invariance property: quasi-isometric
  (angles between encodings preserved across training runs). Enables
  zero-shot model stitching across CNNs, GCNs, transformers on
  images/text/graphs. **Distinction from ours**: Moschella encodes
  sample *relative* to anchors in continuous space; we encode sample
  into a *discrete code* via learned codebook. Our invariance is label
  permutation (relabelling the 64 codes); theirs is latent isometry
  (rotating the continuous space). Same philosophical aim — enabling
  cross-architecture communication without shared training — two
  different mathematical routes.

- **"Are neural network representations universal or idiosyncratic?"
  (Nature Machine Intelligence 2025, s42256-025-01139-y).** Very
  recent — explicitly frames the live debate between the universal
  representation hypothesis (all networks converge on a common
  computational substrate) and the idiosyncratic view (networks
  diverge by architecture, objective, learning rule). Our 0.91–0.96
  MI/H and 0.99 round-trip belong in this debate as evidence for a
  soft universal hypothesis at the *discrete-output* level.
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
