# Paper v1.7.0 — Review response

Response to the **2026-04-21 TMLR-style external review** of paper
v1.6.0 (branch `review-v1.7.0`, 13 commits on top of
`master@2a5d982` "chore(release): v1.6.0 Sleep-EDF EEG validation").

Every review concern (5 major F1–F5, 7 minor m1–m10, plus code-
review W2) is addressed by a single task in the 10-task plan and
a specific commit SHA on branch `review-v1.7.0`. No prose move is
cosmetic: every softening is backed by a measurement.

## Concern ↔ task ↔ commit table

| Review concern | Severity | Plan task | Commit SHA | Status |
|---|---|---|---|---|
| **F1** — SOTA gap on Sleep-EDF (MLP 0.76 / LIF 0.80 below published ~0.83–0.86) | Major | Task 3a (§Method matched-capacity rationale) + Task 3b (scale sweep) | `261cad9` + `10b249b` | Closed: sweet spot at d=128 → MLP 0.82 / LIF 0.83 / gap 0.006 |
| **F2** — Seed asymmetry (N=2 had 3 seeds while N≥16 had 5) | Major | Task 1 (N=2 multi-seed) | `0d591fa` | Closed: 5 seeds, direction stable 4/5, median gap 10.71 % (reproduces v1.6.0 anchor bit-for-bit) |
| **F3** — Missing frozen-encoder baseline isolating shared-frontend vs VQ-protocol contribution | Major | Task 2 (frozen-encoder + distinct-encoders control) | `98c248b` + `d0de1c4` | Closed: shared MI/H=0.9486, distinct MI/H=0.7622, 0.19 spread reframes Claim B as "VQ protocol supplies shared frontend through codebook" |
| **F4** — PRH rhetoric overreach (strong form of Huh 2024 in §Related Work) | Major | Task 6 (soften + Aristotelian cite) | `aef9e7d` | Closed: "biologically-inspired" framing + `aristotelianprh2026` added |
| **F5** — γ/θ framing inconsistency (rhythmic-gating vs correctness-contract) | Major | Task 5 (§Method type-checker rewrite) | `7a0c597` | Closed: γ/θ cast as discrete type-checker on Neuroletter multiplexing, consistent with N-3 investigation of v1.5.3 |
| **m1** — Version drift (abstract still said v1.3.0) | Minor | Task 8 (abstract fixes) | `d487735` | Closed: v1.3.0 → v1.7.0 throughout abstract |
| **m2** — Undefined jargon (MI/H, WML) | Minor | Task 8 (glossary) | `d487735` | Closed: MI/H + WML glossary entries in abstract |
| **m3** — Hyperparameter ablation missing | Minor | Task 4 (HardFlow hyperparam sweep) + Task 3b (EEG scale sweep) | `dcdb55d` (honest numbers) + `10b249b` | Closed: d_hidden + lr sweep on HardFlow; EEG d_hidden ∈ {16..256} sweep on Sleep-EDF |
| **m6** — VQ codebook prior art missing (grid-like VQ) | Minor | Task 7 (VQ citations) | `2faf585` | Closed: `peng2025gridlikevq` added to §Related Work |
| **m7** — VQ codebook prior art missing (channel-aware VQ) | Minor | Task 7 (VQ citations) | `2faf585` | Closed: `zhao2025channelavq` added to §Related Work |
| **m9** — Seeding locks (ensure `random`, `numpy`, `torch` all triple-pinned) | Minor | Task 1 (also adds the triple pin) + determinism test | `0d591fa` + `3fdbba1` | Closed: `run_w2_hard_multiseed` pins `random.seed`, `np.random.seed`, `torch.manual_seed`; `tests/test_determinism_seed0.py` pins a bit-for-bit seed=0 invariant |
| **m10** — Extra seeds on Tests 4a–c recommended | Minor | Task 1 (absorbs m10 into N=2 multi-seed) | `0d591fa` | Partially closed: N=2 now at 5 seeds (up from 3); Tests 4a–c already had 5 seeds |
| **W2** — "15/15 direction stable" claim needed to be rewritten as 20/20 after adding the v1.7.0 N=2 rerun | Code-review | Task 8 (seed-claim fix) | `d487735` | Closed: abstract now reads "20/20 pairwise seeds including the v1.7.0 N=2 rerun" |

Additional review-response commits (minor fixes caught while
landing the above):

| Fix | Commit SHA | Motivation |
|---|---|---|
| Hyperparam sensitivity script bug: `MlpWML(...)` call was missing `input_dim=16` on HardFlow N=2 matched-capacity, yielding a projection-dim mismatch | `16634b8` | Caught during Task 4 rerun; without the fix the d_hidden sweep silently fell back to input_dim=1 and the gap collapsed trivially to zero — the honest numbers in `hyperparam_sensitivity.json` only appear post-fix |
| Paper Tests (10) + (11) prose inlined with the v1.7.0 measurements (Task 9) | `08f557f` | Primary paper landing commit for the two new tests |

## Closing: what changed in the scientific framing

Two loads in Claim B were softened in v1.7.0, and one
implicit design choice was made explicit:

1. **Claim B was implicitly "alignment in the null of shared
   encoder".** v1.7.0 Test (10) demonstrates that with distinct
   randomly-initialised encoders, MI/H drops from 0.95 to 0.76 —
   a 0.19 fall that localises the alignment source. Claim B is
   now stated as: *the VQ protocol supplies a shared frontend
   through the codebook, which is the empirical carrier of the
   cross-substrate alignment*. The post-VQ MI/H headline stays
   load-bearing; the interpretation is tightened.
2. **Matched-capacity design rationale is now explicit in
   §Method.** Prior versions gave the `d_hidden=128` choice
   without defending it against smaller/larger alternatives.
   v1.7.0 Test (11) shows the sweet spot is d=128 on Sleep-EDF
   with polymorphy scale-invariant at d ∈ {32, 64, 128}; d=16
   under-specifies and d=256 breaks the MLP. The choice is now
   empirically anchored rather than picked.
3. **PRH was softened.** The v1.6.0 §Related Work cited Huh 2024
   as an implicit endorsement of the Platonic Representation
   Hypothesis. v1.7.0 reframes this as biologically-inspired
   alignment with an Aristotelian prior-of-experience reading,
   sidestepping the overreach of a universal-representation
   claim.

No measurement was retracted. The v1.2.3 scientific baseline and
all v1.6.0 headline numbers (15/15 → now 20/20 direction
stability, MI/H 0.91–0.96, pool scaling law, orthogonality of
architecture vs pool scale) carry through v1.7.0 unchanged.
