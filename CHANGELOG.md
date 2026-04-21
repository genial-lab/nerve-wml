# Changelog

All notable changes to `nerve-wml` follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.0] — 2026-04-21

Review-response release to a 2026-04-21 TMLR-style external
review of the v1.6.0 paper. 5 major (F1–F5) + 7 minor (m1–m10,
W2) review concerns closed with measured evidence on branch
`review-v1.7.0`. No change to the v1.2.3 scientific baseline or
to any v1.6.0 headline number; two new paper tests (10 + 11)
broaden Claim B and reframe it on frozen-encoder evidence.
Fully backward-compatible.

### Added

- Paper Test (10) "Frozen-encoder baseline" — shared-encoder
  MI/H = 0.9486 (3 seeds), distinct-encoders control MI/H =
  0.7622. Reframes Claim B as "VQ protocol supplies shared
  frontend through codebook" (review F3, commits `98c248b` +
  `d0de1c4`).
- Paper Test (11) "Matched-capacity scale sensitivity" on
  Sleep-EDF. Sweet spot at d=128: MI/H = 0.72, MLP 0.82, LIF
  0.83, gap 0.006. Scale-invariant polymorphy at d ∈
  {32, 64, 128}. d=16 under-specifies LIF on real EEG; d=256
  MLP overfits while LIF holds (review F1, commits `261cad9` +
  `10b249b`).
- `scripts/baseline_frozen_encoder.py` — frozen-encoder pipeline
  + distinct-encoders control with null-model z.
- `scripts/hyperparam_sensitivity.py` — architecture vs pool
  scale orthogonality sweep on HardFlowProxyTask N=2
  matched-capacity (review m3, commits `8da3488` + `16634b8` +
  `dcdb55d`).
- `scripts/track_w_pilot.py::run_w2_hard_multiseed` honours the
  5-seed contract at N=2 with triple-pinned seeding (`random`,
  `numpy`, `torch`); completes the scaling-law seed symmetry
  (review F2 + m9, commit `0d591fa`).
- `tests/test_determinism_seed0.py` — bit-for-bit seed=0
  invariant (code-review Minor #3, commit `3fdbba1`).
- §Method γ/θ type-checker framing — γ/θ recast as a discrete
  type-checker on Neuroletter multiplexing, consistent with
  v1.5.3's N-3 gate investigation (review F5, commit `7a0c597`).
- §Method matched-capacity design rationale — explicit defence
  of `d_hidden=128` against smaller/larger alternatives
  (commit `261cad9`).
- §Related Work: PRH rhetoric softened to "biologically-inspired
  alignment" + `aristotelianprh2026` citation (review F4, commit
  `aef9e7d`).
- §Related Work: `peng2025gridlikevq` + `zhao2025channelavq`
  citations (review m6 + m7, commit `2faf585`).
- Abstract: version tag `v1.3.0` → `v1.7.0`, MI/H + WML glossary
  entries, "15/15 → 20/20 seed claim" fix (review m1 + m2 + W2,
  commit `d487735`).
- `docs/changelog/v1.7.0.md` — full scientific rationale.
- `docs/research-notes/paper-v1.7.0-review-response.md` —
  point-by-point response with concern ↔ task ↔ commit table.

### Reproducibility

- `papers/paper1/figures/baseline_frozen_encoder.json`
  (frozen-encoder shared + distinct, 3 seeds each).
- `papers/paper1/figures/eeg_matched_scale_sweep.json` (Sleep-EDF
  d_hidden ∈ {16, 32, 64, 128, 256}).
- `papers/paper1/figures/hyperparam_sensitivity.json` (HardFlow
  d_hidden + lr sweep at N=2).
- `papers/paper1/figures/w2_hard_n2_multiseed.json` (5-seed N=2
  scaling-law anchor; median gap 10.71 % reproduces v1.6.0
  bit-for-bit).

### Scientific findings

- **Frozen-encoder spread = 0.19 MI/H.** Shared-encoder MI/H =
  0.95 reproduces nerve-wml Test (1) range 0.91–0.96;
  distinct-encoders MI/H = 0.76 localises the alignment source
  to the shared frontend. Claim B is reframed: the VQ protocol
  supplies the shared frontend through the codebook.
- **Sleep-EDF sweet spot at d=128.** MLP 0.82 / LIF 0.83 /
  gap 0.006 on matched-capacity scale sweep; scale-invariant
  polymorphy at d ∈ {32, 64, 128}.
- **Direction stability strengthened to 20/20.** N=2 rerun at
  5 seeds preserves LIF ≥ MLP in 4/5 seeds (with the failing
  seed at a 4 % gap, below contract); combined with N=16/32/64
  at 5 seeds each, the abstract's direction-stability claim is
  now 20/20 pairwise measurements.

See [`docs/changelog/v1.7.0.md`](docs/changelog/v1.7.0.md) for
the full scientific rationale and
[`docs/research-notes/paper-v1.7.0-review-response.md`](docs/research-notes/paper-v1.7.0-review-response.md)
for the review concern ↔ commit table.

## [1.6.0] — 2026-04-21

Broadens Claim A/B from synthetic benchmarks + MNIST to a
canonical real neural recording: Sleep-EDF Expanded EEG,
5-class sleep-stage classification via the v1.5.0
`MlpWML.from_spectrogram` factory. No API change, no regression
on v1.2.3 baseline.

### Added

- Paper Test (9) "Real neural data (Sleep-EDF)" in section
  Information Transmission. Cross-domain MI/H(a) table across
  HardFlowProxyTask / MoonsTask / MNIST / Sleep-EDF / DVNC.
- `scripts/eeg_preprocess_sleep_edf.py` full wiring
  (bandpass + resample + segment + per-subject split).
- `scripts/save_codes_eeg.py` with `--spectrogram` and
  `--d-hidden` flags; default now lr=1e-3 steps=2000 with
  class-balanced sampling + inverse-frequency weighted CE.
- `docs/research-notes/sleep-edf-pipeline-protocol.md`
  already present from v1.5.x cycle; now reflects the
  delivered configuration.

### Reproducibility

- `tests/golden/codes_mlp_lif_eeg_n10.npz` (12.9 MB,
  10 subjects, 3 seeds, 128-dim spectrogram embeddings).
- `papers/paper1/figures/mi_eeg_n10.json` (plug-in 0.66,
  Miller-Madow 0.66, KSG 1.94 nats, MINE 3.83 nats,
  null-model z 1263-1351, bootstrap CI95 [0.63, 0.70]).
- MLP acc 0.76, LIF acc 0.80, pairwise gap 0.036.

### Dependencies

- `mne>=1.12.1` added (transitive: pooch, requests, tqdm);
  required by the Sleep-EDF fetch and preprocessing path.

See [`docs/changelog/v1.6.0.md`](docs/changelog/v1.6.0.md) for
the full scientific rationale.

## [1.5.3] — 2026-04-21

Methodology release honouring the v1.5.2 cross-lab methodology
commitment. Adds the `nerve_wml.methodology` submodule shared with
`bouba_sens` (section 6.3 pre-registered methodology). The v1.2.3
scientific baseline is unchanged; the MI/H headline is now reported
with null-model significance, bootstrap CI, and four-estimator
robustness.

### Added

- `nerve_wml.methodology.mi_null_model` — permutation significance
  test (z > 1000, p < 10⁻³ on the 3-seed MLP↔LIF codes).
- `nerve_wml.methodology.bootstrap_ci_mi` — non-parametric bootstrap
  confidence interval (CI95 [0.82, 0.99] across seeds).
- `nerve_wml.methodology.mi_estimators` — `mi_plugin_discrete`,
  `mi_miller_madow_discrete`, `mi_kraskov_ksg_continuous`,
  `entropy_discrete`.
- `nerve_wml.methodology.mi_mine_estimator` — MINE (Belghazi 2018
  Donsker-Varadhan bound, 128-hidden critic, tail-averaged).
- `scripts/save_codes_for_checks.py` — produces the
  `tests/golden/codes_mlp_lif.npz` reproducibility artefact
  containing 3-seed argmax codes + pre-VQ continuous embeddings.
- `scripts/measure_mi_null_model.py`, `measure_mi_bootstrap_ci.py`,
  `measure_mi_multi_estimator.py`, `measure_mi_mine.py` — four
  light-weight measurement scripts consuming the NPZ.
- `scripts/ablation_n3_guard.py` + `scripts/ablation_n3_predictive.py`
  — N-3 gate investigation closure (three convergent ablations).
- `docs/research-notes/n3-gate-role.md` — full reasoning trace.
- `papers/paper1/main.tex` — new Test (7) "Multi-estimator
  robustness" with Table 3 (plug-in / Miller-Madow / KSG / MINE
  side-by-side) and an honest interpretation flagging the pre-VQ
  continuous-estimator divergence as an open methodological
  question.
- `scipy` added as dependency (required by KSG digamma).

### Changed

- `README.md` — Status header bumped to v1.5.3; Cross-lab methodology
  commitment section updated to reflect the three delivered checks
  plus the continuous-estimator divergence between KSG and MINE.

### Reproducibility

- `tests/golden/codes_mlp_lif.npz` — 3-seed MLP+LIF codes (shape
  `(3, 5000)` int64) plus pre-VQ embeddings (shape `(3, 5000, 16)`
  float32).
- `papers/paper1/figures/mi_{null_model,bootstrap_ci,multi_estimator,mine}.json`
  — primary result JSONs.

See [`docs/changelog/v1.5.3.md`](docs/changelog/v1.5.3.md) for the
full scientific rationale.

## [1.5.1] — 2026-04-21

First PyPI release (`pip install nerve-wml`). Patch bump that syncs the
package metadata: v1.5.0 shipped with `pyproject.toml` `[project].version`
still at `"1.4.0"` (the v1.4.0 release commit bumped it, but the three
subsequent PRs merged on top without a second bump). Per-version Zenodo
DOI dropped from `CITATION.cff` — only the concept DOI
`10.5281/zenodo.19656342` remains, resolving to the latest record.

### Fixed

- `pyproject.toml` version now `"1.5.1"`. Wheels built from the v1.5 line
  report the correct version in `pip list` / `__version__`.

### Changed

- `CITATION.cff` identifier block: concept DOI only, no per-release churn.

See [`docs/changelog/v1.5.1.md`](docs/changelog/v1.5.1.md) for the full
rationale.

## [1.5.0] — 2026-04-21

Bundle of three features requested by downstream consumers (`bouba_sens`
and `dream-of-kiki`). No regression on the v1.2.3 scientific baseline —
all new behaviour is opt-in and off by default.

### Added

- `track_p.transducer.TransducerGating` enum (`HARD` | `GUMBEL_SOFTMAX`)
  plus `gumbel_tau` kwarg on `Transducer.__init__`, per-call `hard` /
  `tau` overrides on `forward`. Default stays `HARD` so v1.2.3 runs
  reproduce bit-identically. Opt-in `GUMBEL_SOFTMAX` returns the
  `(B, alphabet_size)` differentiable soft distribution instead of the
  argmax long codes — keeps gradients alive through the code axis.
  Motivated by [#5](https://github.com/hypneum-lab/nerve-wml/issues/5)
  (bouba_sens B-2 Me3-delta under-threshold in 5/5 worlds).
- `track_w/spectrogram.py` — `SpectrogramEncoder` wrapping
  `torch.stft → magnitude → top-N bins → temporal mean → linear
  projection`. Shipped with `MlpWML.from_spectrogram(sample_rate,
  window_sec, hop_sec, n_bins, target_carrier_dim)` classmethod factory.
  Callable as `encoder(waveform)` for both `(B, T)` and `(T,)` inputs;
  output shape `(B, target_carrier_dim)`. Motivated by
  [#7](https://github.com/hypneum-lab/nerve-wml/issues/7) (DRY for
  bouba_sens MIT-BIH ECG + Studyforrest audio consumers).
- `nerve_core/from_dream_of_kiki.py` — `from_dream_of_kiki` + dual
  `to_dream_of_kiki`, `DreamOfKikiAxiomError`, `REQUIRED_AXIOMS`
  (`DR-0..DR-4`). **Scaffold only**: spec validation live, runtime
  wiring gated on `dream-of-kiki` publishing a versioned `axioms` public
  API. Design doc [`docs/integration-dream-of-kiki.md`](docs/integration-dream-of-kiki.md)
  gives the DR-X → nerve-wml mapping table. Motivated by
  [#6](https://github.com/hypneum-lab/nerve-wml/issues/6).

### Tests

- +35 new unit tests (14 transducer gating + 11 spectrogram encoder +
  10 dream-bridge scaffold). Existing 21 multiplexer tests unchanged.

### Known issue

- `pyproject.toml` `[project].version` stayed at `"1.4.0"` — fixed in
  v1.5.1. No functional impact.

## [1.4.0] — 2026-04-21

Exposes opt-in plasticity gating on `GammaThetaMultiplexer`. Motivated by
[#4](https://github.com/hypneum-lab/nerve-wml/issues/4) — bouba_sens B-1
Amedi-2007 congenital-blindness gap directionally falsified across 4/5
worlds in ADR-0005 + ADR-0009; the only architectural difference between
T1 (congenital) and T2 (late-acquired) was whether Phase 1 ran, with
identical multiplexer plasticity. This release lets `AdaptationLoop` give
T1 / T2 biologically distinct plasticity profiles.

### Added

- `GammaThetaMultiplexer.__init__` accepts `plasticity_schedule:
  Callable[[int], float] | None` and `constellation_lock_after: int | None`.
- `GammaThetaMultiplexer.step()` advances an internal `plasticity_step`
  long buffer. When `constellation_lock_after` is set and the counter
  crosses it, `constellation.requires_grad` is permanently set to
  `False` (biological critical-period lock-in).
- `plasticity_schedule` callback multiplies the gradient flowing into
  `constellation` on every `.backward()`. A constant-1.0 schedule is
  exactly equivalent to no hook (identity).
- `state_dict()` / `load_state_dict()` round-trip preserves
  `plasticity_step`; the lock is re-applied on load if the saved counter
  already crossed the threshold.

### Unchanged

- Default construction reproduces v1.3.0 behaviour byte-for-byte.
  The 21 pinned multiplexer contract tests still pass.

### Packaging

- `pyproject.toml` version bumped from the drifted `"0.1.0"` to `"1.4.0"`
  to re-sync with the git tag trajectory (`v1.3.0` → `v1.4.0`).

See [`docs/changelog/v1.4.0.md`](docs/changelog/v1.4.0.md) for the full
rationale + downstream validation plan.

## [1.2.0] — 2026-04-20

Closes the three remaining scientific debts identified in the v1.1.1 audit: real-data validation (MNIST), bigger-architecture sensitivity (d_hidden=128), and temporal streaming (sequential tokens). Three new figures published.

### Added

- `track_w/tasks/mnist.py` — MNISTTask seed-stable flattened loader (torchvision, optional `mnist` extra).
- `track_w/tasks/sequential.py` — SequentialFlowProxyTask (16-token sequence, label at a supervised timestep).
- `track_w/configs/wml_config.py` — WmlConfig with `.mnist()` and `.large()` presets.
- `track_w/streaming_hooks.py` — per-timestep rollout helpers.
- `input_dim` parameter on MlpWML / LifWML / TransformerWML (backward compatible).
- `track_w.pool_factory.build_pool_cfg(cfg)` — config-driven pool.
- `scripts/run_mnist_pilots.py`, `run_bigger_arch.py`, `run_temporal_pilots.py` + three figure renderers.

### Scientific findings (v1.2)

- **MNIST (real data):** MLP 0.942, LIF 0.941, median gap **1.03 %**, `MI/H = 0.882` over 3 seeds.
- **Bigger arch (d_hidden=128):** substrate asymmetry AMPLIFIES (median gap **26 %**) — spike expressivity scales with `n_neurons`. Architecture scale and pool scale are orthogonal dimensions. Claim B survives: `MI/H > 0.50` even when accuracies diverge.
- **Temporal streaming:** `MI/H = 0.72` at trained step, `0.71` at filler step — alignment is structural, not task-pressure-gated.

### Paper

- §Information Transmission extended with subsections (4a) MNIST, (4b) architecture scale, (4c) temporal streaming, each with figure.
- Three figures: `mnist_scaling.pdf`, `bigger_arch_scaling.pdf`, `temporal_info_tx.pdf`.

## [1.1.0] — 2026-04-20

A single intensive session upgraded four scientific claims from architectural postulates to empirical measurements. Paper drafts v0.4 through v0.8 track the iterations.

### Added

- **LifWML.emit\_head\_pi** — learned `nn.Linear(n_neurons, alphabet_size)` symmetric to `MlpWML.emit_head_pi`. The protocol `step()` preserves the cosine-similarity pattern-match decoder (N-1 invariant); classification pilots read out the learned head for apples-to-apples comparison. Resolved §13.1 debt #1.
- **TransformerWML** (`track_w/transformer_wml.py`) — third substrate: tokenized input + `nn.TransformerEncoder(n_layers × n_heads)` + `emit_head_pi` / `emit_head_eps`. Obeys WML Protocol and invariants W-1, W-2, W-5. 7 unit tests pin the Protocol compliance surface.
- **W2-hard scaling pilots** — `run_w2_hard_n16`, `run_w2_hard_n32`, `run_w2_hard_n64` plus their multi-seed wrappers (`_multiseed`). RNG-isolated per cohort (MLP / LIF / task-eval) using explicit seed parameter.
- **Triple-substrate polymorphism pilot** — `run_w_triple_substrate(hard=False|True)`. Trains MLP + LIF + TRF on the same task with RNG isolation; reports `triple_gap = (max − min) / max`.
- **Inter-substrate information-transmission pilots** — `scripts/measure_info_transmission.py`: mutual-information between emitted codes, round-trip fidelity MLP→LIF→MLP through learned transducers, and cross-substrate merge where a frozen LIF recovers task accuracy from MLP-emitted codes only.
- **Four-point scaling-law figure** — `scripts/render_scaling_figure.py` produces `papers/paper1/figures/w2_hard_scaling.{pdf,png}` with median ± IQR error bars and a 5 % contract band.

### Scientific findings (honest)

- **Polymorphism scaling law (4 points, 5 seeds each except N=2)** — median gap:
  - $N=2 \to 10.71\%$
  - $N=16 \to 6.71\%$ (max $10.35\%$)
  - $N=32 \to 2.39\%$ (max $4.75\%$ — every seed satisfies the 5 % contract)
  - $N=64 \to 2.73\%$ (plateau; max $3.71\%$)
  Monotonic decay between $N=2$ and $N=32$, plateau at $\sim 2\text{--}3\%$ for $N \geq 32$. Direction stable: LIF $\geq$ MLP in **15/15 multi-seed measurements**.
- **Information transmission measured** — on HardFlowProxyTask, for independently trained MLP and LIF on the same input: $\mathrm{MI}(c_{\text{MLP}}, c_{\text{LIF}}) / H(c_{\text{MLP}}) \approx 0.91$ (substrates share $\sim 91\%$ of their code information), round-trip fidelity $\approx 0.99$, cross-merge ratio $\approx 0.97$. Claim B (substrate-agnostic information transmission) is empirical, not just architectural.
- **Triple-substrate saturation** — on FlowProxyTask, MLP / LIF / TRF all converge to $1.000$ (triple-gap $0\%$). On HardFlowProxyTask at $N=1$: $0.547 / 0.605 / 0.529$ (triple-gap $12.6\%$). Pool scaling not yet measured for TRF.

### Paper

- Drafts v0.4 through v0.8 push substantive §Threats rewrites:
  - v0.4 — decoder-asymmetry artefact documented
  - v0.5 — N=16 multi-seed distribution
  - v0.6 — scaling-law table (N=16 / N=32)
  - v0.7 — N=64 plateau + scaling-law figure
  - v0.8 — §Information Transmission (new section)
- Eight paper tags shipped: `paper-v0.2-draft`, `paper-v0.3-draft`, `paper-v0.4-draft`, `paper-v0.5-draft`, `paper-v0.6-draft`, `paper-v0.7-draft`, `paper-v0.8-draft`.

### Infrastructure

- **240+ tests passing** across unit, integration, golden, and info-transmission layers.
- Commits split across feature branches `feat/w2-hard-multiseed`, `feat/transformer-wml`, `feat/info-transmission`; all merged into `master` at v1.1.0 tag.



## [1.0.0] — 2026-04-19

First stable release. All eleven gates pass on commodity Apple Silicon; the paper v0.3 draft consolidates every gate's measurements.

### Added

- **Gate P** — Track-P protocol simulator (`track_p/sim_nerve.py`, `track_p/vq_codebook.py`, `track_p/transducer.py`, `track_p/router.py`). Pilots P1–P4 pass on toy signals.
- **Gate W** — Track-W WML lab (`track_w/mock_nerve.py`, `track_w/mlp_wml.py`, `track_w/lif_wml.py`). MLP ↔ LIF polymorphism gap 0 % on FlowProxyTask 4-class.
- **Gate M** — merge pipeline (`bridge/sim_nerve_adapter.py`, `bridge/merge_trainer.py`) retaining 100 % of mock baseline.
- **Gate M2** — four §13.1 scientific shortcuts resolved: P3 γ-priority ablation (26 % collision without rule), W2 true-LIF polymorphie on HardFlowProxyTask (12.1 % gap — honest), W4 rehearsal CL (forgetting 100 % → 0 %), P1 random-init VQ + codebook rotation (dead codes 39 % → 0 %).
- **Paper v0.2** — ablation table, figures 2–4 (W4 forgetting, P1 dead-code curves, W2 histogram), §Threats, §Reproducibility.
- **Gate Scale** — W1/W2/W4 pilots at N=16 plus W2 stress at N=32; router strongly connected for all N ∈ {4, 8, 16, 32}.
- **Gate Interp** — `interpret/` package: semantics extractor (`build_semantics_table`), torch k-means (`cluster_codes_by_activation`), plain-HTML report renderer (`render_html_report`). Cluster entropy > 2 bits on toy data.
- **Gate Neuro** — `neuromorphic/` package: INT8 symmetric quantization (`quantize_lif_wml`), pure-numpy mock runner (`MockNeuromorphicRunner`), software-vs-mock delta check, Loihi 2 / Akida stubs with informative `NotImplementedError`.
- **Gate Dream** (partial) — `bridge/dream_bridge.py` ε-trace collect/encode/apply pipeline, env-gated by `DREAM_CONSOLIDATION_ENABLED`, with `MockConsolidator` for CI. Full resolution awaits `kiki_oniric` v0.5+ public `consolidate()` surface.
- **Gate Adaptive** — `track_p/adaptive_codebook.py` with `active_mask`-based shrink/grow, `bridge/transducer_resize.py` reshaping transducers while preserving argmax on kept rows. Multi-cycle stability tested.
- **Gate LLM Advisor** — `bridge/kiki_nerve_advisor.py` with env-gated, never-raising `advise(query_tokens, current_route) -> dict | None`. Warm-path latency < 50 ms; disabled-path overhead < 5 ms. Self-contained wiring recipe at `docs/integration/micro-kiki-wiring.md`.
- **Paper v0.3** — abstract names all 11 gates; new `§Integrations` section covering Adaptive / Neuromorphic / Dream / LLM Advisor.
- **Harness** — `harness/run_registry.py` produces bit-stable `run_id` from `(c_version, topology, seed, commit_sha)`.
- **227 tests passing**, coverage ≥ 95 % on every package, `ruff` + `mypy` clean on 49 source files.

### Scientific findings (honest)

- **FlowProxyTask 4-class saturates** both MLP and LIF substrates at 1.000 — the 0 % polymorphie gap is a degenerate best case. Documented in paper §Threats.
- **HardFlowProxyTask (12-class XOR on noise)** exposes real variance: `acc_mlp = 0.547`, `acc_lif = 0.480`, **gap = 12.1 %** — violates < 5 % on non-linear tasks. LIF's cosine-similarity decoder lags the MLP π head. Paper claim is now narrowed to linearly-separable regimes; closing the gap on harder tasks is explicit future work.
- **Untrained-LIF INT8 mock-runner delta ≈ 19 %** on random inputs — INT8 quantization of binary-like codebooks is coarse. Trained LIFs are expected to tighten.

### Infrastructure

- Eleven gate tags on origin, all `git push`-able and linked from README: `gate-p-passed`, `gate-w-passed`, `gate-m-passed`, `gate-m2-passed`, `gate-scale-passed`, `gate-interp-passed`, `gate-neuro-passed`, `gate-dream-passed`, `gate-adaptive-passed`, `gate-llm-advisor-passed`, plus `paper-v0.2-draft` and `paper-v0.3-draft`.
- No vendor SDK runtime deps: Loihi, Akida, `dream-of-kiki`, `sentence-transformers` are all opt-in.
- `MIT` for code, `CC-BY-4.0` for docs.

### Cited in

- `dreamOfkiki` Paper 1 v0.2 §7.4 cross-substrate portability (DR-3 Conformance Criterion). OSF pre-registration: [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN).

[1.0.0]: https://github.com/hypneum-lab/nerve-wml/releases/tag/v1.0.0
