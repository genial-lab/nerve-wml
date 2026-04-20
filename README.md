# nerve-wml

[![DOI](https://zenodo.org/badge/DOI/pending.svg)](https://zenodo.org/record/pending)
[![OSF](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2FQ6JYN-lightgrey)](https://doi.org/10.17605/OSF.IO/Q6JYN)

**Substrate-agnostic nerve protocol for inter-module communication in hybrid neural systems.**

Citation : each release is archived on Zenodo (DOI per version) and linked to the parent programme's OSF pre-registration ([10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN)). Replace the pending DOI badge with the actual one after the first Zenodo auto-mint (v1.1.2 and above).

Research engine that validates a discrete-code communication layer between heterogeneous neural modules (World Model Languages, or WMLs). Modules exchange **neuroletters** over a sparse learned topology, multiplexed on gamma/theta rhythms, and converted between local codebooks by per-edge transducers. The paper draft is at [`papers/paper1/main.tex`](papers/paper1/main.tex); the full spec is at [`docs/superpowers/specs/2026-04-18-nerve-wml-design.md`](docs/superpowers/specs/2026-04-18-nerve-wml-design.md).

## Status — v1.2.3 (2026-04-20)

The project is now empirically defensible across three experimental axes: real data, architecture scale, and temporal streaming. Two claims are quantified:

**Claim A — Substrate-agnostic polymorphism (task competence converges).**
Three structurally distinct substrates (stateless MLP, spiking LIF with surrogate-gradient, attention-based Transformer) reach comparable accuracies via the shared Nerve Protocol.

**Claim B — Substrate-agnostic information transmission (codes align).**
Independent substrates share 91–96 % of their emitted code information; a frozen LIF can recover a trained MLP's task competence via a learned linear transducer.

### Headline measurements

| Axis | Finding | Reference |
|---|---|---|
| **Pool scaling law** (MLP ↔ LIF, HardFlow) | $N=2 \to 10.71\%$, $N=16 \to 6.71\%$, $N=32 \to 2.39\%$, $N=64 \to 2.73\%$ plateau. 5 % contract holds distributionally at $N \geq 32$. | `figures/w2_hard_scaling.pdf` |
| **Triple-substrate pool** (MLP + LIF + TRF) | $N=15 \to 8.16\%$, $N=30 \to 5.86\%$, $N=60 \to 4.33\%$ | v1.1.4 |
| **Mutual information** (codes MLP ↔ LIF) | $\mathrm{MI}/H = 0.91$ at $N=1$ (5 seeds), **0.96** at $N=16$ pool (192 cross-pairs) | `figures/info_transmission.pdf` |
| **Round-trip fidelity** (MLP → LIF → MLP) | **0.99** mean (3 seeds) | v0.8 |
| **Cross-substrate merge** (LIF fed by MLP codes only) | **0.97** mean (3 seeds) | v0.8 |
| **MNIST real data** | MLP 0.942, LIF 0.941, gap **1.03 %**, MI/H **0.882** | `figures/mnist_scaling.pdf` |
| **MoonsTask** (2nd distribution) | MI/H = **0.74** (3 seeds) | v1.1.4 |
| **Architecture scale** ($d_\text{hidden}=128$) | Gap AMPLIFIES to 26 % on XOR (arch vs pool scale are orthogonal); Claim B survives | `figures/bigger_arch_scaling.pdf` |
| **Temporal streaming** (16-token sequence) | MI/H = **0.72** at trained step, **0.71** at filler step — structural alignment | `figures/temporal_info_tx.pdf` |
| **Direction stability** (LIF ≥ MLP on hard task) | **15/15** pairwise seeds + 5/5 triple-substrate | — |

LIF's spike dynamics give it a substrate-intrinsic $\sim 2$–$3\%$ expressivity edge on XOR-style boundaries (plateau floor). Pool averaging compresses this, architecture width amplifies it.

### Seven concrete findings

1. **The original 12.1 % gap was a decoder asymmetry bug, not a substrate limit.** LIF had a fixed cosine decoder, MLP had a learned head; symmetrizing flipped the sign (LIF now leads).
2. **Single-seed measurements lie.** Multi-seed revealed the N=16 median is 6.7 %, not the lucky 1.68 %.
3. **Scaling law is real and monotonic.** Four-point decay $10.7\% \to 6.7\% \to 2.4\% \to 2.7\%$ plateau.
4. **Claim B is empirical, not architectural.** MI 0.91–0.96, round-trip 0.99, cross-merge 0.97.
5. **Substrate-direction is stable in 15/15 seeds.** LIF's spike edge is a real property, not noise.
6. **Architecture scale and pool scale are orthogonal.** Pool compresses the gap; arch width amplifies it.
7. **Code alignment is structural, not task-gated.** MI at filler timesteps $\approx$ MI at trained timesteps (0.71 vs 0.72).

### Methodological findings (v1.2.1–v1.2.3)

- **MI/H vs CKA on the same argmax codes** (v1.2.1). Mean 0.953 (MI/H) vs 0.910 (CKA argmax one-hot) over 3 seeds. The 4.3 pp gap tracks soft many-to-one code mappings that kernel-alignment metrics miss. MI/H is not CKA renamed — it is the discrete-protocol cousin with measurably different semantics. See `scripts/measure_cka_vs_mi.py` and `docs/positioning.md`.
- **Related Work verified** (v1.2.2). Paper §Related Work cites Kornblith 2019 CKA, Morcos 2018 PWCCA, Moschella 2022 relative representations (ICLR 2023), Saxe 2024 universality, and Hinton 2015 KD — all verified via WebFetch, provenance table in `docs/positioning.md`.
- **KD match-compute ablation honest verdict** (v1.2.3). At matched compute on HardFlowProxyTask (3 seeds), cross-merge (0.508) ≈ KD-through-transducer (0.520) within noise. Vanilla Hinton KD (0.534) is best because the student can re-train its core. Cross-merge's contribution is **methodological, not performance-based**: it isolates protocol channel capacity from student learning capacity by freezing both substrates and supervising with ground-truth labels only. See `scripts/measure_kd_ablation.py`.

### What the paper genuinely claims vs not

Three findings probably novel: (1) the four-point scaling law with plateau at $\sim 2\text{–}3\%$ substrate-intrinsic floor, (2) reproducible $\sim 2\text{–}3\%$ LIF spike-expressivity edge over matched-capacity MLP on XOR-on-noise (15/15 seeds), (3) orthogonality of pool-scale (compresses gap) and architecture-scale (amplifies gap).

The paper explicitly does **not** claim: a new learning algorithm, superiority over knowledge distillation on task accuracy, or universal representations — that debate is addressed by Saxe 2024 and the Nature MI 2025 editorial (s42256-025-01139-y) cited in `docs/positioning.md`.

## Status — 11 gates

| Tag | What it proves |
|---|---|
| [`gate-p-passed`](../../releases/tag/gate-p-passed) | Track-P protocol simulator correct on toy signals |
| [`gate-w-passed`](../../releases/tag/gate-w-passed) | `MlpWML` and `LifWML` interoperate with < 5 % gap through the same nerve (N=4) |
| [`gate-m-passed`](../../releases/tag/gate-m-passed) | Merge fine-tunes only transducers; retains ≥ 95 % of mock-baseline accuracy |
| [`gate-m2-passed`](../../releases/tag/gate-m2-passed) | Four scientific shortcuts from §13.1 resolved with honest measurements |
| [`gate-scale-passed`](../../releases/tag/gate-scale-passed) | Polymorphie + continual learning hold at N=16 pools; router stays connected to N=32 |
| [`gate-interp-passed`](../../releases/tag/gate-interp-passed) | Per-WML `code → concept` semantics table rendered as HTML |
| [`gate-neuro-passed`](../../releases/tag/gate-neuro-passed) | LifWML → INT8 artefact → pure-numpy mock runner (Loihi / Akida stubs documented) |
| [`gate-dream-passed`](../../releases/tag/gate-dream-passed) | ε-trace consolidation bridge to dream-of-kiki (schema v0; partial — awaits kiki_oniric v0.5+) |
| [`gate-adaptive-passed`](../../releases/tag/gate-adaptive-passed) | Per-WML alphabet shrinks/grows via `active_mask` + transducer resize |
| [`gate-llm-advisor-passed`](../../releases/tag/gate-llm-advisor-passed) | Env-gated, never-raising `NerveWmlAdvisor` for micro-kiki, < 50 ms warm latency |

Paper drafts: `paper-v0.2-draft` … `paper-v0.9-draft` track the iterations that produced the v1.2 claims above. Release tags `v1.0.0`, `v1.1.0` … `v1.1.4`, `v1.2.0` archive the code snapshots; see `CHANGELOG.md` for per-version findings.

## Install

```bash
uv sync --all-extras
```

Python 3.12+, macOS arm64 (MLX-friendly) or Linux x86_64. No vendor SDK deps are pulled by default (Loihi, Akida, dream-of-kiki, sentence-transformers are all optional integrations).

## Run the suite

```bash
uv run pytest -m "not slow"    # 220+ tests under 80 s on commodity M-series
uv run pytest                  # full suite incl. paper figure rendering
uv run pytest --cov=nerve_core --cov=track_p --cov=track_w --cov=bridge --cov=harness --cov=interpret --cov=neuromorphic
```

## Reproduce the gate numbers

```bash
uv run python scripts/track_p_pilot.py       # Gate P (+ Task 6 ablation)
uv run python scripts/track_w_pilot.py       # Gate W
uv run python scripts/track_w_pilot.py scale # Gate Scale (N=16, N=32)
uv run python scripts/merge_pilot.py         # Gate M
uv run python scripts/interpret_pilot.py     # Gate Interp (emits reports/interp/*.html)
uv run python scripts/adaptive_pilot.py      # Gate Adaptive
```

## Reproduce the v1.1 / v1.2 findings

```bash
# v1.1 scaling law + information transmission + triple substrate
uv run python scripts/render_scaling_figure.py      # 4-point pool scaling (N=2..64)
uv run python scripts/render_info_tx_figure.py      # MI + round-trip + cross-merge
uv run python scripts/measure_info_transmission.py  # full info-tx battery

# v1.2 real data + bigger arch + temporal
uv sync --extra mnist                               # pull torchvision
uv run python scripts/render_mnist_figure.py        # MNIST Claims A + B
uv run python scripts/render_bigger_arch_figure.py  # d=128 gap amplification
uv run python scripts/render_temporal_figure.py     # streaming MI per timestep
```

## Build the paper

```bash
uv run python scripts/render_paper_figures.py   # regenerate figures from frozen golden NPZs
cd papers/paper1 && tectonic main.tex           # or pdflatex, bibtex, pdflatex, pdflatex
```

## Integrations (env-gated, default off)

- **Dream consolidation**: `DREAM_CONSOLIDATION_ENABLED=1` + install `dream-of-kiki` locally → `bridge.dream_bridge.DreamBridge`.
- **LLM advisor (micro-kiki)**: `NERVE_WML_ENABLED=1` + `NERVE_WML_CHECKPOINT_PATH=/path/to/checkpoint` → `bridge.kiki_nerve_advisor.NerveWmlAdvisor`. Wiring recipe: [`docs/integration/micro-kiki-wiring.md`](docs/integration/micro-kiki-wiring.md).
- **Neuromorphic hardware**: install `lava-nc` or `akida` → wire in `neuromorphic.loihi_stub` / `neuromorphic.akida_stub`. Schema v0: [`docs/neuromorphic/deployment-guide.md`](docs/neuromorphic/deployment-guide.md).

## Cited in

- **dreamOfkiki — Paper 1 v0.2 (2026-04-19), §7.4 cross-substrate portability** — [github.com/hypneum-lab/dream-of-kiki](https://github.com/hypneum-lab/dream-of-kiki). The Gate W and Gate M measurements reported here (MlpWML / LifWML polymorphism on FlowProxyTask and HardFlowProxyTask) provide the empirical corroboration cited in Paper 1 as independent evidence of the substrate-agnosticism principle (DR-3 Conformance Criterion). OSF pre-registration: [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN).

## Program context

This repository is part of **hypneum-lab**, which develops executable formal frameworks for cognitive AI. The programmatic parent is `dreamOfkiki` (paper 1 formal framework, paper 2 empirical); `nerve-wml` is the reference implementation for the substrate-agnostic communication principle.

Sibling repositories:

- [dream-of-kiki](https://github.com/hypneum-lab/dream-of-kiki) — formal framework (axioms DR-0..DR-4, Conformance Criterion, Paper 1)
- [kiki-flow-research](https://github.com/hypneum-lab/kiki-flow-research) — Wasserstein-gradient-flow engine (upstream)
- [micro-kiki](https://github.com/hypneum-lab/micro-kiki) — 35 domain-expert MoE-LoRA deployable instance (advisor consumer)
- **nerve-wml** (this repo) — substrate-agnostic nerve protocol + cross-substrate polymorphism

## Repository layout

```
nerve_core/        Neuroletter, Nerve + WML Protocols, invariants (N-1..N-5, W-1..W-4)
track_p/           Track-P — SimNerve, VQCodebook, Transducer, SparseRouter, AdaptiveCodebook
track_w/           Track-W — MockNerve, MlpWML, LifWML, toy tasks, training loop, pool factory
bridge/            Merge, dream, LLM advisor — SimNerveAdapter, MergeTrainer, DreamBridge, NerveWmlAdvisor
harness/           R1 reproducibility — run_registry
interpret/         Gate Interp — code_semantics, clustering, HTML renderer
neuromorphic/      Gate Neuro — spike_encoder, INT8 export, mock_runner, vendor stubs
scripts/           All gate pilots + figure renderers + freeze_golden
tests/             Unit + integration + golden NPZ regressions
docs/              specs/, integration/, neuromorphic/, dream/, interpret/
papers/paper1/     LaTeX source + bib + Makefile (figures regenerated deterministically)
```

## License

MIT (code) + CC-BY-4.0 (docs).
