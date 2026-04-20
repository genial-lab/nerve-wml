# nerve-wml — Substrate-Agnostic Nerve Protocol for Inter-WML Communication

**Design spec — v0.1**

- **Date** : 2026-04-18
- **Author** : Clément Saillant (L'Electron Rare) and nerve-wml contributors
- **Status** : brainstorm complete, awaiting implementation plan
- **Source** : interactive brainstorm session 2026-04-18, artifacts in `.superpowers/brainstorm/72309-1776509772/`
- **License** : MIT (code) + CC-BY-4.0 (docs)

---

## 1. Summary

`nerve-wml` explores an inter-module communication protocol inspired by biological nerves rather than natural language. A **WML** (World Model Language) is a cluster of neurons — concretely either a small MLP or a LIF (leaky integrate-and-fire) neuron population. WMLs exchange **neuroletters** : codes drawn from a learned local vocabulary of 64 symbols, transported on bidirectional nerves that carry **predictions** (π, descending) and **prediction errors** (ε, ascending), multiplexed on gamma/theta rhythms.

The project validates two scientific claims :

1. **Substrate-agnosticism**. An identical nerve interface supports heterogeneous WML implementations (MLP and LIF populations interoperate without knowing each other's substrate). Target : performance gap < 5 % on a toy continual-learning task.
2. **Protocol in isolation**. The nerve protocol (alphabet + transducers + γ/θ multiplexing + sparse routing) can be validated independently of any WML, via information-theoretic tests (capacity, disambiguation, dead-code rate).

Two parallel tracks drive the work : **Track-P** (protocol simulator, no WMLs) and **Track-W** (WML lab against a mock nerve). They merge when both pass gates. This spec is the single source of truth for the design ; an implementation plan will follow in a separate document.

---

## 2. Design axes (consolidated from brainstorm)

| Axis | Decision |
|---|---|
| Signal carried on the nerve | Discrete **neuroletters** — learned alphabet, |Σ| = 64 codes, ~6 bits per letter |
| Topology | **Learned sparse graph** — K ≪ N channels per WML, discovered during training via top-K Gumbel routing |
| Directionality | Physically bidirectional / semantically asymmetric — **π (predictions, ↓)** and **ε (errors, ↑)** |
| Synchrony | **Rhythmic multiplexing γ / θ** — γ (40 Hz) carries predictions, θ (6 Hz) carries errors, no collision by phase |
| WML grain | **Heterogeneous pool** — mix of micro-MLPs and LIF neuron populations sharing the same nerve interface |
| Codebook scope | **Local per WML** — each WML owns a 64-code codebook ; learned transducers at nerve endpoints map between neighbors' codebooks |

---

## 3. Architecture overview

Two parallel tracks with a merge gate :

```
Track-P · Protocol Sim              Track-W · WML Lab
──────────────────────              ──────────────────
γ/θ oscillators                     4 WMLs (2 MLP + 2 LIF)
   │                                   │
   ▼                                   ▼
VQ-64 codec                         Mock nerve (same API)
   │                                   │
   ▼                                   ▼
π/ε role heads                      Toy continual task
   │                                   │
   ▼                                   ▼
Sparse router (K-k)                 Polymorphie gate
   │                                   │
GATE P                              GATE W
   │                                   │
   └─────────── shared API ────────────┘
                    │
                    ▼
          MERGE — replace mock with sim
                    │
                  GATE M
                    │
                Paper draft
```

- **Track-P** validates the protocol against toy signals (sinusoids, impulses). No WMLs. Output : info-theoretic benchmarks.
- **Track-W** validates 4 WMLs (2 MLP, 2 LIF) against a `MockNerve` on a toy continual-learning task. Output : polymorphie proof (gap < 5 %).
- **Merge** swaps `MockNerve` for `SimNerve`, fine-tunes transducers only, confirms end-to-end works within 95 % of mock performance.

---

## 4. Contracts (`nerve_core/`)

### 4.1 Message

```python
from dataclasses import dataclass
from enum import Enum

class Role(Enum):
    PREDICTION = 0   # π — descending, continuous
    ERROR      = 1   # ε — ascending, bursty

class Phase(Enum):
    GAMMA = 0        # fast, 40 Hz — carries π
    THETA = 1        # slow,  6 Hz — carries ε

@dataclass(frozen=True)
class Neuroletter:
    code:      int      # 0..63, on-wire shared integer index
    role:      Role
    phase:     Phase    # in strict mode, derived from role
    src:       int      # emitter WML id
    dst:       int      # receiver WML id
    timestamp: float    # simulation time in seconds
```

### 4.2 Nerve protocol (shared between Track-P and Track-W)

```python
class Nerve(Protocol):
    ALPHABET_SIZE: int   = 64
    GAMMA_HZ:      float = 40.0
    THETA_HZ:      float = 6.0

    def send(self, letter: Neuroletter) -> None: ...
    def listen(self, wml_id: int,
               role:  Role  | None = None,
               phase: Phase | None = None) -> list[Neuroletter]: ...
    def time(self) -> float: ...
    def tick(self, dt: float) -> None: ...
    def routing_weight(self, src: int, dst: int) -> float: ...
```

### 4.3 Nerve endpoint (transducer)

```python
@dataclass
class NerveEndpoint:
    """One transducer per (src, dst) nerve. Maps src local code → dst local code."""
    transducer: Tensor  # [64, 64]
                        # Gumbel-softmax during training, argmax at inference
                        # T[i, j] = P(dst receives code j | src emitted code i)
```

Parameter cost : 64² = 4096 per nerve. For 4 WMLs with K=3 fan-out → ~12 nerves × 4k = ~50k params total. Acceptable.

### 4.4 WML protocol (implemented by both MLP and LIF variants)

```python
class WML(Protocol):
    id:       int         # stable identifier
    codebook: Tensor      # [64, d] — local learned alphabet

    def step(self, nerve: Nerve, t: float) -> None:
        """One simulation tick :
           1. inbound = nerve.listen(self.id)
           2. internal compute (MLP forward OR LIF dynamics)
           3. emit predictions via nerve.send(...)
           4. if surprise > threshold, emit errors"""

    def parameters(self) -> Iterable[Tensor]: ...
```

### 4.5 Invariants

| ID | Invariant | Rationale |
|---|---|---|
| N-1 | `len(listen(wml_id)) == 0` is valid | Silence is legitimate (sparse, event-driven ε) |
| N-2 | `send()` is idempotent on `(src, dst, code, timestamp)` | Required for bit-stable golden tests |
| N-3 | `role == ERROR` ⟺ `phase == THETA` in strict mode | Bastos-Friston 2012 canonical form |
| N-4 | `routing_weight(i, j) ∈ {0, 1}` after gate-P pruning ; continuous Gumbel during training | Sparse topology discovery. Pruning triggers at end of P4, freezing top-K edges per WML. |
| N-5 | Each WML owns its codebook ; cross-codebook mapping via transducers | Expressivity vs shared-alphabet trade-off |
| W-1 | `step()` never mutates another WML | Isolation via Nerve only |
| W-2 | `parameters()` includes `codebook` and all internal weights | VQ commitment loss is backpropagated |
| W-3 | No access to `routing_weight` from inside `step()` | Router is global, owned by Track-P |
| W-4 | `id` is stable across a WML's lifetime | Nerves index by id |

---

## 5. WML implementations

### 5.1 `MlpWML`

```python
class MlpWML:
    id: int
    codebook: Tensor                    # [64, d_hidden=128] — local VQ-VAE
    core: nn.Module                     # 4 × Linear(128, 128) + ReLU
    emit_head_pi:  nn.Linear            # hidden → logits over 64 codes
    emit_head_eps: nn.Linear
    threshold_eps: float = 0.30         # below which no ε is emitted — calibrated via L_surprise

    def step(self, nerve: Nerve, t: float) -> None:
        inbound = nerve.listen(self.id)
        h_in = self._decode_incoming(inbound)       # VQ embed + weighted sum
        h    = self.core(h_in)

        # Prediction (steady-state, every γ tick)
        for dst in self._predicted_targets():
            code_pi = self.emit_head_pi(h).argmax(dim=-1)
            nerve.send(Neuroletter(code_pi, Role.PREDICTION, Phase.GAMMA,
                                   self.id, dst, t))

        # Error (bursty, only when surprise exceeds threshold)
        surprise = self._mismatch_vs_prediction(h_in, inbound)
        if surprise > self.threshold_eps:
            for dst in self._error_targets():
                code_eps = self.emit_head_eps(surprise).argmax(dim=-1)
                nerve.send(Neuroletter(code_eps, Role.ERROR, Phase.THETA,
                                       self.id, dst, t))
```

### 5.2 `LifWML`

> **v1.1 update (2026-04-20):** `LifWML` now owns a learned linear
> `emit_head_pi = nn.Linear(n_neurons, alphabet_size)` symmetric to
> `MlpWML.emit_head_pi`, used by the Track-W training pilots
> (`run_w2_hard*`). The protocol `step()` retains the cosine
> pattern-match decoder for the legitimate biological asymmetry
> (silence when no pattern match, invariant N-1) — this preserves
> the intentional asymmetry below. In classification benchmarks the
> learned head reads out the spike pattern; in the protocol itself
> pattern-matching still gates emission.

```python
class LifWML:
    id: int
    codebook: Tensor         # [64, n_neurons=100] — spike-pattern codebook
    n_neurons: int = 100
    v_mem:     Tensor        # [n_neurons], membrane potential (continuous state)
    v_thr:     float = 1.0
    tau_mem:   float = 20e-3
    decoder:   SpikeDecoder  # spike pattern → code index via matching
    emit_head_pi:  nn.Linear # [n_neurons → alphabet_size] — learned readout (v1.1)
    emit_head_eps: nn.Linear # symmetric ε readout (v1.1)

    def step(self, nerve: Nerve, t: float) -> None:
        inbound = nerve.listen(self.id)
        i_in    = self._codes_to_current(inbound)       # code → [n_neurons] current

        # Continuous LIF integration with surrogate gradient for backprop
        dt = 1e-3
        self.v_mem = self.v_mem + dt / self.tau_mem * (-self.v_mem + i_in)
        spikes     = (self.v_mem > self.v_thr).float()  # STE/surrogate gradient
        self.v_mem = self.v_mem * (1 - spikes)          # reset after firing

        # Intentional asymmetry vs MlpWML : LIF emits only when a spike pattern
        # clearly matches a codebook entry (confidence threshold in decoder).
        # This is biologically plausible (spikes are sparse) and acceptable
        # under invariant N-1 (silence is legitimate).
        code_pi = self.decoder.match_pattern(spikes, self.codebook)
        if code_pi is not None:
            for dst in self._predicted_targets():
                nerve.send(Neuroletter(code_pi, Role.PREDICTION, Phase.GAMMA,
                                       self.id, dst, t))

        # Error emission via a dedicated "mismatch neuron" accumulating inbound vs expected
        ...
```

---

### 5.3 `TransformerWML` (v1.1 — third substrate)

Added in v1.1.0 to upgrade the polymorphism claim from two to three
structurally distinct substrates. Obeys the WML Protocol (§4.4) and
invariants W-1, W-2, W-5.

```python
class TransformerWML:
    id: int
    codebook: Tensor          # [64, d_model=16] — shared with emit_head geometry
    d_model:  int = 16
    n_tokens: int = 4         # input is split into n_tokens chunks of d_model/n_tokens
    d_token:  int             # = d_model // n_tokens
    token_proj: nn.Linear     # [d_token → d_model], per-token projection
    encoder:    nn.TransformerEncoder  # n_layers × n_heads, batch_first=True
    emit_head_pi:  nn.Linear  # [d_model → alphabet_size]
    emit_head_eps: nn.Linear  # [d_model → alphabet_size]

    def core(self, x: Tensor) -> Tensor:
        """Tokenize [B, d_model] → attend → mean-pool → [B, d_model]."""
        B = x.shape[0]
        tokens = x.view(B, n_tokens, d_token)
        tokens = self.token_proj(tokens)
        return self.encoder(tokens).mean(dim=1)

    def step(self, nerve: Nerve, t: float) -> None:
        # Same structure as MlpWML.step — listen, core, emit π (γ), emit ε (θ)
        # when surprise = ||h − h_prior|| > threshold_eps.
        ...
```

Triple-substrate polymorphism on `FlowProxyTask` saturates all three
at 1.000 (triple-gap 0 %). On `HardFlowProxyTask` at N=1 (5 seeds),
median triple-gap = 5.59 %, max = 23.08 % — wider than the pairwise
MLP-LIF distribution because `triple_gap = (max − min) / max` is
worst-case sensitive. Pool-scale triple-substrate measurement is
deferred (§13.1 open item).

---

## 6. End-to-end data flow (one γ + θ cycle, 4 WMLs mixed)

Setup : `WML₁` (MLP, top) · `WML₂` (MLP, mid) · `WML₃` (LIF, mid) · `WML₄` (LIF, bottom). Hierarchical sparse topology. One full cycle = 25 ms γ burst (predictions ↓) + 170 ms θ burst (errors ↑ if surprise > threshold).

### γ phase (predictions descend)

- t=0 : `WML₁.core.step()` → `code_local=17` (e.g. « moving-object situation »).
- `send(Neuroletter(code=17, Role.π, Phase.γ, src=1, dst=2, t=0))`.
- Nerve 1→2 transducer : 17 → argmax 41.
- `WML₂.listen(2) → [41]` → `core.step()` → `code_local=23`, sent to WML₃.
- Nerve 2→3 : 23 → 9. `WML₃._codes_to_current` injects spike pattern → 4 neurons fire → decoder matches → `code_local=5`.
- Nerve 3→4 : 5 → 62. `WML₄` receives, integrates dynamics, spikes emerge at t=25 ms.

### θ phase (errors ascend, conditional)

- At t=30 ms, `WML₄` observes actual input : stationary object (not predicted).
- Surprise = ‖actual − predicted_from_WML₃‖ = 0.81 > 0.30 threshold. Emit ε.
- `send(Neuroletter(code=8, Role.ε, Phase.θ, src=4, dst=3, t=30ms))`.
- Nerve 4→3 (ε transducer, distinct from π) : 8 → 19.
- `WML₃` integrates ε, cascades surprise=0.44 → emit ε upstream.
- Eventually `WML₁` receives ε_local=2 and updates codebook₁ via commitment loss toward the new observation.

**Property** : each WML sees only its local codes (17, 41, 9, 5, 62 in WML₁-WML₂-WML₃-WML₃-WML₄ respectively). Transducers mediate ; γ/θ prevents phase collision ; no ε fires when surprise < threshold (natural sparsity).

---

## 7. Training strategy

### 7.1 Losses (per batch)

```python
L_total = L_task                        # toy continual-learning loss
        + λ_vq        * L_vq            # 0.25 × ‖sg(z) − e‖² + β × ‖z − sg(e)‖²
        + λ_trans     * L_entropy       # −Σ T log T per transducer (avoid collapse)
        + λ_route     * L_sparsity      # enforce K-active via Gumbel straight-through
        + λ_sep       * L_role_sep      # separate π vs ε distributions (KL ≥ margin)
        + λ_surprise  * L_surprise      # calibrate ε threshold (predicted vs actual mismatch)
```

Initial coefficients : `λ_vq=0.25, λ_trans=0.01, λ_route=0.1, λ_sep=0.05, λ_surprise=0.1`.

### 7.2 Curriculum — Track-P (no WMLs)

| Step | Goal | Pass criterion |
|---|---|---|
| P1 | VQ codebook alone on toy signals | dead codes < 10 % ; perplexity ≥ 32/64 |
| P2 | VQ + one transducer (toy src/dst pair) | KL(T ‖ uniform) > 1.0 ; retention ≥ 95 % |
| P3 | + γ/θ oscillator multiplexing | no phase collision ; constant latency |
| P4 | + sparse routing (4-node toy topology) | K-active ± 1 ; graph connected |

**Gate P** : P1-P4 green on 3 independent seeds.

### 7.3 Curriculum — Track-W (uses `MockNerve`)

| Step | Goal | Pass criterion |
|---|---|---|
| W1 | 2 MlpWMLs on a 2-step toy task | task solved, codes stable |
| W2 | + 2 LifWMLs (total 4) on the same task | perf gap MLP/LIF < 5 % |
| W3 | + ε feedback updates upstream transducers | baseline (no ε) beaten by ≥ 10 % |
| W4 | Continual learning : two sequential tasks | forgetting on task 1 < 20 % after task 2 |

**Gate W** : W1-W4 green, polymorphie invariant `|perf_mlp − perf_lif| / perf_mlp < 0.05`.

### 7.4 Merge training

Swap `MockNerve` for `SimNerve`, freeze WML internals and codebooks, fine-tune only the nerve transducers for ~20 % of Track-W step count.

**Gate M** : merged performance ≥ 95 % of Track-W-with-mock baseline.

### 7.5 Stability tricks

- Commitment loss β=0.25 (van den Oord 2017, VQ-VAE).
- EMA codebook update (no direct gradient on embeddings, prevents dead codes).
- Gumbel-τ annealing for routing : τ = 1.0 → 0.1 over 10 k steps.
- Surrogate LIF gradient : `grad ≈ 1 / (π · (1 + (α·v)²))` with α=2 (fast sigmoid).
- Codebook rotation (Zeghidour 2022) on dead codes after 50 idle steps.
- Role-separation loss : `L_role_sep = max(0, margin − KL(p_π ‖ p_ε))`.

### 7.6 Compute budget

| Track | Steps | Hardware | Wall time |
|---|---|---|---|
| Track-P (P1-P4) | ~30 k | GrosMac M5 (MLX) | ~3 h |
| Track-W (W1-W4) | ~50 k | GrosMac + KXKM if LIF slow | ~6-10 h |
| Merge (M) | ~10 k | GrosMac | ~1 h |
| **Total** | ~90 k | — | **~10-14 h** |

---

## 8. Testing strategy

Four levels, each tied to a gate.

### 8.1 L1 — Unit tests

```
tests/unit/
├── test_neuroletter.py     # immutable, hash, serialization
├── test_nerve_mock.py      # FIFO queue, role/phase filters
├── test_nerve_sim.py       # γ/θ timing, no collision, dt stepping
├── test_vq_codebook.py     # round-trip, EMA update, dead-code detection
├── test_transducer.py      # forward (Gumbel), backward, entropy reg
├── test_routing.py         # Gumbel top-K, sparsity, connectedness
├── test_surrogate_lif.py   # LIF step, spike gradient, membrane reset
└── test_role_heads.py      # π / ε separation, threshold dispatch
```

Coverage target : 90 %.

### 8.2 L2 — Info-theoretic (gate P)

```python
def test_capacity():
    """Empirical capacity in bits/s on random code emissions."""
    n = SimNerve(alphabet_size=64)
    emit_random_codes(n, duration_s=10)
    bits_per_letter = measure_shannon_entropy(n)   # ≤ log2(64) = 6
    rate_hz         = GAMMA_HZ + THETA_HZ          # ≈ 46 Hz
    assert bits_per_letter * rate_hz > 200         # bits/s minimum target

def test_collision_rate_by_phase():
    """γ and θ letters never collide."""
    assert count_same_time_different_phase(history) == 0

def test_dead_codes_after_training():
    """After warmup, < 10 % of the 64 codes unused."""
    dead = (cb.usage_counter == 0).sum()
    assert dead < 7

def test_disambiguation_pi_vs_eps():
    """π and ε distinguishable by code distribution."""
    assert kl_divergence(p_pi, p_eps) > 1.0
```

### 8.3 L3 — Integration (gates W, M)

```python
def test_end_to_end_cycle():
    """One γ+θ cycle, 4 mixed WMLs, flow assertions."""
    pool  = [MlpWML(0), MlpWML(1), LifWML(2), LifWML(3)]
    nerve = MockNerve()
    history = simulate(pool, nerve, steps=100)
    assert history.has_predictions_flowing_down()
    assert history.has_errors_flowing_up_when_surprise_high()

def test_polymorphie_w_gate():
    """Central scientific assertion."""
    perf_mlp = train_and_eval([MlpWML(i) for i in range(4)], task=SplitMNIST())
    perf_lif = train_and_eval([LifWML(i) for i in range(4)], task=SplitMNIST())
    gap = abs(perf_mlp - perf_lif) / perf_mlp
    assert gap < 0.05          # THE scientific invariant

def test_merge_gate_m():
    """Perf with SimNerve ≥ 95 % of perf with MockNerve."""
    pool_mock = load_gate_W_checkpoint()
    perf_mock = eval(pool_mock, MockNerve())
    pool_sim  = swap_nerve(pool_mock, SimNerve())
    perf_sim  = eval(pool_sim, SimNerve())
    assert perf_sim / perf_mock > 0.95
```

### 8.4 L4 — Golden regressions

Reproducibility contract (inspired by `dreamOfkiki/harness/storage/run_registry.py`) :

```python
run_id = sha256(c_version + topology + seed + commit_sha)[:16]
```

Frozen artifacts in `tests/golden/*.npz` :

- `cycle_trace_4wmls_seed0.npz` — neuroletter trace for 1000 cycles, seed 0.
- `vq_codebook_4wmls_end_of_training.npz` — codebooks at gate W.
- `transducers_merged.npz` — transducers post-merge, gate M.

CI test :

```python
def test_cycle_trace_bit_stable():
    expected = np.load("tests/golden/cycle_trace_4wmls_seed0.npz")
    actual   = run_cycle(seed=0, steps=1000)
    np.testing.assert_array_equal(actual.codes, expected["codes"])
```

### 8.5 Gate summary

| Gate | Required tests | Blocking ? |
|---|---|---|
| P | L1 (nerve, vq, transducer, routing) + L2 (capacity, collision, dead codes, disambig) | yes — blocks Track-P |
| W | L1 (WML) + L3 (polymorphie gap < 5 %, continual learning) | yes — blocks Track-W |
| M | Gate W + `test_merge_gate_m` + L4 golden frozen | yes — blocks release |

---

## 9. Module layout

```
nerve-wml/
├── pyproject.toml
├── README.md
├── CLAUDE.md
│
├── nerve_core/
│   ├── neuroletter.py      # Neuroletter, Role, Phase
│   ├── protocols.py        # Nerve, WML (Protocol classes)
│   └── invariants.py       # N-1..N-5, W-1..W-4 runtime guards
│
├── track_p/
│   ├── oscillators.py      # γ (40 Hz) / θ (6 Hz) continuous
│   ├── vq_codebook.py      # VQ-VAE, EMA, codebook rotation
│   ├── transducer.py       # 64×64 Gumbel soft matrix
│   ├── router.py           # top-K Gumbel sparse routing
│   ├── sim_nerve.py        # SimNerve implementing Nerve with real γ/θ
│   └── info_theoretic.py   # capacity, disambig, collision metrics
│
├── track_w/
│   ├── mock_nerve.py       # in-memory Nerve without rhythms
│   ├── mlp_wml.py          # MlpWML, 4×Linear(128) + π/ε heads
│   ├── lif_wml.py          # LifWML, 100 LIF + surrogate grad
│   ├── training.py         # W1-W4 curriculum + composite loss
│   └── tasks/
│       ├── split_mnist.py  # toy continual-learning task
│       └── flow_proxy.py   # signal proxy inspired by kiki_flow_core
│
├── bridge/
│   └── merge_trainer.py    # swap MockNerve → SimNerve, fine-tune 20 %
│
├── harness/
│   ├── run_registry.py     # run_id = sha256(version + topo + seed + commit)
│   └── storage.py
│
├── scripts/
│   ├── track_p_pilot.py    # curriculum P1-P4 → gate-P report
│   ├── track_w_pilot.py    # curriculum W1-W4 → gate-W report
│   ├── merge_pilot.py      # merge → gate-M report
│   └── eval_polymorphie.py # THE scientific test (gap < 5 %)
│
├── tests/
│   ├── unit/               # L1
│   ├── info_theoretic/     # L2
│   ├── integration/        # L3
│   ├── golden/             # L4 *.npz snapshots
│   └── conftest.py
│
├── docs/
│   ├── specs/              # this file lives here
│   ├── invariants/         # N-1..N-5, W-1..W-4
│   └── reference/
│       └── bastos-friston-2012.md
│
└── papers/
    └── paper1/
        ├── main.tex
        └── figures/
```

---

## 10. Location

Target : new repository `electron-rare/nerve-wml`, sibling to `electron-rare/dream-of-kiki`, `electron-rare/kiki-flow-research`, `electron-rare/micro-kiki`.

Rationale :

- Distinct identity : not a fork, not a plug-in — a standalone research engine.
- Consistent with existing pattern (each research track lives in its own repo with cross-links).
- Cross-links :
  - To `dream-of-kiki` : the accumulation of ε during an episode becomes consolidation targets during offline « dream » cycles — future `P_equ` integration.
  - To `kiki-flow-research` : shares VQ codec and streaming concepts.
  - To `micro-kiki` : phase 3+ plug-in to LLM runtime.

Alternative (deferred) : sub-project inside `dream-of-kiki/` if seen as empirical validation of the C-axioms rather than a standalone engine.

---

## 11. Scope — explicitly out of PoC1 (YAGNI)

| Feature | Why out of scope |
|---|---|
| LLM integration (plug into `micro-kiki`) | Phase 3+, requires isolated validation first |
| Neuromorphic hardware deployment (Loihi, Akida) | Deployment concern, not scientific validation |
| Online / streaming learning | Batch only |
| More than 4 WMLs | Scaling tests in phase 2 |
| STDP / Hebbian alternatives to backprop | Stick to surrogate gradient |
| Offline « dream » consolidation cycles | Alignment with `dream-of-kiki` is a follow-up phase |
| Natural language I/O | The « language » is neuroletters, not text |
| HTTP API / serving | This is a lab, not a service |
| Transformer integration | Stay with small MLPs + LIF |
| More than 2 continual tasks | If it works on 2, extend later |
| Multi-GPU / distributed training | GrosMac + optionally KXKM is sufficient |

---

## 12. References

- Bastos, A. M. et al. (2012). *Canonical microcircuits for predictive coding*. Neuron.
- Rao, R. P. & Ballard, D. H. (1999). *Predictive coding in the visual cortex*. Nature Neuroscience.
- van den Oord, A. et al. (2017). *Neural Discrete Representation Learning* (VQ-VAE).
- Zeghidour, N. et al. (2022). *SoundStream*.
- Neftci, E. O. et al. (2019). *Surrogate Gradient Learning in Spiking Neural Networks*.
- Related in-house projects : `electron-rare/dream-of-kiki` (predictive coding axioms, profile `P_equ`) ; `electron-rare/kiki-flow-research` (VQ + streaming surrogate) ; `electron-rare/micro-kiki` (downstream LLM integration target).

---

## 13. Open questions / future work

- **Multi-alphabet extension.** Could the 64-code alphabet grow or shrink adaptively per WML once the system is stable ? — **PLAN 8 RESOLVED (post-hoc, 2026-04-19)**. `track_p.adaptive_codebook.AdaptiveCodebook` wraps a fixed 64-slot `VQCodebook` with an `active_mask` so shrink/grow are logical, not physical. `bridge.transducer_resize` produces a reshaped `Transducer` preserving argmax on kept rows. `gate-adaptive-passed` verifies a shrink→grow cycle completes without the alphabet collapsing below 4 codes. Remaining: wire into live WMLs at training time (currently applied post-hoc on checkpoints).
- **Hardware deployment.** How do the LIF WMLs port to Loihi or Akida once the software PoC is validated ?
- **Dream integration.** What is the minimal interface between `nerve-wml` and `dream-of-kiki` to replay ε traces during offline consolidation ?
- **Scaling behaviour.** Does the polymorphie gap remain < 5 % at N = 16 or 32 WMLs ?
- **Causal inspection.** Can we read off a learned « neuroletter semantics » table (code → concept) from the trained system, as an interpretability artifact ?

### 13.1 Known limitations from Plan 1/2 execution

The following scientific shortcuts were taken to ship gates P and W on schedule. They are documented here to prevent misinterpretation of results:

- **P3 γ-priority ablation.** Gate P3 asserts 0 collisions, which is trivially true given the γ-priority rule in `SimNerve.listen()`. Measure the collision rate WITHOUT the priority rule to quantify how much work the rule is doing. Expected: ~25 % overlap (γ 50 % active × θ 50 % active, independent). — Raised during Plan 1 code review.
- **W2 true-LIF polymorphie.** The current W2 pilot evaluates `LifWML` via a linear probe on `input_proj`, which bypasses the spike dynamics + pattern-match decoder that constitute the actual LIF path. A faithful polymorphie test would drive the full `step()` loop on a sequence task and read emissions. Expected: non-zero gap, possibly > 5 %. — Raised during Plan 2 execution.
- **W4 true continual learning.** The current W4 pilot uses disjoint output heads (Task 0 → classes 0-1, Task 1 → classes 2-3) and a reduced lr for Task 1. This avoids catastrophic forgetting by construction rather than by algorithmic means. A faithful W4 would share a full-class head and train both tasks at the same lr; forgetting would likely exceed 20 %. EWC or rehearsal would then become meaningful additions. — Raised during Plan 2 execution.
- **P1 fully-random VQ convergence.** `run_p1_random_init` (Plan 3 Task 11) demonstrates the protocol converges without MOG cluster-center init, but does not enforce the dead-code < 10 % gate. Future work: characterise the convergence rate and find a training recipe that reaches gate under random init.

#### Resolution status (Plan 4a)

- **P3 γ-priority ablation** — RESOLVED (2026-04-18). `SimNerve` now accepts `priority_rule: bool`; `run_p3_no_priority` (in `scripts/track_p_pilot.py`) measures a **26 % collision rate** matching the §13.1 prediction of ~25 %. See `tests/integration/test_gate_p3_ablation.py`.
- **W2 true-LIF polymorphie** — RESOLVED (2026-04-20, v1.1). First (v0.4) added `LifWML.emit_head_pi` symmetric to `MlpWML.emit_head_pi`; this revealed the original 12.1 % gap as a fixed-cosine-decoder artefact rather than a substrate limitation. Re-measurement showed the gap flipped sign (LIF > MLP by 10.7 % at N=2). Pool-scale multi-seed (5 seeds × N ∈ {16, 32, 64}) then traced a four-point scaling law: median gap $10.71\% \to 6.71\% \to 2.39\% \to 2.73\%$ with a plateau at $\sim 2\text{--}3\%$ beyond $N=32$. Every seed at $N \geq 32$ satisfies the `< 5 %` contract; LIF $\geq$ MLP direction stable in 15/15 multi-seed measurements. Plateau is interpreted as a substrate-intrinsic expressivity floor (spike dynamics on XOR-on-noise), not residual variance. Final paper claim: contract holds distributionally at $N \geq 32$, with honest $\sim 2\text{--}3\%$ LIF edge on non-linear tasks. See `papers/paper1/figures/w2_hard_scaling.pdf` and `tests/integration/track_w/test_w2_hard_scale.py`.

- **Inter-substrate information transmission** — RESOLVED (2026-04-20, v1.1). Previously Claim B was architectural (transducers postulated to transmit information across substrates). Now three direct measurements pin it: $\mathrm{MI}(c_{\text{MLP}}, c_{\text{LIF}}) / H(c_{\text{MLP}}) = 0.91$ at N=1 and $0.96$ at N=16 pool scale (192 cross-pair measurements); round-trip fidelity MLP→LIF→MLP $= 0.99$; cross-substrate merge ratio $= 0.97$ (LIF frozen, fed only by MLP-emitted codes through a learned transducer). Pool-scale (2) and (3) hold over 64 cross-pairs per seed. See `scripts/measure_info_transmission.py`, `tests/integration/track_w/test_info_transmission.py`, `papers/paper1/figures/info_transmission.pdf`.

- **Third substrate (TransformerWML)** — ADDED (2026-04-20, v1.1). Attention-based WML validates substrate-agnostic claim across three structurally distinct implementations. Saturates FlowProxyTask at 1.000 (triple-gap 0 %). On HardFlowProxyTask at N=1, median triple-gap 5.59 % (5 seeds). Pool-scale triple-substrate measurement (N=16/32 with TRF included) is **still open** — requires pool_factory refactor to support 3 substrate types.
- **W4 true continual learning** — RESOLVED (2026-04-18). Shared-head baseline (`run_w4_shared_head`) shows **100 % forgetting without mitigation**. Rehearsal recipe (`run_w4_rehearsal` with `rehearsal_frac=0.3`) drops it to **0 %**, well under the 20 % gate. See `tests/integration/track_w/test_gate_w4_honest.py`.
- **P1 fully-random VQ convergence** — RESOLVED (2026-04-18). `VQCodebook.rotate_dead_codes` (Zeghidour 2022) invoked every 500 steps brings dead-code fraction from 39 % → **0 %** at 16 000 steps, without any MOG cluster-center leak. Gate enforced by `tests/integration/test_gate_p1_random.py`.

---

## 14. Sign-off

- [ ] Design approved by author
- [ ] Implementation plan written (see `docs/superpowers/plans/YYYY-MM-DD-nerve-wml-plan.md`)
- [ ] Track-P gate P passed
- [ ] Track-W gate W passed
- [ ] Merge gate M passed
- [ ] Paper draft v0.1 released
