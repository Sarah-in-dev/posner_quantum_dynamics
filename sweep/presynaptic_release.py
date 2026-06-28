"""
Presynaptic stochastic release module for Model 6 (stimulus-construction layer).

Produces brief cleft-glutamate EVENTS (per step, 0 or 1) for a single synapse,
driven by an activation signal. Consumed by the postsynaptic NMDAR occupancy
(g_engaged) in analytical_calcium_system.py via stimulus['glutamate'].

LOCUS (locked): release is a presynaptic / stimulus-construction property and
lives HERE, at the experiment seam -- NOT in model6_core, which forwards the
stimulus dict unchanged. The ~166 ms NMDAR coincidence window is a RECEPTOR
property and lives in the channel (g_engaged, tau_nmda); it is NOT duplicated here.

Mechanism (Tsodyks-Markram-style probabilistic uniquantal release with STP):
  - Place-cell firing: inhomogeneous Poisson at rate = baseline + act*peak (Hz).
  - Per spike: facilitation increments pv; uniquantal Bernoulli release at
    Ps = 1 - (1 - pv)^n   (n = current readily-releasable-pool occupancy).
  - Facilitation relaxes to pv0 (tau_f); RRP depletes by 1 per release, recovers
    to N_max (tau_r). Depression bounds sustained release -- deliberate (upstream
    of the downstream calcium/dimer runaway bound).

Per-synapse heterogeneity (INDEPENDENT draws at init):
  - pv0   ~ lognormal(median PV0_MEDIAN)            -> per-vesicle release prob
  - N_max ~ Poisson(RRP_MEAN), clipped [1, 15]      -> RRP size
  - peak  ~ lognormal(median PEAK_RATE_MEDIAN), clipped [PEAK_MIN, PEAK_MAX] Hz
  Ps0 = 1 - (1 - pv0)^N_max EMERGES (NOT drawn directly).

PROVISIONAL (flagged; NOT tuned to any calcium target -- emergent-physics
discipline): the distribution SHAPES (the lognormal sigmas, the RRP draw shape)
are placeholders pending direct reads of the Mizuseki & Buzsaki and Dobrunz &
Stevens figures. Central values are grounded; spreads are provisional. Changing
a sigma is a one-line edit. Do NOT calibrate any of these to reproduce a
calcium level.

Parameter sources:
  pv0~0.03, RRP N~8, Ps0~0.2      Mahajan & Nadkarni 2020 (eNeuro); Dobrunz & Stevens 1997
  tau_f=150 ms, alpha_f=0.03      Klyachko & Stevens 2006; Mahajan & Nadkarni 2020
  tau_r=2 s (RRP recovery)        Klyachko & Stevens 2006
  ~10 ms refractory               Stevens & Wang 1995
  baseline 0-1 Hz, in-field 5-60 (mean ~30)   Fenton & Muller 1998
  peak-rate lognormal spread      Mizuseki & Buzsaki 2013
Canonical reference for exactly this synapse (CA3->CA1, place-cell driven,
probabilistic release + STP feeding NMDAR Ca): Mahajan & Nadkarni 2020.

TIMESTEP: the live runners step at dt = 5 ms. The ~1-2 ms cleft transient is
sub-step; represented as a per-step 0/1 EVENT, and the ~166 ms receptor occupancy
latch (g_engaged) carries it. Firing is discretized as Bernoulli(1 - exp(-rate*dt))
per step (<= 1 spike/step), consistent with the ~10 ms refractory.
"""

import math
import numpy as np

# --- grounded central values ---
PV0_MEDIAN = 0.03            # per-vesicle release probability (median)
PV0_SIGMA_LOG = 0.8          # PROVISIONAL spread (pending Dobrunz & Stevens read)
PV0_MIN, PV0_MAX = 0.005, 0.5

RRP_MEAN = 8.0               # readily-releasable pool size (mean); Dobrunz & Stevens
RRP_MIN, RRP_MAX = 1, 15

PEAK_RATE_MEDIAN = 25.0      # Hz; in-field peak (mean ~30 after lognormal skew)
PEAK_RATE_SIGMA_LOG = 0.6    # PROVISIONAL spread (pending Mizuseki & Buzsaki read)
PEAK_RATE_MIN, PEAK_RATE_MAX = 5.0, 60.0

BASELINE_RATE_HZ = 0.5       # out-of-field spontaneous (0-1 Hz)

TAU_F = 0.150                # s, facilitation relaxation
ALPHA_F = 0.03               # facilitation increment per spike
TAU_R = 2.0                  # s, RRP recovery
TAU_REF = 0.010              # s, absolute refractory


class PresynapticRelease:
    """One stateful presynaptic terminal. Construct once per synapse; step each
    physics timestep with that synapse's activation."""

    def __init__(self, seed,
                 pv0_median=PV0_MEDIAN, pv0_sigma_log=PV0_SIGMA_LOG,
                 rrp_mean=RRP_MEAN,
                 peak_rate_median=PEAK_RATE_MEDIAN, peak_rate_sigma_log=PEAK_RATE_SIGMA_LOG,
                 baseline_rate_hz=BASELINE_RATE_HZ,
                 tau_f=TAU_F, alpha_f=ALPHA_F, tau_r=TAU_R, tau_ref=TAU_REF):
        self.rng = np.random.default_rng(seed)

        # --- per-synapse heterogeneity (independent draws) ---
        self.pv0 = float(np.clip(
            self.rng.lognormal(mean=math.log(pv0_median), sigma=pv0_sigma_log),
            PV0_MIN, PV0_MAX))
        self.N_max = float(np.clip(self.rng.poisson(rrp_mean), RRP_MIN, RRP_MAX))
        self.peak_rate = float(np.clip(
            self.rng.lognormal(mean=math.log(peak_rate_median), sigma=peak_rate_sigma_log),
            PEAK_RATE_MIN, PEAK_RATE_MAX))

        self.baseline_rate = float(baseline_rate_hz)
        self.tau_f = float(tau_f)
        self.alpha_f = float(alpha_f)
        self.tau_r = float(tau_r)
        self.tau_ref = float(tau_ref)

        # --- dynamic state ---
        self.pv = self.pv0          # facilitation variable
        self.n = self.N_max         # RRP occupancy (real-valued during recovery)
        self.refractory_remaining = 0.0

    @property
    def Ps0(self):
        """Baseline resting synaptic release probability -- emergent, reported."""
        return 1.0 - (1.0 - self.pv0) ** self.N_max

    def step(self, act, dt):
        """Advance one timestep. `act` in [0,1]. Returns the cleft-glutamate event
        for this step: 1.0 on uniquantal release, else 0.0."""
        a = min(max(float(act), 0.0), 1.0)

        # 1. RRP recovery toward N_max
        self.n = self.N_max - (self.N_max - self.n) * math.exp(-dt / self.tau_r)
        # 2. facilitation relaxation toward pv0
        self.pv = self.pv0 + (self.pv - self.pv0) * math.exp(-dt / self.tau_f)
        # 3. refractory countdown
        if self.refractory_remaining > 0.0:
            self.refractory_remaining = max(0.0, self.refractory_remaining - dt)

        # 4. firing (inhomogeneous Poisson -> Bernoulli per step; <=1 spike/step)
        rate = self.baseline_rate + a * self.peak_rate
        spike = False
        if self.refractory_remaining <= 0.0:
            p_spike = 1.0 - math.exp(-rate * dt)
            spike = self.rng.random() < p_spike

        # 5. uniquantal release on spike
        cleft = 0.0
        if spike:
            self.refractory_remaining = self.tau_ref
            self.pv = self.pv + self.alpha_f * (1.0 - self.pv)      # facilitation
            Ps = 1.0 - (1.0 - self.pv) ** self.n
            if self.rng.random() < Ps:
                cleft = 1.0
                self.n = max(0.0, self.n - 1.0)                     # uniquantal depletion
        return cleft

    def advance_silent(self, duration):
        """Advance state through a silent gap of `duration` seconds with no firing:
        RRP recovers toward N_max (tau_r) and facilitation relaxes toward pv0 (tau_f).
        Used between trials/passes where per-step stepping would be wasteful.
        Spontaneous baseline release during the gap is neglected (postsynaptic calcium
        is not integrated during an analytical gap anyway)."""
        self.n = self.N_max - (self.N_max - self.n) * math.exp(-duration / self.tau_r)
        self.pv = self.pv0 + (self.pv - self.pv0) * math.exp(-duration / self.tau_f)
        self.refractory_remaining = 0.0


if __name__ == "__main__":
    # Smoke test (NOT the characterization probe): confirm import + run + sane stats.
    dt = 0.005
    print("per-synapse draws (seeds 0-4):")
    for s in range(5):
        r = PresynapticRelease(seed=s)
        print(f"  syn {s}: pv0={r.pv0:.4f}  N_max={r.N_max:.0f}  Ps0={r.Ps0:.3f}  peak={r.peak_rate:.1f} Hz")

    def mean_release_rate(act, n_steps=100000, seed=0):
        r = PresynapticRelease(seed=seed)
        rel = sum(r.step(act, dt) for _ in range(n_steps))
        return rel / (n_steps * dt)   # releases per second

    print("\nrelease rate (Hz) vs activation (syn seed 0):")
    for act in (0.0, 0.5, 1.0):
        print(f"  act={act:.1f}: {mean_release_rate(act):.2f} releases/s")
