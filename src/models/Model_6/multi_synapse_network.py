"""
Multi-Synapse Network Module
============================

Implements realistic multi-synapse architecture where:
- Each synapse is an independent Model6 instance with its own grid
- Synapses are positioned along a dendritic segment
- Quantum fields couple through the shared microtubule network
- Network-level threshold determines commitment

ARCHITECTURE:
------------
```
Dendrite shaft (with microtubules running through)
    │
    ├── Spine 1 (Model6 instance, own 100x100 grid) ─┐
    │       └── ~5 dimers, own calcium dynamics      │
    ├── Spine 2 (Model6 instance, own 100x100 grid)  │
    │       └── ~5 dimers, own calcium dynamics      ├── Within 20µm
    ├── ...                                          │   Fields couple via MT
    ├── Spine N (Model6 instance, own 100x100 grid) ─┘
    │       └── ~5 dimers, own calcium dynamics
```

KEY PHYSICS:
-----------
1. Each synapse independently produces ~4-6 dimers (stochastic)
2. Dimers create local quantum fields
3. Fields propagate through dendritic microtubules (decay with distance)
4. Network field = sum of all synapse contributions (with distance weighting)
5. Commitment occurs when network field exceeds threshold (~50 dimers worth)

This replaces the "multiply by N" hack with actual multi-synapse physics.

"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from model6_parameters import (
    compute_metabolic_power, P_BASAL_W, bose_einstein_occupation, hbar,
)
from entanglement_topology import compute_betti, compute_synapse_quotient_betti

logger = logging.getLogger(__name__)


@dataclass
class SynapseState:
    """State of a single synapse in the network"""
    position_um: np.ndarray  # (x, y, z) position in microns
    dimer_count: float = 0.0
    coherence: float = 0.0
    collective_field_kT: float = 0.0
    eligibility: float = 0.0
    committed: bool = False
    committed_level: float = 0.0
    calcium_peak_uM: float = 0.0


@dataclass 
class NetworkState:
    """Aggregate state of the multi-synapse network"""
    n_synapses: int
    total_dimers: float = 0.0
    network_field_kT: float = 0.0
    mean_coherence: float = 0.0
    mean_eligibility: float = 0.0
    n_committed: int = 0
    network_committed: bool = False
    network_commitment_level: float = 0.0
    synapse_states: List[SynapseState] = field(default_factory=list)

class NetworkEntanglementTracker:
    """
    Tracks entanglement ACROSS synapses in a multi-synapse network
    
    Physics basis (Fisher 2015):
    - Entanglement from shared ATP hydrolysis pool
    - Nearby synapses share dendritic J-coupling environment
    - Coupling decays with inter-synapse distance
    
    This replaces the naive sum of per-synapse entangled counts
    with true network-level entanglement tracking.
    """

    # Cross-synapse bond formation/dissolution rate constants.
    # K_ENTANGLE_EM_BASE matches the per-synapse Pathway 2 base rate used in
    # DimerParticleSystem (k_entangle = 0.5 in the old _update_entanglement).
    # K_DISENTANGLE_BASE matches the old k_disentangle base (= 0.1).
    K_ENTANGLE_EM_BASE: float = 0.5
    K_DISENTANGLE_BASE: float = 0.1
    WERNER_ENTANGLEMENT_BOUND: float = 0.5  # Werner state entangled iff F > 1/2

    def __init__(self, coupling_length_um: float = 5.0, 
                 j_coupling_threshold: float = 5.0,
                 coherence_threshold: float = 0.3):
        self.coupling_length_um = coupling_length_um
        self.j_coupling_threshold = j_coupling_threshold
        self.coherence_threshold = coherence_threshold
        
        # Network state
        self.all_dimers = []  # List of dimer dicts keyed by stable_id

        # === Two-container split (option (c) cutover, May 13 2026) ===
        self.cross_synapse_bonds: Dict[Tuple, float] = {}        # tracker-owned cross-spine edges, weight = P_S_i × P_S_j
        self.intra_synapse_bonds_cache: Dict[Tuple, float] = {}  # rebuilt each collect_dimers from per-synapse systems

        # === PO-7: NETWORK-SHARED PROVENANCE EVENTS (opt-in; all-off bit-identical) ===
        # Fisher's inheritance channel lifted from per-synapse to a NETWORK pool. Hydrolysis
        # events carry ABSOLUTE (x,y) network coordinates; a newborn dimer in ANY synapse claims
        # its nearest events within reach, and two dimers (possibly in DIFFERENT synapses) that
        # claim the two daughters of one event bond — a cross-synapse edge, Fisher-inherited from
        # a shared hydrolysis origin, and therefore η-FREE (no condensate mediation, so the dead
        # pump r≈0.077/η=0 does not block it). Kept in a SEPARATE container so the η-gated
        # _update_entanglement dissolution never rewrites these edges. Off => empty => no unions
        # in _find_all_clusters => identical partition. See docs/PREREG_PO7_NETWORK_PROVENANCE.md.
        self.provenance_network: bool = False                    # master opt-in flag
        self.provenance_net_event_rate: float = 0.5             # events / active-cell / step (reported, not tuned)
        self.provenance_net_age_s: float = 2.0                  # events expire (phosphates consumed ~s)
        self.provenance_net_reach_nm: float = 500.0            # claim radius (shared-origin phosphate diffusion reach)
        self.provenance_net_ca_threshold: float = 1e-6         # M; matches atp_system active-site rule
        self.provenance_net_event_slots: int = 2               # entangled daughters per hydrolysis event
        self.provenance_net_k: int = 2                         # phosphate pairs per Ca6(PO4)4 dimer (LOCKED qsc:43)
        self._prov_bonds: Dict[Tuple, float] = {}              # (gid_i, gid_j) -> Werner F = P_S_i*P_S_j
        self._prov_events: list = []                           # shared pool: {id,pos(abs nm xy),t,slots_free,holders}
        self._prov_next_event_id: int = 0
        self._prov_time: float = 0.0                           # tracker-local clock for event aging
        self._prov_seen: set = set()                           # global_ids already granted provenance (born-with)
        self._prov_last_stats: dict = {}                       # diagnostics for the probe (overlap fraction etc.)
        # PO-7 (advisor R4): WHICH-SPIN provenance. Fisher's inheritance names the phosphate
        # slot, not just the partner. 4 spins per Ca6(PO4)4, K=2 claims => claim j takes spin j.
        # Side-band only — no dynamics read these, so they perturb nothing on any path. They are
        # the prerequisite for the monogamy bound (a spin mediates at most one bond) and for a
        # channel rule derived from the inherited pair instead of bond direction.
        self._prov_slot_of: Dict = {}                          # global_id -> {event_id: spin idx}
        self._prov_bond_spins: Dict = {}                       # (gid_lo, gid_hi) -> (spin_lo, spin_hi)

        # === PO-7 UNIT 11: SHARED PER-DIMER SPIN LEDGER (intra + cross) ===
        # Unit 9 gave every dimer four 31P slots and made INTRA bonds claim them
        # (dimer_particles._create_bond:709). Cross-synapse bonds are not created there, so
        # they consumed NO spins: a dimer could hold 4 intra bonds AND unlimited cross bonds,
        # and monogamy was enforced per-synapse while violated network-wide. Unit 10 measured
        # the consequence (largest_frac 0.209 -> 0.959 as cross bonds accumulated unchecked).
        # A dimer's ledger lives on ITS OWN synapse's DimerParticleSystem, so cross bonds now
        # claim from the SAME four slots the intra bonds spend. Gated by the existing opt-in:
        # inert unless that synapse's dimer_particles.spin_resolved is True.
        self._synapses_ref = None                              # set by collect_dimers; who owns which ledger
        self._cross_spin_frustrated: int = 0                   # cross bonds refused for want of a free spin
        self._cross_bond_spins: Dict = {}                      # (gid_lo, gid_hi) -> (spin_lo, spin_hi)

    @property
    def entanglement_bonds(self):
        """Union of cross-synapse and intra-synapse bonds. Computed on demand.

        This is the read-only view used by callers that need every bond regardless
        of class — primarily connected-component partitioning for per-cluster
        quantum measurement. Mutations go through the canonical containers
        (cross_synapse_bonds, or DimerParticleSystem.entanglement_bonds via the
        intra cache mirror).
        """
        return set(self.cross_synapse_bonds.keys()) | set(self.intra_synapse_bonds_cache.keys())

    def collect_dimers(self, synapses: List, positions: np.ndarray):
        """
        Collect all dimers from all synapses with position info.

        Uses stable IDs: (synapse_idx, dimer.id) tuples that survive
        across collect_dimers() calls as long as the dimer exists.

        As of May 13 2026 (option (c) cutover, edit 2/7):
        - Each dimer dict now carries 'eta' (synapse._backbone_eta) and
          'mt_invaded' (synapse._mt_invaded). These are needed for the
          cross-synapse bond formation rate in _update_entanglement.
        - self.intra_synapse_bonds_cache is rebuilt every call by
          reading each synapse's per-synapse DimerParticleSystem.entanglement_bonds
          and projecting bond.dimer_i/j into network stable-id tuples
          with weight = bond.strength.

        Parameters
        ----------
        synapses : List[Model6QuantumSynapse]
            All synapse models in network
        positions : np.ndarray
            Synapse positions in microns, shape (n_synapses, 3)
        """
        self.all_dimers = []
        self.intra_synapse_bonds_cache = {}  # rebuilt every call
        # PO-7 Unit 11: remember who owns which spin ledger. collect_dimers runs first in
        # step(), so every cross-bond path below can reach a dimer's own DimerParticleSystem
        # without threading `synapses` through signatures that never had it.
        self._synapses_ref = synapses

        for syn_idx, synapse in enumerate(synapses):
            if not hasattr(synapse, 'dimer_particles'):
                continue

            particle_system = synapse.dimer_particles
            synapse_pos = positions[syn_idx]

            # Synapse-level state for cross-synapse formation (read once per synapse)
            eta = float(getattr(synapse, '_backbone_eta', 0.0))
            mt_invaded = bool(getattr(synapse, '_mt_invaded', False))

            for dimer in particle_system.dimers:
                stable_id = (syn_idx, dimer.id)
                self.all_dimers.append({
                    'global_id': stable_id,
                    'dimer': dimer,
                    'synapse_idx': syn_idx,
                    'synapse_pos_um': synapse_pos,
                    'local_j': dimer.local_j_coupling,
                    'coherence': dimer.coherence,
                    'P_S': dimer.singlet_probability,
                    'eta': eta,
                    'mt_invaded': mt_invaded,
                })

            # Read-through: per-synapse bonds projected into network stable-id space.
            # EntanglementBond dataclass (dimer_particles.py:69) has
            # fields: dimer_i (int), dimer_j (int), strength (float), formation_time (float).
            for bond in particle_system.entanglement_bonds:
                id_a = (syn_idx, bond.dimer_i)
                id_b = (syn_idx, bond.dimer_j)
                key = (min(id_a, id_b), max(id_a, id_b))
                self.intra_synapse_bonds_cache[key] = bond.strength
    
    def step(self, dt: float, synapses: List, positions: np.ndarray,
             coupling_weights: Optional[np.ndarray] = None) -> dict:
        """
        Update network-level entanglement

        Returns
        -------
        dict with network metrics
        """
        # Collect current dimers from all synapses
        self.collect_dimers(synapses, positions)

        n = len(self.all_dimers)
        if n < 2:
            return {
                'n_total_dimers': n,
                'n_entangled_network': n if n == 1 else 0,
                'n_bonds': 0,
                'f_entangled': 0.0,
                'betti0': 0,
                'betti1': 0,
                'component_sizes': [],
            }

        # Update entanglement bonds
        self._update_entanglement(dt, coupling_weights)

        # PO-7: network-shared provenance events (opt-in; no state/RNG touched when off).
        if self.provenance_network:
            self._step_network_provenance(dt, synapses, positions)

        # Find largest connected cluster
        largest_cluster = self._find_largest_cluster()
        
        # Compute fraction
        entangled_ids = set()
        for bond in self.entanglement_bonds:
            entangled_ids.add(bond[0])
            entangled_ids.add(bond[1])
        
        f_entangled = len(entangled_ids) / n if n > 0 else 0.0

        topo = self.compute_entanglement_topology()

        return {
            'n_total_dimers': n,
            'n_entangled_network': len(largest_cluster),
            'n_bonds': len(self.entanglement_bonds),
            'f_entangled': f_entangled,
            'betti0': topo.betti0,
            'betti1': topo.betti1,
            'component_sizes': topo.component_sizes,
        }
    
    # =========================================================================
    # PO-7 UNIT 11 — the shared per-dimer spin ledger, reached from the network
    # =========================================================================

    def _ledger_for(self, gid):
        """The spin ledger owning `gid` = (syn_idx, dimer_id), or None when that synapse is
        not spin-resolved. A dimer's four 31P nuclei live on its OWN synapse's
        DimerParticleSystem — that is the whole point: intra and cross bonds spend the same
        four slots. Returns None (=> inert) whenever the opt-in flag is off, which is what
        makes the flag-OFF path byte-identical."""
        syns = self._synapses_ref
        if syns is None:
            return None
        syn_idx = gid[0]
        if not (0 <= syn_idx < len(syns)):
            return None
        ps = getattr(syns[syn_idx], 'dimer_particles', None)
        if ps is None or not getattr(ps, 'spin_resolved', False):
            return None
        return ps

    def _claim_cross_spins(self, key, required=None):
        """Claim one 31P spin at EACH endpoint of cross/provenance bond `key`. Returns True
        if the bond may form, False if it is refused for want of a free spin.

        Mirrors dimer_particles._create_bond:709-719 exactly: claim i, claim j, roll back i
        if j fails, count the refusal as frustration. `required` pins named slots — that is
        how provenance-inherited bonds stay frustratable (the inherited nucleus sits in a
        specific slot, so two inheritances competing for one slot cannot both be satisfied).

        Returns True unconditionally when neither endpoint is spin-resolved, so this is a
        no-op on the default path.
        """
        gi, gj = key
        li, lj = self._ledger_for(gi), self._ledger_for(gj)
        if li is None and lj is None:
            return True
        if key in self._cross_bond_spins:
            return True                      # already holding its two nuclei
        ri, rj = required if required else (None, None)
        si = li._claim_spin(gi[1], key, ri) if li is not None else -1
        if si is None:
            self._cross_spin_frustrated += 1
            return False
        sj = lj._claim_spin(gj[1], key, rj) if lj is not None else -1
        if sj is None:
            if li is not None:
                li._release_spin(gi[1], key)  # roll back i's claim
            self._cross_spin_frustrated += 1
            return False
        self._cross_bond_spins[key] = (si, sj)
        return True

    def _release_cross_spins(self, key):
        """Release both endpoints' nuclei for cross/provenance bond `key`.

        MUST be called on every path that removes the bond. The prune at the top of
        _update_entanglement rebuilds cross_synapse_bonds by dict comprehension, which
        releases nothing — leaving that path unhandled leaks spins and makes the ledger
        over-frustrate monotonically over a run.
        """
        if self._cross_bond_spins.pop(key, None) is None:
            return
        for gid in key:
            led = self._ledger_for(gid)
            if led is not None:
                led._release_spin(gid[1], key)

    def _update_entanglement(self, dt: float, coupling_weights=None):
        """
        Cross-synapse bond dynamics.

        As of May 13 2026 (option (c) cutover, edit 3/6):
        - Intra-synapse bonds are NOT formed here. They are owned by
          DimerParticleSystem via Pathway 1 (birth coherence, same ATP burst)
          and Pathway 2 (local EM-mediated), and read through to
          self.intra_synapse_bonds_cache by collect_dimers().
        - Cross-synapse bonds (different synapse_idx on each endpoint) are
          owned here. Formation rate is gated by:
            * Backbone Fröhlich condensation: sqrt(eta_i × eta_j)
              (linear in eta is correct — eta itself already encodes the
               second-order phase transition per Zhang/Agarwal/Scully 2019;
               weak-regime kinetic modulation per Reimers 2009)
            * Spatial coupling along the lattice: coupling_weights[i,j]
            * Microtubule invasion at both spines: hard gate (lattice continuity)
            * Singlet character: P_S_i × P_S_j (also stored as edge weight)
        - The entanglement_bonds property exposes the union of both bond
          classes for callers that need all bonds (e.g. connected-component
          partitioning).

        Vectorised 2026-07-16 (physics unchanged). The scalar all-pairs double
        loop paid O(n_dimers^2) in Python while discarding the intra pairs it
        iterated. The rates factorise per synapse pair — eta and mt_invaded are
        per-SYNAPSE (collect_dimers reads them once per synapse and copies them
        onto every dimer dict), w_spatial is per synapse pair, and only P_S is
        per-dimer — so k_cross and F_werner are outer products over the two
        spines' P_S vectors. Same rate constants, same Werner fidelity, same
        one-uniform-draw-per-pair-per-step semantics.

        NOT bit-reproducible against the old loop for a given seed: the draws
        are the same in number and distribution but are consumed as a block per
        synapse pair rather than one at a time, so the RNG stream order differs.
        Equivalence is therefore established distributionally, plus exactly in
        the saturated (p_form->1) and gated (mt_invaded False) limits.
        """
        # Prune cross-synapse bonds for dimers that no longer exist
        current_ids = {d['global_id'] for d in self.all_dimers}
        # PO-7 Unit 11: a rebuilt dict releases nothing. Hand back the nuclei of every bond
        # this prune drops, or the ledger over-frustrates monotonically as dimers turn over.
        # Guarded on the ledger being non-empty, so the flag-OFF path pays one dict test.
        if self._cross_bond_spins:
            for k in [k for k in self.cross_synapse_bonds
                      if k[0] not in current_ids or k[1] not in current_ids]:
                self._release_cross_spins(k)
        self.cross_synapse_bonds = {
            k: v for k, v in self.cross_synapse_bonds.items()
            if k[0] in current_ids and k[1] in current_ids
        }

        # Cross-synapse formation only runs when caller supplied coupling_weights.
        # (Backward compat: if caller doesn't pass it, no cross bonds form —
        # the test then exercises the intra-synapse path only.)
        #
        # THIS RETURN USED TO BE SILENT, and that silence is why the defect survived
        # a fix that "covered" the file: substrate-audit item 16 records the
        # 2026-07-18 fix reaching one call site in run_spatial_discovery.py and
        # missing the other, so every learning trial formed ZERO cross-synapse bonds
        # while looking healthy. A no-op that announces nothing is indistinguishable
        # from a no-op that was correct. It now announces itself.
        #
        # dt > 0 is the discriminator: analytical_gap deliberately calls this with
        # dt=0 to prune stale bonds only (run_theta_burst_45s.py:391), where forming
        # nothing is the intent, not a defect. Warning there would train people to
        # ignore the warning.
        if coupling_weights is None:
            if dt > 0 and not getattr(self, '_warned_no_coupling_weights', False):
                self._warned_no_coupling_weights = True
                logger.warning(
                    "_update_entanglement called with dt=%.4g and coupling_weights=None: "
                    "NO cross-synapse bonds will form. If this is a driver call site, the "
                    "network topology is empty and any result reading it is reading an "
                    "empty graph. Pass coupling_weights=network.coupling_weights. "
                    "(Warned once per tracker.)", dt)
            return

        # Bucket dimers by synapse. Order within a bucket fixes the row/col index.
        by_syn: Dict[int, list] = {}
        for d in self.all_dimers:
            by_syn.setdefault(d['synapse_idx'], []).append(d)
        syn_indices = sorted(by_syn)

        # Bucket existing cross bonds by synapse pair — O(n_bonds), sparse.
        existing_by_pair: Dict[Tuple, list] = {}
        for key in self.cross_synapse_bonds:
            (sa, _), (sb, _) = key
            existing_by_pair.setdefault((sa, sb), []).append(key)

        for ai in range(len(syn_indices)):
            for bi in range(ai + 1, len(syn_indices)):
                a, b = syn_indices[ai], syn_indices[bi]   # a < b
                A, B = by_syn[a], by_syn[b]
                if not A or not B:
                    continue

                # Hard gate: both spines must have invading MTs. Per-synapse, so
                # one representative dimer answers for the whole block.
                if not (A[0]['mt_invaded'] and B[0]['mt_invaded']):
                    for key in existing_by_pair.get((a, b), ()):
                        if self.cross_synapse_bonds.pop(key, None) is not None:
                            self._release_cross_spins(key)   # PO-7 U11
                    continue

                # Geometric mean of eta — both synapses must couple
                eta_factor = float((A[0]['eta'] * B[0]['eta']) ** 0.5)
                w_spatial = float(coupling_weights[a, b])

                P_A = np.array([d['P_S'] for d in A], dtype=float)
                P_B = np.array([d['P_S'] for d in B], dtype=float)
                P_product = np.outer(P_A, P_B)

                # Werner fidelity F = P_S_i*P_S_j*w_spatial;
                # weak/long-range bonds are low-fidelity.
                F_werner = P_product * w_spatial

                k_cross = (self.K_ENTANGLE_EM_BASE
                           * eta_factor * w_spatial * P_product)

                # Since a < b, global_id (a, ·) < (b, ·) always, so the canonical
                # min/max key is (A-side, B-side) without a per-pair comparison.
                gid_A = [d['global_id'] for d in A]
                gid_B = [d['global_id'] for d in B]
                row = {g: r for r, g in enumerate(gid_A)}
                col = {g: c for c, g in enumerate(gid_B)}

                exists = np.zeros((len(A), len(B)), dtype=bool)
                for key in existing_by_pair.get((a, b), ()):
                    r, c = row.get(key[0]), col.get(key[1])
                    if r is not None and c is not None:
                        exists[r, c] = True

                # One uniform draw per pair per step, as in the scalar loop:
                # it resolves formation where absent and dissolution where present.
                R = np.random.random((len(A), len(B)))

                p_form = 1.0 - np.exp(-k_cross * dt)
                k_diss = self.K_DISENTANGLE_BASE * (1.0 - eta_factor * P_product)
                np.maximum(k_diss, 0.0, out=k_diss)
                p_diss = 1.0 - np.exp(-k_diss * dt)

                form = (~exists) & (R < p_form)
                diss = exists & (R < p_diss)
                # Survivors refresh F (P_S drifts as coherence evolves).
                write = form | (exists & ~diss)

                # PO-7 Unit 11: a NEWLY forming bond must claim a free 31P spin at BOTH
                # endpoints from the same per-dimer ledger the intra bonds spend, or it is
                # refused (frustration). Refreshes of surviving bonds already hold their two
                # nuclei and re-claim nothing. Spin claiming is inherently sequential, so the
                # `form` mask is walked in np.nonzero (row-major) order — deterministic, and
                # with no set/dict iteration anywhere in the loop.
                for r, c in zip(*np.nonzero(write)):
                    key = (gid_A[r], gid_B[c])
                    if form[r, c] and not self._claim_cross_spins(key):
                        continue
                    self.cross_synapse_bonds[key] = float(F_werner[r, c])
                for r, c in zip(*np.nonzero(diss)):
                    key = (gid_A[r], gid_B[c])
                    if self.cross_synapse_bonds.pop(key, None) is not None:
                        self._release_cross_spins(key)

    def _step_network_provenance(self, dt: float, synapses: List, positions: np.ndarray):
        """PO-7: NETWORK-SHARED hydrolysis-event pool -> η-free cross-synapse edges.

        Only called when self.provenance_network is True (guarded by the caller in step()),
        so every state mutation and RNG draw here is off the default path. Faithful to the
        per-synapse provenance in dimer_particles.py, lifted to the network:
          - events are INPUT-PLACED: sourced at each synapse's calcium-elevated cells
            (atp_system's active-site rule), positioned in ABSOLUTE network nm;
          - a newborn dimer (in ANY synapse) claims its <=k nearest events within reach that
            still have a free slot (deterministic nearest-selection, born-with provenance);
          - the two holders of one event's two daughters BOND (shared entangled origin), with
            Werner fidelity F = P_S_i * P_S_j. If the two holders sit in DIFFERENT synapses the
            edge is cross-synapse — Fisher-inherited, needing NO η (no condensate mediation).
        Edges die by coherence death (either endpoint P_S <= 0.5) exactly as the per-synapse
        build does. Kept in self._prov_bonds so the η-gated _update_entanglement never rewrites
        them; _find_all_clusters unions them under the same Werner bound.
        """
        self._prov_time += dt  # tracker-local clock (tracker has no self.time)
        # Absolute (x,y) nm position, P_S and entanglement of every current dimer, by global_id.
        grid_center = {}
        for syn_idx, synapse in enumerate(synapses):
            ps = getattr(synapse, 'dimer_particles', None)
            if ps is not None:
                grid_center[syn_idx] = np.array(
                    [ps.grid_shape[0] * ps.dx_nm / 2.0, ps.grid_shape[1] * ps.dx_nm / 2.0])
        abs_xy, P_of, ent_of = {}, {}, {}
        for d in self.all_dimers:
            si = d['synapse_idx']
            c = grid_center.get(si)
            if c is None:
                continue
            syn_nm = np.asarray(d['synapse_pos_um'][:2], float) * 1000.0
            abs_xy[d['global_id']] = syn_nm + (np.asarray(d['dimer'].position[:2], float) - c)
            P_of[d['global_id']] = float(d['P_S'])
            ent_of[d['global_id']] = bool(d['dimer'].is_entangled)
        current_ids = set(abs_xy.keys())

        # (1) Prune prov bonds: endpoint gone OR either endpoint decohered (coherence death).
        # PO-7 fix (a): fidelity is REFRESHED from the live P_S each step, not frozen at claim
        # time. _find_all_clusters tests these values against the Werner bound, so a stale F let
        # a decohered pair keep counting as an edge. F = P_S_i * P_S_j, recomputed here.
        # PO-7 fix (b): coherence death prunes on P_S <= 0.5 (the Werner bound), matching this
        # method's docstring and the per-synapse build. The prior code tested is_entangled only —
        # the same dropped-coherence-death channel as the U16 write-once bug (07fd02a).
        for key in list(self._prov_bonds.keys()):
            a, b = key
            if (a not in current_ids or b not in current_ids
                    or not (ent_of.get(a, False) and ent_of.get(b, False))):
                self._prov_bonds.pop(key, None)
                self._prov_bond_spins.pop(key, None)
                self._release_cross_spins(key)     # PO-7 U11: hand the nuclei back
                continue
            f = float(P_of[a] * P_of[b])
            if P_of[a] <= 0.5 or P_of[b] <= 0.5:
                self._prov_bonds.pop(key, None)   # coherence death
                self._prov_bond_spins.pop(key, None)
                self._release_cross_spins(key)     # PO-7 U11
            else:
                self._prov_bonds[key] = f
        self._prov_seen &= current_ids
        for gid in [g for g in self._prov_slot_of if g not in current_ids]:
            self._prov_slot_of.pop(gid, None)

        # (2) Age out expired events (phosphates consumed within ~seconds).
        self._prov_events = [e for e in self._prov_events
                             if self._prov_time - e['t'] <= self.provenance_net_age_s
                             and e['slots_free'] > 0]

        # (3) Generate new events from each synapse's calcium field, placed in ABSOLUTE nm.
        for syn_idx, synapse in enumerate(synapses):
            ps = getattr(synapse, 'dimer_particles', None)
            c = grid_center.get(syn_idx)
            if ps is None or c is None:
                continue
            try:
                ca = synapse.calcium.get_concentration()
            except Exception:
                ca = None
            if ca is None:
                continue
            active = np.argwhere(ca > self.provenance_net_ca_threshold)
            if active.size == 0:
                continue
            n_new = np.random.poisson(self.provenance_net_event_rate * len(active) * dt)
            if n_new <= 0:
                continue
            syn_nm = np.asarray(positions[syn_idx][:2], float) * 1000.0
            pick = active[np.random.randint(0, len(active), size=int(n_new))]
            for cell in pick:
                local = np.array([(cell[0] + 0.5) * ps.dx_nm, (cell[1] + 0.5) * ps.dx_nm])
                self._prov_events.append({'id': self._prov_next_event_id,
                                          'pos': syn_nm + (local - c), 't': self._prov_time,
                                          'slots_free': self.provenance_net_event_slots,
                                          'holders': []})
                self._prov_next_event_id += 1

        # (4) Assign provenance to newly-seen dimers (born-with); claim <=k nearest events in reach.
        new_ids = sorted(g for g in current_ids if g not in self._prov_seen)
        if self._prov_events and new_ids:
            epos = np.array([e['pos'] for e in self._prov_events])
            for gid in new_ids:
                self._prov_seen.add(gid)
                dist = np.linalg.norm(epos - abs_xy[gid], axis=1)
                order = np.argsort(dist)
                claimed = 0
                for k in order:
                    if dist[k] > self.provenance_net_reach_nm:
                        break  # sorted ascending: nothing farther is reachable
                    e = self._prov_events[k]
                    if e['slots_free'] <= 0:
                        continue
                    e['slots_free'] -= 1
                    spin_here = claimed              # this dimer's spin receiving the daughter
                    e['holders'].append(gid)
                    e.setdefault('holder_spins', []).append(spin_here)
                    self._prov_slot_of.setdefault(gid, {})[e['id']] = spin_here
                    if len(e['holders']) >= 2:
                        partner = e['holders'][-2]
                        if partner != gid and partner in current_ids:
                            key = (min(gid, partner), max(gid, partner))
                            # the mediating spin pair for this shared-origin edge
                            partner_spin = e['holder_spins'][-2]
                            pair = ((partner_spin, spin_here) if partner < gid
                                    else (spin_here, partner_spin))
                            # PO-7 Unit 11: the inherited nucleus sits in a NAMED slot, so
                            # this claims `required=pair` — matching the per-synapse
                            # provenance path (dimer_particles.py:516-517). Two inheritances
                            # competing for one slot cannot both be satisfied.
                            if self._claim_cross_spins(key, required=pair):
                                self._prov_bonds[key] = float(P_of[gid] * P_of[partner])
                                self._prov_bond_spins[key] = pair
                    claimed += 1
                    if claimed >= self.provenance_net_k:
                        break
        else:
            self._prov_seen |= set(new_ids)

        # Diagnostics for the probe: cross vs intra edge split and event-pool overlap fraction.
        n_cross = sum(1 for (a, b) in self._prov_bonds if a[0] != b[0])
        multi = sum(1 for e in self._prov_events
                    if len({h[0] for h in e['holders']}) >= 2)
        claimed_evt = sum(1 for e in self._prov_events if e['holders'])
        self._prov_last_stats = {
            'n_events': len(self._prov_events),
            'n_prov_bonds': len(self._prov_bonds),
            'n_cross_bonds': n_cross,
            'n_intra_bonds': len(self._prov_bonds) - n_cross,
            'overlap_frac': (multi / claimed_evt) if claimed_evt else 0.0,
        }

    def _calculate_coupling(self, d_i: dict, d_j: dict) -> float:
        """
        Calculate effective coupling between two dimers
        
        Same synapse: use J-coupling directly
        Different synapses: J-coupling × distance decay
        """
        j_local_i = d_i['local_j']
        j_local_j = d_j['local_j']
        
        # Both need sufficient J-coupling
        if j_local_i < self.j_coupling_threshold or j_local_j < self.j_coupling_threshold:
            return 0.0
        
        j_factor = min(j_local_i, j_local_j) / 20.0  # Normalize
        
        if d_i['synapse_idx'] == d_j['synapse_idx']:
            # Same synapse - full coupling
            return j_factor
        else:
            # Different synapses - distance-dependent
            dist_um = np.linalg.norm(d_i['synapse_pos_um'] - d_j['synapse_pos_um'])
            
            # Exponential decay with distance
            # At coupling_length_um, factor = 1/e ≈ 0.37
            spatial_factor = np.exp(-dist_um / self.coupling_length_um)
            
            return j_factor * spatial_factor
    
    def get_synapse_correlation_matrix(self, synapses: List) -> np.ndarray:
        """
        Compute correlation matrix between synapses based on entanglement.
        
        Physics: C_ij = E_ij × P_S_i × P_S_j

        Quantum derivation: For two Werner states with singlet probabilities P_S_i, P_S_j,
        the inter-dimer bond fidelity undergoes independent decoherence in each environment.
        Measurement correlation scales linearly with bond fidelity = P_S_i × P_S_j.
        
        Parameters
        ----------
        synapses : List[Model6QuantumSynapse]
            All synapse models in network
            
        Returns
        -------
        np.ndarray : Correlation matrix (n_synapses × n_synapses)
        """
        n_syn = len(synapses)
        C = np.zeros((n_syn, n_syn))
        
        if not self.all_dimers:
            return C
        
        # Get mean P_singlet per synapse
        P_singlet = []
        for syn in synapses:
            if hasattr(syn, 'dimer_particles') and syn.dimer_particles.dimers:
                mean_ps = np.mean([d.singlet_probability for d in syn.dimer_particles.dimers])
            else:
                mean_ps = 0.25  # Thermal
            P_singlet.append(mean_ps)
        P_singlet = np.array(P_singlet)
        
        # Count cross-synapse bonds
        # Build lookup: stable_id -> synapse_idx
        dimer_to_synapse = {d['global_id']: d['synapse_idx'] for d in self.all_dimers}

        # Count bonds between each synapse pair
        bond_counts = np.zeros((n_syn, n_syn))
        for id_i, id_j in self.entanglement_bonds:
            syn_i = dimer_to_synapse.get(id_i)
            syn_j = dimer_to_synapse.get(id_j)
            if syn_i is not None and syn_j is not None and syn_i != syn_j:
                bond_counts[syn_i, syn_j] += 1
                bond_counts[syn_j, syn_i] += 1
        
        # Count dimers per synapse for normalization
        dimers_per_synapse = np.zeros(n_syn)
        for d in self.all_dimers:
            dimers_per_synapse[d['synapse_idx']] += 1
        
        # Build correlation matrix
        for i in range(n_syn):
            for j in range(i+1, n_syn):
                # Skip if either synapse has no coherent dimers
                if P_singlet[i] < 0.5 or P_singlet[j] < 0.5:
                    continue
                
                # E_ij = normalized bond count
                max_bonds = min(dimers_per_synapse[i], dimers_per_synapse[j])
                if max_bonds > 0:
                    E_ij = bond_counts[i, j] / max_bonds
                else:
                    E_ij = 0.0
                
                # C_ij = E_ij × P_S_i × P_S_j (product form - quantum mechanically derived)
                coherence_factor = P_singlet[i] * P_singlet[j]
                C[i, j] = E_ij * coherence_factor
                C[j, i] = C[i, j]
        
        # Clamp to valid correlation range [0, 1]
        np.fill_diagonal(C, 0.0)  # No self-correlation
        C = np.clip(C, 0.0, 1.0)

        # PSD safeguard for multivariate normal sampling
        # (rarely needed - only if bond topology is inconsistent)
        if n_syn > 1:
            eigenvalues = np.linalg.eigvalsh(C + np.eye(n_syn))  # Add identity for full correlation matrix
            if np.min(eigenvalues) < 1e-10:
                # Project to nearest PSD: clip negative eigenvalues
                eigenvalues_full, eigenvectors = np.linalg.eigh(C + np.eye(n_syn))
                eigenvalues_full = np.maximum(eigenvalues_full, 1e-10)
                C_full = eigenvectors @ np.diag(eigenvalues_full) @ eigenvectors.T
                C = C_full - np.eye(n_syn)  # Remove identity to get off-diagonal
                C = np.clip(C, 0.0, 1.0)
                np.fill_diagonal(C, 0.0)

        return C


    def get_coordination_factor(self, synapses: List) -> float:
        """
        Get overall coordination factor for network.
        
        Returns value in [0, 1] indicating degree of cross-synapse entanglement.
        """
        C = self.get_synapse_correlation_matrix(synapses)
        n = len(synapses)
        if n < 2:
            return 0.0
        
        # Sum of off-diagonal normalized by maximum possible
        off_diag_sum = np.sum(C) - np.trace(C)
        max_possible = n * (n - 1)
        
        return min(1.0, off_diag_sum / max_possible)
    
    
    
    def _find_all_clusters(self) -> List[set]:
        """
        Find all connected components using union-find.

        Returns list of sets, each containing the global_ids in one
        connected component.  Only dimers that participate in at least
        one bond are included (unbonded dimers are singletons and are
        omitted).
        """
        if not self.all_dimers:
            return []

        parent = {d['global_id']: d['global_id'] for d in self.all_dimers}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union all bonded pairs — intra by bare existence, cross by Werner bound
        bonded_ids = set()
        for id_i, id_j in self.intra_synapse_bonds_cache:
            if id_i in parent and id_j in parent:
                union(id_i, id_j)
                bonded_ids.add(id_i)
                bonded_ids.add(id_j)
        for (id_i, id_j), fidelity in self.cross_synapse_bonds.items():
            if fidelity > self.WERNER_ENTANGLEMENT_BOUND:
                if id_i in parent and id_j in parent:
                    union(id_i, id_j)
                    bonded_ids.add(id_i)
                    bonded_ids.add(id_j)
        # PO-7: η-free network-provenance edges, same Werner bound. Empty when the flag is
        # off, so this loop is a no-op on the default path (bit-identical partition).
        if self.provenance_network:
            for (id_i, id_j), fidelity in self._prov_bonds.items():
                if fidelity > self.WERNER_ENTANGLEMENT_BOUND:
                    if id_i in parent and id_j in parent:
                        union(id_i, id_j)
                        bonded_ids.add(id_i)
                        bonded_ids.add(id_j)

        # Group bonded dimers by root
        clusters = {}
        for gid in bonded_ids:
            root = find(gid)
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(gid)

        return list(clusters.values())

    def _find_largest_cluster(self) -> set:
        """
        Find largest connected component using union-find
        """
        clusters = self._find_all_clusters()
        if not clusters:
            return set()
        return max(clusters, key=len)

    def compute_entanglement_topology(self, crosscheck: bool = False):
        """
        Cheap Betti0/Betti1 of the Werner-thresholded entanglement graph.
        Betti0 == len(_find_all_clusters()); Betti1 = independent loops
        (closed entanglement paths). See entanglement_topology.py for why
        this is the honest first instrument (vs persistent-homology vineyards).
        """
        return compute_betti(
            self.all_dimers,
            self.intra_synapse_bonds_cache,
            self.cross_synapse_bonds,
            werner_bound=self.WERNER_ENTANGLEMENT_BOUND,
            crosscheck=crosscheck,
        )
    
    def _has_bond(self, id_i, id_j) -> bool:
        key = (min(id_i, id_j), max(id_i, id_j))
        return (key in self.cross_synapse_bonds
                or key in self.intra_synapse_bonds_cache)

    def _add_bond(self, id_i, id_j, weight: float = 0.0):
        # Network tracker only owns cross-synapse bonds. Intra-synapse bonds
        # are owned by DimerParticleSystem and read through.
        key = (min(id_i, id_j), max(id_i, id_j))
        self.cross_synapse_bonds[key] = weight

    def _remove_bond(self, id_i, id_j):
        key = (min(id_i, id_j), max(id_i, id_j))
        if self.cross_synapse_bonds.pop(key, None) is not None:
            self._release_cross_spins(key)        # PO-7 U11


    def perform_quantum_measurement(self, synapses: List) -> np.ndarray:
        """
        Measure entangled network per connected component.

        Each connected component of the bond graph collapses independently:
          - cluster_P_S = mean(P_S) of dimers in that component
          - Single coin flip per component: outcome = random() < cluster_P_S
          - Dimers in components that collapse to singlet are "committed"

        Returns
        -------
        np.ndarray : committed_counts[i] = number of committed dimers in
                     synapse i (absolute count, NOT fraction).
        """
        n_syn = len(synapses)

        clusters = self._find_all_clusters()
        if not clusters:
            self._last_measurement = {
                'total_dimers': len(self.all_dimers),
                'total_bonds': len(self.entanglement_bonds),
                'n_clusters_measured': 0,
                'n_clusters_singlet': 0,
                'singlet_outcomes': 0,
                'committed_counts': np.zeros(n_syn),
            }
            return np.zeros(n_syn)

        # Build global_id → dimer dict lookup
        id_to_dimer = {d['global_id']: d for d in self.all_dimers}

        committed_counts = np.zeros(n_syn)
        n_clusters_singlet = 0
        total_singlet_dimers = 0

        for cluster_ids in clusters:
            # Compute cluster P_S
            cluster_ps = np.mean([
                id_to_dimer[gid]['P_S'] for gid in cluster_ids
                if gid in id_to_dimer
            ])

            # Independent coin flip for this component
            if np.random.random() < cluster_ps:
                n_clusters_singlet += 1
                # All dimers in this component commit
                for gid in cluster_ids:
                    d = id_to_dimer.get(gid)
                    if d is not None:
                        committed_counts[d['synapse_idx']] += 1
                        total_singlet_dimers += 1

        self._last_measurement = {
            'total_dimers': len(self.all_dimers),
            'total_bonds': len(self.entanglement_bonds),
            'n_clusters_measured': len(clusters),
            'n_clusters_singlet': n_clusters_singlet,
            'singlet_outcomes': total_singlet_dimers,
            'committed_counts': committed_counts.copy(),
        }

        return committed_counts

    def perform_independent_measurement(self, synapses: List) -> np.ndarray:
        """
        Control condition: Measure all dimers INDEPENDENTLY (no bond correlation).

        Same physics for individual dimers, but bonds don't cause joint collapse.
        This is the classical control — what happens without entanglement-based
        coordination.

        Returns
        -------
        np.ndarray : committed_counts[i] = number of dimers in synapse i that
                     independently collapsed to singlet (absolute count).
        """
        n_syn = len(synapses)

        if not self.all_dimers:
            return np.zeros(n_syn)

        # Measure ALL dimers independently (bonds ignored)
        committed_counts = np.zeros(n_syn)
        total_count = np.zeros(n_syn)
        total_singlet = 0

        for d in self.all_dimers:
            syn_idx = d['synapse_idx']
            if syn_idx < n_syn:
                total_count[syn_idx] += 1
                P_S = d.get('P_S', 0.25)
                if np.random.random() < P_S:
                    committed_counts[syn_idx] += 1
                    total_singlet += 1

        # Store for diagnostics
        self._last_measurement = {
            'total_dimers': int(np.sum(total_count)),
            'total_bonds': len(self.entanglement_bonds),
            'singlet_outcomes': total_singlet,
            'committed_counts': committed_counts.copy(),
        }

        return committed_counts


class MultiSynapseNetwork:
    """
    Manages N independent synapses with realistic spatial coupling
    
    Each synapse is a full Model6 instance. They couple through:
    1. Shared dendritic voltage (electrical coupling)
    2. Quantum field summation through microtubules
    3. Network-level commitment threshold
    
    Parameters
    ----------
    n_synapses : int
        Number of synapses in the network
    spacing_um : float
        Average spacing between synapses in microns
    pattern : str
        Spatial arrangement: 'linear', 'clustered', 'distributed'
    coupling_length_um : float
        Length constant for field coupling (default 5 µm)
    commitment_threshold_dimers : float
        Network dimer threshold for commitment (default 25)
    """
    
    def __init__(self,
                 n_synapses: int = 10,
                 spacing_um: float = 2.0,
                 pattern: str = 'linear',
                 coupling_length_um: float = 5.0,
                 field_threshold_kT: float = 20.0,
                 params=None,
                 use_correlated_sampling: bool = True):  # ADD THIS
    
        self.n_synapses = n_synapses
        self.spacing_um = spacing_um
        self.pattern = pattern
        self.coupling_length_um = coupling_length_um
        self.field_threshold_kT = field_threshold_kT
        self.params = params
        self.use_correlated_sampling = use_correlated_sampling
        self.disable_auto_commitment = False

        # Generate synapse positions
        self.positions = self._generate_positions()
        
        # Compute distance matrix and coupling weights
        self.distances = self._compute_distances()
        self.coupling_weights = self._compute_coupling_weights()
        
        # Create individual synapse models (lazy initialization)
        self.synapses: List = []  # Will hold Model6 instances
        self._initialized = False
        
        # Network state
        self.network_committed = False
        self.network_commitment_level = 0.0
        self.time = 0.0

        # Network-level measurement tracking. One flag PER GATE — sharing one flag
        # between the coordinated gate and its independent control invalidated the
        # comparison (D19). Each means "measured once this reward episode"; both
        # re-arm on the falling edge of reward.
        self._coordinated_measurement_performed = False
        self._independent_measurement_performed = False

        # Network-level entanglement tracking
        self.entanglement_tracker = NetworkEntanglementTracker(
            coupling_length_um=coupling_length_um
        )
        
        # History
        self.history = {
            'time': [],
            'total_dimers': [],
            'network_field': [],
            'n_committed': [],
            'synapse_dimers': []  # List of lists
        }
        
        logger.info(f"MultiSynapseNetwork created: {n_synapses} synapses, "
                   f"{pattern} pattern, {spacing_um}µm spacing")
    
    def _generate_positions(self) -> np.ndarray:
        """Generate synapse positions based on pattern"""
        
        if self.pattern == 'linear':
            # Synapses along a straight dendrite
            positions = np.zeros((self.n_synapses, 3))
            positions[:, 0] = np.arange(self.n_synapses) * self.spacing_um
            # Small perpendicular jitter (spines don't align perfectly)
            positions[:, 1] = np.random.randn(self.n_synapses) * 0.2
            positions[:, 2] = np.random.randn(self.n_synapses) * 0.2
            
        elif self.pattern == 'clustered':
            # Synapses clustered in a small region
            positions = np.random.randn(self.n_synapses, 3) * self.spacing_um * 0.5
            
        elif self.pattern == 'distributed':
            # Synapses spread across multiple branches
            positions = np.random.uniform(
                -self.spacing_um * 3, 
                self.spacing_um * 3, 
                (self.n_synapses, 3)
            )
            
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")
        
        return positions
    
    def _compute_distances(self) -> np.ndarray:
        """Compute pairwise distances between synapses"""
        n = self.n_synapses
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                distances[i, j] = d
                distances[j, i] = d
        
        return distances
    
    def _compute_coupling_weights(self) -> np.ndarray:
        """
        Compute coupling weights based on distance
        
        Quantum fields decay exponentially along dendritic microtubules.
        Weight_ij = exp(-d_ij / λ) where λ is coupling length constant
        """
        weights = np.exp(-self.distances / self.coupling_length_um)
        # Self-coupling is 1.0
        np.fill_diagonal(weights, 1.0)
        return weights
    
    
    def initialize(self, ModelClass, base_params=None):
        """
        Initialize individual synapse models
        
        Parameters
        ----------
        ModelClass : class
            The Model6QuantumSynapse class to instantiate
        base_params : Model6Parameters, optional
            Base parameters (will be copied for each synapse)
        """
        from copy import deepcopy

        # Store base params on the network so backbone init (line 772 below)
        # and any params-dependent runtime code can see them. Previously
        # self.params stayed at constructor default = None whenever callers
        # (e.g. make_network) omitted params from the constructor call.
        # May 13 2026 fix.
        self.params = base_params

        self.synapses = []
        
        for i in range(self.n_synapses):
            # Each synapse gets its own parameters (for independent stochasticity)
            if base_params is not None:
                params = deepcopy(base_params)
            else:
                params = None
            
            # Create model instance
            model = ModelClass(params=params)
            model._network_controlled = True 
            
            # Store position info
            model._network_position = self.positions[i]
            model._network_index = i
            
            self.synapses.append(model)
        
        self._initialized = True
        logger.info(f"Initialized {self.n_synapses} Model6 instances")

        # Log backbone pump configuration
        if self.params is not None and hasattr(self.params, 'dendritic_backbone') and self.params.dendritic_backbone.enabled:
            bp = self.params.dendritic_backbone
            omega_ang = 2.0 * np.pi * bp.omega_0
            n_bar = bose_einstein_occupation(bp.omega_0)
            P_c = n_bar * hbar * omega_ang**2 / bp.Q
            logger.info(f"Backbone metabolic pump: ω₀/2π={bp.omega_0/1e6:.1f} MHz, "
                        f"Q={bp.Q:.0f}, P_c={P_c*1e15:.1f} fW, "
                        f"P_active_max={bp.p_active_max_W*1e15:.0f} fW")
    
    def disable_network_commitment(self):
        """Disable network-level field commitment for coordination experiments"""
        self.field_threshold_kT = float('inf')
    
    def configure_all(self, **kwargs):
        """Apply configuration to all synapses"""
        for synapse in self.synapses:
            for key, value in kwargs.items():
                if hasattr(synapse, key):
                    setattr(synapse, key, value)
                elif hasattr(synapse, f'set_{key}'):
                    getattr(synapse, f'set_{key}')(value)
    
    def set_microtubule_invasion(self, invaded: bool):
        """Set MT invasion state for all synapses"""
        for synapse in self.synapses:
            synapse.set_microtubule_invasion(invaded)
    
    def step(self, dt: float, stimulus: Dict) -> NetworkState:
        """
        Step all synapses and compute network state
        
        Parameters
        ----------
        dt : float
            Time step in seconds
        stimulus : dict
            Stimulus parameters (applied to all synapses)
            
        Returns
        -------
        NetworkState
            Aggregate network state
        """
        if not self._initialized:
            raise RuntimeError("Network not initialized. Call initialize() first.")
        
        # Step each synapse independently
        synapse_states = []
        
        for i, synapse in enumerate(self.synapses):
            # Each synapse gets the same stimulus
            # (In future, could have synapse-specific stimuli)
            synapse.step(dt, stimulus)
            
            # Get dimer count directly from particle system (fast)
            if hasattr(synapse, 'dimer_particles'):
                dimer_count = len(synapse.dimer_particles.dimers)
            else:
                dimer_count = 0

            state = SynapseState(
                position_um=self.positions[i],
                dimer_count=dimer_count,
                coherence=synapse.get_mean_singlet_probability() if hasattr(synapse, 'get_mean_singlet_probability') else 0.0,
                collective_field_kT=getattr(synapse, '_collective_field_kT', 0.0),
                eligibility=getattr(synapse, '_current_eligibility', 0.0),
                committed=getattr(synapse, '_camkii_committed', False),
                committed_level=getattr(synapse, '_committed_memory_level', 0.0),
                calcium_peak_uM=np.max(synapse.calcium.get_concentration()) * 1e6,
            )
            synapse_states.append(state)

        # === SHARED DENDRITIC BACKBONE FIELD ===
        # The microtubule backbone is continuous along the dendritic segment.
        # Each synapse's reverse coupling (dimer→tubulin modulation) pumps
        # the backbone toward Fröhlich condensation. Above threshold, all
        # synapses benefit from the shared coherent field.
        if self.params is not None and hasattr(self.params, 'dendritic_backbone') and self.params.dendritic_backbone.enabled:
            self._update_backbone_field()

        # Update network-level entanglement
        # Only update entanglement every 10 steps (expensive O(n²) operation)
        if not hasattr(self, '_entanglement_step_counter'):
            self._entanglement_step_counter = 0
        self._entanglement_step_counter += 1

        if self._entanglement_step_counter % 10 == 0:
            self._network_entanglement = self.entanglement_tracker.step(
                dt, self.synapses, self.positions,
                coupling_weights=getattr(self, 'coupling_weights', None)
            )
        
        # =========================================================================
        # GAP 3: COORDINATED THREE-FACTOR GATE EVALUATION
        # =========================================================================
        # When reward is present, evaluate gates with correlated sampling.
        # This ensures entangled synapses commit/fail together.
        #
        # The gate is called UNCONDITIONALLY — it performs its own `reward` check and
        # early-returns. The outer `if reward` that used to wrap this call meant the
        # gate was never invoked on a non-reward step, so the falling edge of reward
        # was never observed and the one-shot measurement latch could never re-arm.
        # That is what made the latch effectively once-per-experiment (D19).
        if self.use_correlated_sampling:
            self._evaluate_coordinated_gate(stimulus)
        else:
            self._evaluate_independent_gate(stimulus)

        if stimulus.get('reward', False):
            # Propagate synapse-level commitment to network flag
            if not self.network_committed:
                if any(getattr(s, '_camkii_committed', False) for s in self.synapses):
                    self.network_committed = True
                    state_now = self._compute_network_state(
                        [SynapseState(
                            position_um=self.positions[i],
                            dimer_count=len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0,
                            coherence=s.get_mean_singlet_probability() if hasattr(s, 'get_mean_singlet_probability') else 0.0,
                            collective_field_kT=getattr(s, '_collective_field_kT', 0.0),
                            eligibility=getattr(s, '_current_eligibility', 0.0),
                            committed=getattr(s, '_camkii_committed', False),
                            committed_level=getattr(s, '_committed_memory_level', 0.0),
                            calcium_peak_uM=np.max(s.calcium.get_concentration()) * 1e6,
                        ) for i, s in enumerate(self.synapses)]
                    )
                    self.network_commitment_level = state_now.mean_eligibility
                    logger.info(
                        f"Network COMMITTED via three-factor gate: "
                        f"field={state_now.network_field_kT:.1f} kT, "
                        f"dimers={state_now.total_dimers:.1f}, "
                        f"n_committed_synapses={state_now.n_committed}"
                    )
        # =========================================================================

        # Compute network-level quantities
        network_state = self._compute_network_state(synapse_states)
        
        # Auto-commitment disabled by default — commitment requires dopamine
        # through the three-factor gate (coordinated or independent pathway).
        # Only legacy experiments that explicitly set disable_auto_commitment=False use this.
        if not getattr(self, 'disable_auto_commitment', True):
            self._check_network_commitment(network_state)
        
        # Update time
        self.time += dt
        
        # Record history
        self._record_history(network_state)
        
        return network_state
    
    def _compute_network_state(self, synapse_states: List[SynapseState]) -> NetworkState:
        """
        Compute aggregate network state with proper field coupling
        
        The key physics:
        - Each synapse contributes its dimers to the network
        - Fields couple through microtubules with distance-dependent decay
        - Total network field determines commitment
        """
        
        # Raw totals
        dimer_counts = np.array([s.dimer_count for s in synapse_states])
        coherences = np.array([s.coherence for s in synapse_states])
        eligibilities = np.array([s.eligibility for s in synapse_states])
        
        # Simple sum of dimers (each synapse contributes independently)
        total_dimers = np.sum(dimer_counts)
        
        # Mean coherence (weighted by dimer count)
        if total_dimers > 0:
            mean_coherence = np.sum(coherences * dimer_counts) / total_dimers
        else:
            mean_coherence = 0.0
        
        # Mean eligibility
        mean_eligibility = np.mean(eligibilities)
        
        # Network field from physics (emergent, not prescribed)
        #
        # Physics basis:
        # - Single coherent dimer: U_single = 6.6 kT (Fisher 2015)
        # - Entanglement fraction: f_ent = 0.3 (only 30% form coherent network)
        # - Coherent summation: √N_ent enhancement
        # - CaMKII is local to dimers (no spatial reduction)
        #
        # U_network = U_single × √(f_ent × N_coherent)
        # For N=50, coherence=0.85: √(0.3 × 42.5) × 6.6 = 23.6 kT ✓
        
        U_single_kT = 6.6  # From Fisher 2015
    
        # Get TRUE entangled network from cross-synapse tracker
        if hasattr(self, '_network_entanglement') and self._network_entanglement:
            n_entangled = self._network_entanglement['n_entangled_network']
            n_bonds = self._network_entanglement['n_bonds']
        else:
            n_entangled = 0
            n_bonds = 0
        
        # === Q2 FIELD: From entangled dimers ===
        N_collective_threshold = 35.0
        
        if n_entangled >= N_collective_threshold:
            n_possible_pairs = n_entangled * (n_entangled - 1) // 2
            if n_possible_pairs > 0:
                f_ent_emergent = n_bonds / n_possible_pairs
            else:
                f_ent_emergent = 0.0
            effective_N = f_ent_emergent * n_entangled
            q2_field = U_single_kT * np.sqrt(effective_N)
        else:
            q2_field = 0.0
        
        # === Q1 FIELD: From tryptophan networks ===
        # Each synapse has its own Q1 field; network field is the mean
        q1_fields = [s.collective_field_kT for s in synapse_states]
        q1_field = np.mean(q1_fields) if q1_fields else 0.0
        
        # === TOTAL NETWORK FIELD ===
        # Q1 and Q2 are coupled systems - use the dominant contribution
        network_field = max(q1_field, q2_field)
        
        # Count committed synapses
        n_committed = sum(1 for s in synapse_states if s.committed)
        
        return NetworkState(
            n_synapses=self.n_synapses,
            total_dimers=total_dimers,
            network_field_kT=network_field,
            mean_coherence=mean_coherence,
            mean_eligibility=mean_eligibility,
            n_committed=n_committed,
            network_committed=self.network_committed,
            network_commitment_level=self.network_commitment_level,
            synapse_states=synapse_states
        )
    
    
    def _update_backbone_field(self):
        """Compute backbone condensation eta from metabolic drive P_met.

        The backbone's condensation ratio (eta) modulates f_coherent in
        invaded spines via cooperative robustness — NOT by injecting an
        additive field (shaft MTs are too far for near-field coupling).

        Steps:
        1. Compute per-synapse metabolic power P_met from upstream signals
        2. Spatially aggregate active component (basal excluded from sum)
        3. Threshold: eta = (r-1)/(r+1) where r = P_met_agg / P_c
        4. Pass eta to each synapse (gated on MT invasion in model6_core)
        """
        bp = self.params.dendritic_backbone

        # Critical power: P_c = n̄_s · ℏω₀² / Q  (threshold is n_ex ≥ n̄_s)
        omega_ang = 2.0 * np.pi * bp.omega_0
        n_bar = bose_einstein_occupation(bp.omega_0)
        P_c = n_bar * hbar * omega_ang**2 / bp.Q

        # Per-synapse metabolic power from upstream signals
        p_met = np.array([
            compute_metabolic_power(
                getattr(s.spine_plasticity, 'E_invasion', 0.0),
                s.calcium.channels.get_open_fraction(),
                bp.p_active_max_W,
            ) for s in self.synapses
        ])

        # Aggregate ONLY the active component; basal is per-spine, not summed
        p_active = p_met - P_BASAL_W
        p_met_agg = P_BASAL_W + self.coupling_weights @ p_active

        for i, synapse in enumerate(self.synapses):
            r = p_met_agg[i] / P_c
            eta = (r - 1.0) / (r + 1.0) if r >= 1.0 else 0.0
            synapse.set_backbone_condensation_eta(eta)


        # Throttled diagnostics (every 200 backbone updates)
        if not hasattr(self, '_backbone_diag_n'):
            self._backbone_diag_n = 0
        self._backbone_diag_n += 1
        if self._backbone_diag_n % 200 == 0:
            s = self.synapses[0]
            mt_inv = getattr(s, '_mt_invaded', False)
            r0 = p_met_agg[0] / P_c
            eta0 = (r0 - 1.0) / (r0 + 1.0) if r0 >= 1.0 else 0.0
            print(f"[backbone diag] P_met={p_met[0]*1e15:.2f}fW  "
                  f"P_agg={p_met_agg[0]*1e15:.2f}fW  P_c={P_c*1e15:.2f}fW  "
                  f"r={r0:.3f}  eta={eta0:.4f}  invaded={mt_inv}")

    def sample_correlated_eligibilities(self, rng: np.random.Generator = None) -> np.ndarray:
        """
        Sample eligibilities with quantum correlations for reward application.
        
        This is the "measurement" - entangled synapses get correlated outcomes.
        
        Returns
        -------
        np.ndarray : Sampled eligibilities for each synapse
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Get mean eligibilities from each synapse
        mean_elig = np.array([s.get_eligibility() for s in self.synapses])
        
        # Get correlation matrix from entanglement
        self.entanglement_tracker.collect_dimers(self.synapses, self.positions)
        C = self.entanglement_tracker.get_synapse_correlation_matrix(self.synapses)
        
        # Build covariance matrix
        base_var = 0.2  # Variance scale for quantum fluctuations
        cov = np.eye(self.n_synapses) * base_var
        
        for i in range(self.n_synapses):
            for j in range(i+1, self.n_synapses):
                cov[i, j] = C[i, j] * base_var
                cov[j, i] = cov[i, j]
        
        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(cov)
        if np.min(eigvals) < 0:
            cov += np.eye(self.n_synapses) * (-np.min(eigvals) + 1e-6)
        
        # Sample from multivariate normal
        sampled = rng.multivariate_normal(mean_elig, cov)
        
        return np.clip(sampled, 0, 1)


    def apply_reward_correlated(self, reward: float, learning_rate: float = 0.05) -> np.ndarray:
        """
        Apply reward with entanglement-based coordination.
        
        Correlated synapses get similar eligibility samples, producing
        coordinated weight updates without backpropagation.
        
        Parameters
        ----------
        reward : float
            Scalar reward signal (can be negative)
        learning_rate : float
            Learning rate for weight updates
            
        Returns
        -------
        np.ndarray : Weight changes for each synapse
        """
        # Sample eligibilities with correlations
        sampled_elig = self.sample_correlated_eligibilities()
        
        # Compute weight changes
        delta_w = learning_rate * reward * sampled_elig
        
        # Apply to synapses (need to implement weight storage)
        # This hooks into the CaMKII commitment mechanism
        for i, syn in enumerate(self.synapses):
            if hasattr(syn, '_committed_memory_level'):
                syn._committed_memory_level += delta_w[i]
                syn._committed_memory_level = np.clip(syn._committed_memory_level, 0, 1)
        
        return delta_w
    
    
    def _check_network_commitment(self, state: NetworkState):
        """
        Check if network crosses commitment threshold
        
        PARTLY emergent (claim corrected 2026-07-18). This block previously ended
        "No fitted parameters!". Two fitted parameters sit on this exact path:
          - `field_threshold_kT = 20.0` (~:717) — the "~20 kT thermal noise" figure,
            which carries no citation anywhere in the tree
          - `mean_eligibility > 0.3` (below) — a chosen cut with no derivation
        The ~50-coherent-dimer figure is also not independent: `n_dimer_threshold = 50`
        is declared "Fisher's prediction" in model6_parameters.py, so deriving ~50 from
        this threshold and then citing it as agreement is circular.
        What IS emergent here: the field itself, which is computed rather than set.
        """
        if self.network_committed:
            return  # Already committed, can't uncommit
        
        # Physics-based threshold: field must overcome thermal fluctuations
        if state.network_field_kT >= self.field_threshold_kT:
            if state.mean_eligibility > 0.3:
                self.network_committed = True
                # Commitment level scales with field strength above threshold
                excess = state.network_field_kT / self.field_threshold_kT
                self.network_commitment_level = min(1.0, state.mean_eligibility * excess)
                
                logger.info(f"Network COMMITTED: field={state.network_field_kT:.1f} kT "
                          f"(threshold={self.field_threshold_kT} kT), "
                          f"dimers={state.total_dimers:.1f}, "
                          f"level={self.network_commitment_level:.2f}")
    
    def step_with_coordination(self, dt: float, stimulus: dict) -> NetworkState:
        """
        Step network with coordinated three-factor gate evaluation.
        
        When reward is present, use correlated sampling to decide which
        synapses open their plasticity gates. Entangled synapses tend to
        commit together or fail together.
        
        This replaces independent gate evaluation with coordinated "measurement".
        """
        reward_present = stimulus.get('reward', False)
        
        # Step individual synapses (calcium, dimers, etc.)
        for synapse in self.synapses:
            synapse.step(dt, stimulus)
        
        # Update network entanglement tracking
        self.entanglement_tracker.collect_dimers(self.synapses, self.positions)
        # coupling_weights MUST be passed — `_update_entanglement` early-returns without
        # it (~:279), so omitting it means ZERO cross-synapse bonds form on this entry
        # point. This method is named `step_with_coordination` and the class docstring
        # advertises "true network-level entanglement tracking"; without this argument it
        # delivered neither. (2026-07-18 substrate audit, drift item 16.)
        ent_metrics = self.entanglement_tracker.step(
            dt, self.synapses, self.positions,
            coupling_weights=getattr(self, 'coupling_weights', None))
        self._network_entanglement = ent_metrics
        
        # === COORDINATED THREE-FACTOR GATE ===
        # Unconditional — the gate checks `reward` itself, and needs to SEE the
        # non-reward steps in order to re-arm its one-shot latch (D19).
        self._evaluate_coordinated_gate(stimulus)
        
        # Build network state for history and return
        synapse_states = []
        for i, synapse in enumerate(self.synapses):
            if hasattr(synapse, 'dimer_particles'):
                dimer_count = len(synapse.dimer_particles.dimers)
            else:
                dimer_count = 0
            ss = SynapseState(
                position_um=self.positions[i],
                dimer_count=dimer_count,
                coherence=synapse.get_mean_singlet_probability() if hasattr(synapse, 'get_mean_singlet_probability') else 0.0,
                collective_field_kT=getattr(synapse, '_collective_field_kT', 0.0),
                eligibility=getattr(synapse, '_current_eligibility', 0.0),
                committed=getattr(synapse, '_camkii_committed', False),
                committed_level=getattr(synapse, '_committed_memory_level', 0.0),
                calcium_peak_uM=np.max(synapse.calcium.get_concentration()) * 1e6,
            )
            synapse_states.append(ss)
        network_state = self._compute_network_state(synapse_states)
        
        self.time += dt
        self._record_history(network_state)
        
        return network_state


    # Hill function parameters for count-dependent plasticity drive
    HILL_N = 4        # Cooperativity (matches CaMKII ultrasensitive activation)
    HILL_K_HALF = 20  # Half-max at 20 committed dimers per synapse

    def _committed_count_to_drive(self, count: float) -> float:
        """
        Convert committed dimer count to plasticity drive via Hill function.

        drive = count^n / (K_half^n + count^n)

        Produces 0-1 output with ultrasensitive (sigmoidal) dependence on
        the number of committed dimers.  K_half=20 means a synapse needs
        ~20 committed dimers for half-max drive; n=4 gives sharp onset.
        """
        n = self.HILL_N
        k = self.HILL_K_HALF
        return float(count ** n / (k ** n + count ** n))

    def _evaluate_coordinated_gate(self, stimulus: dict):
        """
        Evaluate three-factor gate with QUANTUM MEASUREMENT.

        Physics: When reward arrives, each connected component of the
        entanglement bond graph is measured independently.
        - Dimers within a component collapse TOGETHER (perfect correlation)
        - Different components get independent coin flips
        - Per-synapse committed_count = number of dimers that collapsed to singlet
        - Plasticity drive: THIS GATE DOES NOT USE THE HILL FUNCTION (corrected
          2026-07-18). The line previously read "Plasticity drive = Hill(committed_count)
          with n=4, K_half=20". `_committed_count_to_drive` DOES exist (~:1319) and IS
          live — but its ONLY caller is `_evaluate_independent_gate` (~:1561), i.e. the
          CONTROL condition. On THIS path the May-12 DDSC rewire replaced direct
          commitment with a measurement token consumed by CaMKII
          (`model6_core.py` molecular_memory branch), and the docstring was never
          updated. Here the gate is a bare `count > 0`: a SINGLE committed dimer opens
          it, and `count` grades nothing.

        The gate has three factors:
        1. Quantum: committed_count > 0 (at least some committed dimers)
        2. Dopamine: reward signal present (classical)
        3. Calcium: postsynaptic activity (classical)
        """
        dopamine_present = stimulus.get('reward', False)
        if not dopamine_present:
            # Reward episode over — RE-ARM. The latch below means "measure once per
            # reward episode", NOT once per network lifetime. It was never re-armed,
            # so across a multi-trial run the measurement fired on trial 0 and every
            # later reward returned at the latch (research log D19: observed 1 call in
            # trial 0, 0 in trials 1-2, while spine volume kept climbing off a stale
            # gate flag). Re-arming on the falling edge fixes this for every driver,
            # including ones that never think to reset it.
            self._coordinated_measurement_performed = False
            return

        if self._coordinated_measurement_performed:
            return
        self._coordinated_measurement_performed = True

        # Ensure dimer registry is current
        self.entanglement_tracker.collect_dimers(self.synapses, self.positions)

        # === QUANTUM MEASUREMENT (per connected component) ===
        committed_counts = self.entanglement_tracker.perform_quantum_measurement(
            self.synapses
        )

        # Diagnostic output
        measurement = getattr(self.entanglement_tracker, '_last_measurement', {})
        n_bonds = measurement.get('total_bonds', 0)
        n_dimers = measurement.get('total_dimers', 0)
        n_singlet = measurement.get('singlet_outcomes', 0)
        n_clust = measurement.get('n_clusters_measured', 0)
        n_clust_s = measurement.get('n_clusters_singlet', 0)

        logger.info(f"QUANTUM MEASUREMENT: {n_dimers} dimers, {n_bonds} bonds, "
                    f"{n_clust} clusters ({n_clust_s} singlet), "
                    f"{n_singlet} committed dimers")
        logger.info(f"  Committed counts: {[f'{c:.0f}' for c in committed_counts]}")

        # === THREE-FACTOR GATE EVALUATION ===
        for i, syn in enumerate(self.synapses):
            count = committed_counts[i]

            # Calcium factor
            calcium_uM = getattr(syn, '_peak_calcium_uM', 0.0)
            calcium_elevated = calcium_uM > 0.5  # µM threshold

            # Three-factor AND gate
            gate_open = (
                count > 0 and          # Quantum: at least some committed dimers
                dopamine_present and    # Reward signal
                calcium_elevated        # Postsynaptic activity
            )

            syn._plasticity_gate = gate_open

            # Measurement performed but commitment is NOT immediate.
            # Record measurement outcome — dissolved dimers will return
            # calcium, driving CaMKII cascade (DDSC mechanism).
            # _camkii_committed will be set by CaMKII molecular_memory
            # reaching threshold in the synapse's step().
            if gate_open and not getattr(syn, '_camkii_committed', False):
                syn._measurement_gate_opened = True
                syn._measurement_time = self.time
                syn._measurement_dimer_count = int(count)

        # === POST-MEASUREMENT: Entanglement is consumed ===
        # Measurement collapses the quantum state
        if hasattr(self, '_apply_measurement_collapse'):
            self._apply_measurement_collapse()


    def _apply_measurement_collapse(self):
        """
        Apply measurement collapse to entanglement network.
        
        Physics: Quantum measurement destroys superposition.
        After reward-triggered "collapse", cross-synapse entanglement
        bonds are weakened or broken.
        
        This is a one-time effect when reward is applied.
        """
        # Reduce bond strengths significantly (measurement effect)
        # Don't completely destroy - some residual correlation remains
        # DEAD (flagged 2026-07-18): `collapse_factor` is assigned and NEVER READ. The
        # claim "bonds reduced to 30% strength" describes a mechanism that does not run.
        # What actually happens is below: discordant bonds are removed with probability
        # 0.8; concordant bonds are untouched. Left assigned rather than deleted so the
        # audit trail is visible; delete it with the surrounding rewrite.
        collapse_factor = 0.3  # UNUSED — see note above
        
        tracker = self.entanglement_tracker
        
        # For bonds, we can't easily modify strength (they're in a set)
        # Instead, we accelerate decay by marking measurement time
        self._last_measurement_time = self.time
        
        # Alternative: remove some bonds probabilistically
        # Bonds between synapses that BOTH committed remain (correlated collapse)
        # Bonds where one committed and one didn't break (decoherence)
        
        committed_synapses = set()
        for i, syn in enumerate(self.synapses):
            if getattr(syn, '_camkii_committed', False):
                committed_synapses.add(i)
        
        if not hasattr(tracker, 'entanglement_bonds'):
            return
        
        # Build dimer-to-synapse lookup
        if not tracker.all_dimers:
            return
        
        dimer_to_synapse = {d['global_id']: d['synapse_idx'] for d in tracker.all_dimers}
        
        # Find bonds to break
        bonds_to_remove = set()
        for bond in tracker.cross_synapse_bonds:
            id_i, id_j = bond
            syn_i = dimer_to_synapse.get(id_i, -1)
            syn_j = dimer_to_synapse.get(id_j, -1)
            
            if syn_i < 0 or syn_j < 0:
                continue
            
            # Break bond if commitment was discordant (one yes, one no)
            i_committed = syn_i in committed_synapses
            j_committed = syn_j in committed_synapses
            
            if i_committed != j_committed:
                # Discordant - bond breaks with high probability
                if np.random.random() < 0.8:
                    bonds_to_remove.add(bond)
            else:
                # Concordant - bond weakens but persists
                # (handled by accelerated decay)
                pass
        
        # Remove discordant bonds
        for bond in bonds_to_remove:
            if tracker.cross_synapse_bonds.pop(bond, None) is not None:
                tracker._release_cross_spins(bond)   # PO-7 U11: hand the nuclei back
    
    
    def _evaluate_independent_gate(self, stimulus: dict):
        """
        Control condition: Measure all dimers INDEPENDENTLY.

        Same quantum physics for individual dimers (P_S determines collapse
        probability), but bonds DON'T cause joint collapse.  Returns
        committed dimer counts per synapse, converted to plasticity drive
        via the Hill function `_committed_count_to_drive` (~:1319). NOTE (2026-07-18):
        "the same ... as the coordinated gate" was BACKWARDS — the coordinated gate does
        NOT use it (it defers to CaMKII via the DDSC rewire). This control gate is the
        function's only caller, so the control and the experimental condition convert
        committed_count to drive by DIFFERENT mechanisms. That asymmetry is worth
        knowing before the two are compared.

        Key difference from coordinated gate:
        - Coordinated: Connected components collapse together (correlation = 1)
        - Independent: All dimers collapse independently (correlation = 0)
        """
        dopamine_present = stimulus.get('reward', False)
        if not dopamine_present:
            # Re-arm — see _evaluate_coordinated_gate. SEPARATE flag: this gate is the
            # CONTROL condition, and it previously shared `_network_measurement_performed`
            # with the coordinated gate, so whichever ran first locked the other out for
            # the network's lifetime. That made the coordinated-vs-independent comparison
            # — the control for whether the correlated partition matters at all — invalid
            # by construction (research log D19).
            self._independent_measurement_performed = False
            return

        if self._independent_measurement_performed:
            return
        self._independent_measurement_performed = True

        # Ensure dimer registry is current
        self.entanglement_tracker.collect_dimers(self.synapses, self.positions)

        # === INDEPENDENT MEASUREMENT ===
        # All dimers collapse independently (bonds ignored)
        committed_counts = self.entanglement_tracker.perform_independent_measurement(
            self.synapses
        )

        # Diagnostic output
        measurement = getattr(self.entanglement_tracker, '_last_measurement', {})
        n_dimers = measurement.get('total_dimers', 0)
        n_singlet = measurement.get('singlet_outcomes', 0)

        logger.info(f"INDEPENDENT MEASUREMENT: {n_dimers} dimers, "
                    f"{n_singlet} singlet outcomes (bonds ignored)")
        logger.info(f"  Committed counts: {[f'{c:.0f}' for c in committed_counts]}")

        # === THREE-FACTOR GATE EVALUATION ===
        for i, syn in enumerate(self.synapses):
            count = committed_counts[i]

            # Calcium factor
            calcium_uM = getattr(syn, '_peak_calcium_uM', 0.0)
            calcium_elevated = calcium_uM > 0.5

            # Three-factor AND gate
            gate_open = (
                count > 0 and          # At least some committed dimers
                dopamine_present and    # Reward signal
                calcium_elevated        # Postsynaptic activity
            )

            syn._plasticity_gate = gate_open

            # Commitment with count-dependent graded drive
            if gate_open and not getattr(syn, '_camkii_committed', False):
                syn._camkii_committed = True
                syn._commitment_time = self.time
                syn._committed_memory_level = self._committed_count_to_drive(count)
                syn._committed_dimer_count = int(count)
    
    def _record_history(self, state: NetworkState):
        """Record network state to history"""
        self.history['time'].append(self.time)
        self.history['total_dimers'].append(state.total_dimers)
        self.history['network_field'].append(state.network_field_kT)
        self.history['n_committed'].append(state.n_committed)
        self.history['synapse_dimers'].append(
            [s.dimer_count for s in state.synapse_states]
        )
    
    def get_experimental_metrics(self) -> Dict:
        """Get metrics for experimental comparison"""
        
        if not self.synapses:
            return {}
        
        # Collect from all synapses
        dimer_counts = [getattr(s, '_previous_dimer_count', 0) for s in self.synapses]
        coherences = [getattr(s, '_previous_coherence', 0) for s in self.synapses]
        fields = [getattr(s, '_collective_field_kT', 0) for s in self.synapses]
        
        # Cross-synapse entanglement metrics (THE KEY SPATIAL EFFECT)
        ent = self._network_entanglement if hasattr(self, '_network_entanglement') and self._network_entanglement else {}
        n_entangled_network = ent.get('n_entangled_network', 0)
        n_total_bonds = ent.get('n_bonds', 0)
        
        # Count within vs cross-synapse bonds
        within_bonds = 0
        cross_bonds = 0
        if hasattr(self, 'entanglement_tracker') and self.entanglement_tracker.all_dimers:
            dimer_synapse = {d['global_id']: d['synapse_idx'] for d in self.entanglement_tracker.all_dimers}
            for bond in self.entanglement_tracker.entanglement_bonds:
                id1, id2 = bond
                if dimer_synapse.get(id1) == dimer_synapse.get(id2):
                    within_bonds += 1
                else:
                    cross_bonds += 1
        
        # Q2 field from entangled dimer network
        U_single_kT = 6.6
        q2_field = U_single_kT * np.sqrt(n_entangled_network) if n_entangled_network > 0 else 0.0

        # Cheap topology of the Werner-thresholded entanglement graph.
        # betti1 = independent loops (closed entanglement paths). betti0 == n
        # connected components. See entanglement_topology.py.
        # Raw whole-graph Betti is dominated by within-spine clique-fill; the
        # synapse-quotient Betti is the honest cross-synapse lens (loops that
        # span synapses). Surface BOTH, clearly labelled.
        topo_betti0 = topo_betti1 = 0
        topo_component_sizes: List[int] = []
        cross_betti0 = cross_betti1 = 0
        if hasattr(self, 'entanglement_tracker') and self.entanglement_tracker.all_dimers:
            tr = self.entanglement_tracker
            _topo = tr.compute_entanglement_topology()
            topo_betti0 = _topo.betti0
            topo_betti1 = _topo.betti1
            topo_component_sizes = _topo.component_sizes
            _cross = compute_synapse_quotient_betti(
                tr.all_dimers, tr.cross_synapse_bonds,
                werner_bound=tr.WERNER_ENTANGLEMENT_BOUND,
            )
            cross_betti0 = _cross.betti0   # connected synapse-clusters
            cross_betti1 = _cross.betti1   # closed cross-synapse loops (the signal)

        return {
            'n_synapses': self.n_synapses,
            'betti0_raw': topo_betti0,
            'betti1_raw': topo_betti1,
            'raw_component_sizes': topo_component_sizes,
            'betti0_cross': cross_betti0,
            'betti1_cross': cross_betti1,
            'total_dimers': sum(dimer_counts),
            'mean_dimers_per_synapse': np.mean(dimer_counts),
            'std_dimers_per_synapse': np.std(dimer_counts),
            'mean_coherence': np.mean(coherences),
            'mean_field_kT': np.mean(fields),  # Q1 field (local tryptophan)
            'q2_field_kT': q2_field,  # Q2 field (entangled dimer network)
            'n_entangled_network': n_entangled_network,
            'within_synapse_bonds': within_bonds,
            'cross_synapse_bonds': cross_bonds,
            'total_bonds': n_total_bonds,
            'n_clusters': sum(
                s.dimer_particles.get_network_metrics().get('n_clusters', 0)
                for s in self.synapses if hasattr(s, 'dimer_particles')
            ),
            'mean_connectivity': (n_total_bonds / max(sum(dimer_counts), 1e-12)),
            'network_committed': self.network_committed,
            'network_commitment_level': self.network_commitment_level,
            'pattern': self.pattern,
            'spacing_um': self.spacing_um,
            'coupling_length_um': self.coupling_length_um
        }
    
    
    def set_coordination_mode(self, enable: bool = True):
        """
        Configure network for coordination experiments.
        
        When enabled:
        - Disables automatic network commitment (field threshold check)
        - Commitment only happens via three-factor gate at reward time
        - Correlated sampling enabled for entangled synapses
        
        When disabled:
        - Normal operation with field-threshold commitment
        
        Parameters
        ----------
        enable : bool
            True for coordination experiments, False for normal operation
        """
        self.disable_auto_commitment = True
        self.use_correlated_sampling = enable
        
        if enable:
            logger.info("Coordination mode ENABLED: commitment only via reward gate")
        else:
            logger.info("Coordination mode DISABLED: normal operation")
    
    
    def reset(self):
        """Reset network state (but keep synapses and coordination mode)"""
        self.network_committed = False
        self.network_commitment_level = 0.0
        self.time = 0.0
        # Note: disable_auto_commitment and use_correlated_sampling are preserved
        self.history = {
            'time': [],
            'total_dimers': [],
            'network_field': [],
            'n_committed': [],
            'synapse_dimers': []
        }
        
        # Reset individual synapses
        for synapse in self.synapses:
            synapse._camkii_committed = False
            synapse._committed_memory_level = 0.0
            synapse._measurement_performed = False
            # Was missing: reset() left the measurement token set, so a "reset"
            # network could still commit off a pre-reset measurement (D19).
            synapse._measurement_gate_opened = False

        # Reset network state
        self._network_state = {
            'committed': False,
            'commitment_level': 0.0
        }

           # Network state
        self.network_committed = False
        self.network_commitment_level = 0.0
        self.time = 0.0

        # Network-level measurement tracking
        self._coordinated_measurement_performed = False
        self._independent_measurement_performed = False


# =============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# =============================================================================

def create_network_from_condition(condition, ModelClass, params=None):
    """
    Factory function to create MultiSynapseNetwork from ExperimentCondition
    
    Parameters
    ----------
    condition : ExperimentCondition
        Experiment condition with n_synapses, spatial_pattern, etc.
    ModelClass : class
        Model6QuantumSynapse class
    params : Model6Parameters, optional
        Base parameters
        
    Returns
    -------
    MultiSynapseNetwork
        Configured network
    """
    network = MultiSynapseNetwork(
        n_synapses=condition.n_synapses,
        spacing_um=condition.synapse_spacing_um,
        pattern=condition.spatial_pattern,
        commitment_threshold_dimers=25.0  # From theory
    )
    
    network.initialize(ModelClass, params)
    
    # Apply condition-specific settings
    network.set_microtubule_invasion(condition.mt_invaded)
    
    return network


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-SYNAPSE NETWORK VALIDATION")
    print("=" * 70)
    
    # Test network creation
    print("\n1. Creating network with 10 synapses...")
    network = MultiSynapseNetwork(
        n_synapses=10,
        spacing_um=2.0,
        pattern='linear'
    )
    
    print(f"   Positions shape: {network.positions.shape}")
    print(f"   Distance matrix shape: {network.distances.shape}")
    print(f"   Coupling weights shape: {network.coupling_weights.shape}")
    
    print("\n2. Synapse positions (µm):")
    for i, pos in enumerate(network.positions):
        print(f"   Synapse {i}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    print("\n3. Distance statistics:")
    distances = network.distances[np.triu_indices(10, k=1)]
    print(f"   Mean distance: {np.mean(distances):.2f} µm")
    print(f"   Min distance: {np.min(distances):.2f} µm")
    print(f"   Max distance: {np.max(distances):.2f} µm")
    
    print("\n4. Coupling statistics:")
    weights = network.coupling_weights[np.triu_indices(10, k=1)]
    print(f"   Mean coupling: {np.mean(weights):.3f}")
    print(f"   Min coupling: {np.min(weights):.3f}")
    print(f"   Max coupling: {np.max(weights):.3f}")
    
    print("\n5. Testing different patterns:")
    for pattern in ['linear', 'clustered', 'distributed']:
        net = MultiSynapseNetwork(n_synapses=10, pattern=pattern)
        dist = net.distances[np.triu_indices(10, k=1)]
        coup = net.coupling_weights[np.triu_indices(10, k=1)]
        print(f"   {pattern:12s}: mean_dist={np.mean(dist):.2f}µm, "
              f"mean_coupling={np.mean(coup):.3f}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nTo use with Model6:")
    print("  from multi_synapse_network import MultiSynapseNetwork")
    print("  from model6_core import Model6QuantumSynapse")
    print("  ")
    print("  network = MultiSynapseNetwork(n_synapses=10)")
    print("  network.initialize(Model6QuantumSynapse, params)")
    print("  network.set_microtubule_invasion(True)")
    print("  ")
    print("  for t in range(n_steps):")
    print("      state = network.step(dt, {'voltage': -0.01, 'reward': False})")
    print("      print(f'Total dimers: {state.total_dimers:.1f}')")