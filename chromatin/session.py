# chromatin/session.py
from __future__ import annotations
import time
import numpy as np
from . import analysis as an
from .physics import A_TYPE, B_TYPE, MIN_LOOP_SPAN, Polymer, SimParams, SimulationUnstable
from . import widgets
from .constants import *

class Session:
    """One playthrough of one level."""

    def __init__(self, level, seed: int, two_player: bool, rounds: int, turn_seconds: int,
                hard: bool = False):
        self.level = level
        self.seed = seed
        self.two_player = two_player
        self.rounds = rounds
        self.turn_seconds = turn_seconds
        self.hard = hard

        self.target = None
        self.poly: Polymer | None = None
        self.mode = "loop"
        self.map_mode = "split"

        self.round = 1
        self.turn = P_LOOP
        self.turn_start = 0.0
        self.best = [0.0, 0.0]
        self.last = [0.0, 0.0]
        self.history: list[list[float]] = [[], []]

        self.P_live: np.ndarray | None = None
        self.C_live: np.ndarray | None = None
        self.e1_live: np.ndarray | None = None
        self.rep_live: an.Report | None = None
        self.rep: an.Report | None = None
        self.best_total = 0.0
        self.measurements = 0
        self.sim_time = 0.0
        self.paused = False

    # -------------------------------------------------------------- helpers
    def begin(self, target):
        self.target = target
        n = target.n
        self.poly = Polymer(n, SimParams(), seed=self.seed * 31 + 7)
        self.P_live = self.poly.contacts()
        self.refresh_live(force=True)
        self.turn_start = time.time()

    def allowed(self, kind: str) -> bool:
        if not self.two_player:
            return True
        return (kind == "loop") == (self.turn == P_LOOP)

    def refresh_live(self, force: bool = False):
        p = self.poly
        oe, C, e1 = an.pipeline(self.P_live, p.types.astype(float))
        self.C_live, self.e1_live = C, e1
        t = self.target
        self.rep_live = an.evaluate(self.P_live, t.P, C, t.C, e1, t.e1,
                                    list(p.loops), list(t.loops))

    def time_left(self) -> float | None:
        if not self.turn_seconds:
            return None
        return max(0.0, self.turn_seconds - (time.time() - self.turn_start))

class LabSession:
    """Chromatin MiNI-Lab: no target, no scoring -- just a live polymer whose
    every force-field parameter and bead count you can change on the fly."""

    def __init__(self, n: int, seed: int):
        self.seed = seed
        self.params = SimParams()
        self.poly = Polymer(n, self.params, seed=seed)

        rng = np.random.default_rng(seed)
        t = np.empty(n, dtype=np.int8)
        i, cur = 0, A_TYPE
        while i < n:
            L = int(rng.integers(3, 7))
            t[i:i + L] = cur
            cur = -cur
            i += L
        loops = [(2, min(n - 3, max(6, n // 2)))] if n >= 12 else []
        self.poly.load_config(t, loops)

        self.paused = False
        self.paint: int | None = None
        self.P_live = self.poly.contacts()
        self.C_live = None
        self.e1_live = None
        self.scale: widgets.Scale | None = None
        self._frame = 0

        # Instability tracking. `unstable` gates further stepping once the
        # integrator has diverged; `unstable_reason` carries a short,
        # human-readable diagnosis for the menu-screen message.
        self.unstable = False
        self.unstable_reason = ""

        self._refresh(rebuild_scale=True)

    def _refresh(self, rebuild_scale: bool = False) -> None:
        oe, C, e1 = an.pipeline(self.P_live, self.poly.types.astype(float))
        self.C_live, self.e1_live = C, e1
        # Rescale the colour range from the live map itself -- this is what
        # makes the heatmaps visibly "breathe" as you drag a slider. Throttled
        # to every few frames so the colours don't flicker every tick.
        if rebuild_scale or (self._frame % 6 == 0):
            self.scale = widgets.Scale(self.P_live, C)

    def _diagnose_instability(self) -> str:
        """Best-effort guess at *why* it diverged, so the menu message is
        actually actionable rather than just 'something broke'."""
        p = self.params
        ratio = p.dt * p.k_bond / max(1e-9, p.gamma)
        if ratio > 0.5:
            return (f"bond stiffness too high for dt/friction "
                    f"(dt·k_bond/γ = {ratio:.2f}, keep it under ~0.5) -- "
                    f"lower k_bond, raise γ, or lower dt")
        if p.gamma < 0.3:
            return f"friction γ = {p.gamma:.2f} is too low for a stable integrator"
        if p.k_angle > 60:
            return f"bending stiffness k_angle = {p.k_angle:.1f} is extreme"
        if p.ev_eps > 120 and p.ev_rc > 1.6:
            return "excluded volume is both very tall and very wide -- beads got squeezed out"
        return "positions diverged (NaN/Inf) -- try reverting your last parameter change"

    def step(self) -> None:
        if self.paused or self.unstable:
            return
        try:
            self.poly.step(self.poly.p.steps_per_frame)
            if not np.all(np.isfinite(self.poly.pos)):
                raise SimulationUnstable("positions diverged (NaN/Inf)")
            c = self.poly.contacts()
            if not np.all(np.isfinite(c)):
                raise SimulationUnstable("contact map diverged (NaN/Inf)")
            self.P_live = 0.985 * self.P_live + 0.015 * c
            self._frame += 1
            self._refresh()
        except SimulationUnstable as e:
            self.unstable = True
            self.unstable_reason = f"{e}  --  {self._diagnose_instability()}"
        except Exception as e:
            self.unstable = True
            self.unstable_reason = f"unexpected error: {e}"

    def apply_param(self, key: str, value) -> None:
        import dataclasses
        self.params = dataclasses.replace(self.params, **{key: value})
        self.poly.set_params(self.params)

    def randomize_all(self) -> None:
        """Reroll compartments and loops only. Force-field parameters are left
        untouched, so the current physics stays fixed while you explore
        different starting configurations under it."""
        n = self.poly.n
        rng = np.random.default_rng(np.random.default_rng().integers(1, 999999))

        # --- random compartment blocks
        types = np.empty(n, dtype=np.int8)
        cur = int(rng.choice([A_TYPE, B_TYPE]))
        i = 0
        while i < n:
            L = int(rng.integers(3, 8))
            types[i:i + L] = cur
            cur = -cur
            i += L
        if len(np.unique(types)) == 1:
            types[n // 2:] = -types[0]

        # --- random loops, a handful, non-overlapping-ish
        n_loops = int(rng.integers(1, max(2, n // 6)))
        loops: list[tuple[int, int]] = []
        tries = 0
        while len(loops) < n_loops and tries < 200:
            tries += 1
            i0 = int(rng.integers(0, n - MIN_LOOP_SPAN - 1))
            span = int(rng.integers(MIN_LOOP_SPAN, max(MIN_LOOP_SPAN + 1, n // 2)))
            j0 = i0 + span
            if j0 >= n:
                continue
            if any(max(abs(i0 - a), abs(j0 - b)) < 2 for a, b in loops):
                continue
            loops.append((i0, j0))
        loops.sort()

        self.poly.load_config(types, loops)
        self._refresh(rebuild_scale=True)

    def set_bead_count(self, new_n: int) -> None:
        from .physics import N_MIN, N_MAX
        new_n = int(np.clip(new_n, N_MIN, N_MAX))
        if new_n == self.poly.n:
            return
        old_types, old_loops = self.poly.clone_config()
        old_n = self.poly.n

        if new_n < old_n:
            types = old_types[:new_n].copy()
            # any loop touching a removed bead is dropped -- this is the
            # "if he removes some beads, the forces that used them go too"
            # behaviour you asked for.
            loops = [(i, j) for (i, j) in old_loops if j < new_n]
        else:
            fill = old_types[-1] if old_n else A_TYPE
            types = np.concatenate([old_types, np.full(new_n - old_n, fill, dtype=np.int8)])
            loops = list(old_loops)

        self.poly = Polymer(new_n, self.params, seed=self.seed)
        self.poly.load_config(types, loops)
        self.P_live = self.poly.contacts()
        # A fresh Polymer is unlikely to inherit instability, but resetting
        # here means changing bead count is also how a user can "escape" a
        # bad parameter state without leaving the Lab entirely.
        self.unstable = False
        self.unstable_reason = ""
        self._refresh(rebuild_scale=True)