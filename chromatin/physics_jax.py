"""JAX-accelerated stepping engine, used everywhere the game steps a polymer:
the live per-frame Session and Lab, the menu's decorative background chain,
and level-target generation (levels.py) -- one engine, same physics, same
results everywhere. physics.py (NumPy) stays only as the correctness
reference (see test_physics_jax.py) and the automatic fallback when jax
isn't installed. This module re-expresses exactly the same six force terms
in jax.numpy and fuses `steps_per_frame` Euler-Maruyama updates into ONE
compiled call via `lax.scan`, instead of `steps_per_frame` separate Python
calls into NumPy. That Python/dispatch overhead -- not the FLOPs themselves,
which are tiny at this bead count -- is what was costing ~0.2-1ms per call
even for a handful of microseconds of real math, and it was being paid 60
times per rendered frame.

Loops are packed into a fixed-size (MAX_LOOPS) padded array so tying or
untying a loop never changes array shapes and never forces a JIT recompile
mid-game. Force-field parameters (sliders in the Lab) are passed as plain
traced values too, for the same reason: only bead count N and the number of
integration steps per call are "static" (shape-affecting) and trigger a
(cheap, one-off, cached) recompile when they change.

Falls back to nothing on its own -- `FastPolymer` in physics.py decides at
runtime whether to use this module or the plain NumPy `Polymer.step`.
"""
from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)   # match NumPy's float64 exactly

    # JAX already prefers a GPU (CUDA/Metal/ROCm) backend on its own when one
    # is installed and present -- but a half-working driver can make backend
    # init raise instead of just skipping to CPU. So confirm the chosen
    # backend actually runs one op, and force plain CPU if it doesn't.
    try:
        jnp.zeros(1).block_until_ready()
    except Exception:
        jax.config.update("jax_platform_name", "cpu")
        jnp.zeros(1).block_until_ready()

    JAX_BACKEND = jax.default_backend()   # "gpu" / "cpu" / "tpu", for diagnostics
    print(f"[chromatin] physics engine: JAX on {JAX_BACKEND.upper()} ({jax.devices()[0]})")
    HAVE_JAX = True
except Exception:                                # pragma: no cover - jax optional
    HAVE_JAX = False
    JAX_BACKEND = None

MAX_LOOPS = 200   # >= N_MAX; loop count never reaches this in practice


# --------------------------------------------------------------- packing
def pack_loops(loops, max_loops: int = MAX_LOOPS):
    """Variable-length loop list -> fixed-size (li, lj, mask) numpy triple."""
    k = min(len(loops), max_loops)
    li = np.zeros(max_loops, dtype=np.int32)
    lj = np.zeros(max_loops, dtype=np.int32)
    mask = np.zeros(max_loops, dtype=np.float64)
    for idx in range(k):
        a, b = loops[idx]
        li[idx], lj[idx] = a, b
        mask[idx] = 1.0
    return li, lj, mask


# Fixed field order shared by pack_params() and the unpack inside _forces_core.
_PARAM_FIELDS = ("dt", "gamma", "kT", "b0", "k_bond", "k_angle", "k_loop",
                  "k_grab", "ev_eps", "ev_rc", "sigma", "eps_AA", "eps_BB",
                  "eps_AB", "eps_domain", "k_wall")


def pack_params(p) -> np.ndarray:
    return np.array([getattr(p, f) for f in _PARAM_FIELDS], dtype=np.float64)


if HAVE_JAX:
    # ------------------------------------------------------------ eps matrix
    def _eps_matrix(types, li, lj, mask, eps_AA, eps_BB, eps_AB, eps_domain, sep):
        """Same rule as Polymer.eps_matrix: base epsilon by type pair, plus
        eps_domain once per loop whose [i, j] span covers both beads."""
        is_a = types > 0
        aa = is_a[:, None] & is_a[None, :]
        bb = (~is_a)[:, None] & (~is_a)[None, :]
        eps = jnp.where(aa, eps_AA, jnp.where(bb, eps_BB, eps_AB))

        idx = jnp.arange(types.shape[0])
        in_span = ((idx[None, :] >= li[:, None]) &
                   (idx[None, :] <= lj[:, None])).astype(eps.dtype)       # (MAX_LOOPS, N)
        dom = jnp.einsum('ka,kb,k->ab', in_span, in_span, mask)
        eps = eps + eps_domain * dom
        return jnp.where(sep < 2, 0.0, eps)

    # ---------------------------------------------------------------- forces
    def _forces(pos, b0, k_bond, k_angle, li, lj, mask, k_loop,
                ev_eps, ev_rc, eps, k_wall, box, sep, ev_mask, eye,
                is_grabbed, grab_target, k_grab, grab_active, sigma):
        f = jnp.zeros_like(pos)

        # 1 -- backbone
        dv = pos[1:] - pos[:-1]
        dist = jnp.linalg.norm(dv, axis=1) + 1e-12
        mag = k_bond * (dist - b0)
        fb = (mag / dist)[:, None] * dv
        f = f.at[:-1].add(fb)
        f = f.at[1:].add(-fb)

        # 2 -- bending (k_angle == 0 just zeroes this term out -- no branch needed)
        uh = dv / dist[:, None]
        ua, ub = uh[:-1], uh[1:]
        c = jnp.einsum('ij,ij->i', ua, ub)
        Gu = k_angle * (ub - c[:, None] * ua) / dist[:-1, None]
        Gv = k_angle * (ua - c[:, None] * ub) / dist[1:, None]
        f = f.at[:-2].add(-Gu)
        f = f.at[1:-1].add(Gu - Gv)
        f = f.at[2:].add(Gv)

        # 3 -- loop anchors (padding slots carry mask == 0 -> zero contribution)
        dvl = pos[lj] - pos[li]
        distl = jnp.linalg.norm(dvl, axis=1) + 1e-12
        magl = k_loop * (distl - b0) * mask
        fl = (magl / distl)[:, None] * dvl
        f = f.at[li].add(fl)
        f = f.at[lj].add(-fl)

        # dense O(N^2) pair block
        d = pos[:, None, :] - pos[None, :, :]
        r2 = jnp.einsum('ijk,ijk->ij', d, d)
        r2 = jnp.where(eye, 1.0, r2)
        r = jnp.sqrt(r2)
        r_safe = jnp.where(r < 1e-6, 1e-6, r)
        u = d / r_safe[:, :, None]

        # 4 -- soft excluded volume (finite at contact -> chains may cross)
        close = (r < ev_rc) & ev_mask & (~eye)
        rep = jnp.where(close, (2.0 * ev_eps / ev_rc) * (1.0 - r / ev_rc), 0.0)

        # 5 -- Gaussian block-copolymer attraction
        att = -eps * (r / (sigma ** 2)) * jnp.exp(-r2 / (2.0 * sigma ** 2))

        fmag = rep + att
        f = f + jnp.einsum('ij,ijk->ik', fmag, u)

        # 6 -- soft box walls
        over = jnp.abs(pos) - box
        f = f - k_wall * jnp.where(over > 0.0, over, 0.0) * jnp.sign(pos)

        # 7 -- the player's hand
        f = f + grab_active * k_grab * is_grabbed * (grab_target[None, :] - pos)
        return f

    # ------------------------------------------------------------- integrator
    def _integrate_impl(pos, key, n_steps, types, li, lj, mask,
                         grab_idx, grab_target, grab_active,
                         theta, box, sep, ev_mask, eye):
        (dt, gamma, kT, b0, k_bond, k_angle, k_loop, k_grab,
         ev_eps, ev_rc, sigma, eps_AA, eps_BB, eps_AB, eps_domain, k_wall) = theta

        eps = _eps_matrix(types, li, lj, mask, eps_AA, eps_BB, eps_AB, eps_domain, sep)
        mob = dt / gamma
        amp = jnp.sqrt(2.0 * kT * dt / gamma)
        n = pos.shape[0]
        is_grabbed = (jnp.arange(n) == grab_idx).astype(pos.dtype)[:, None]

        def body(carry, _):
            pos, key = carry
            f = _forces(pos, b0, k_bond, k_angle, li, lj, mask, k_loop,
                        ev_eps, ev_rc, eps, k_wall, box, sep, ev_mask, eye,
                        is_grabbed, grab_target, k_grab, grab_active, sigma)
            key, sub = jax.random.split(key)
            noise = jax.random.normal(sub, pos.shape, dtype=pos.dtype)
            pos = pos + f * mob + amp * noise
            pos = jnp.clip(pos, -box * 1.12, box * 1.12)
            return (pos, key), None

        (pos, key), _ = jax.lax.scan(body, (pos, key), xs=None, length=n_steps)
        return pos, key

    _integrate = jax.jit(_integrate_impl, static_argnums=(2,))

else:  # pragma: no cover
    _integrate = None


# -------------------------------------------------------------- the engine
# Imported lazily (not at module top) to avoid a circular import, since
# physics.py doesn't need to know this module exists.
from .physics import Polymer, SimulationUnstable  # noqa: E402


class FastPolymer(Polymer):
    """Drop-in replacement for Polymer: identical public API and identical
    physics (see test_physics_jax.py for the correctness proof), but step()
    runs on the fused JAX engine above instead of a Python loop of NumPy
    calls. Falls back to the exact NumPy path automatically if jax isn't
    installed -- the game always runs, it's just faster with jax present.

    Used everywhere a polymer is stepped: the live Session/LabSession, the
    menu's background chain, and levels.py's target generation -- one engine,
    so results and performance are consistent across the whole game.

    Note on JIT recompiles: jax.jit compiles once per distinct (N, n_steps)
    pair and reuses that compiled function forever after. N only changes
    when a level starts or the Lab's bead-count is changed (rare, discrete
    actions); n_steps is a fixed constant during normal play. The one place
    this can be felt is dragging the Lab's "steps / frame" slider, which can
    briefly (~1s) recompile the first time a new value is visited -- a fair
    trade for cutting steady-state physics cost by 3-5x.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._key = (jax.random.PRNGKey(int(self.rng.integers(0, 2**31 - 1)))
                     if HAVE_JAX else None)
        self._sep_j = self._ev_mask_j = self._eye_j = None   # lazy, cached per-instance
        # Device copies of loops/types/params -- rebuilt only when the host
        # value actually changes (see _loops_device/_types_device/_params_device
        # below), not on every step(). Host->device transfer was being paid
        # every single frame for data that's static almost all the time.
        self._loops_sig = None
        self._li_j = self._lj_j = self._mask_j = None
        self._types_sig = None
        self._types_j = None
        self._params_id = None
        self._params_j = None

    def _jax_arrays(self):
        if self._sep_j is None:
            self._sep_j = jnp.asarray(self._sep)
            self._ev_mask_j = jnp.asarray(self._ev_mask)
            self._eye_j = jnp.asarray(self._eye)
        return self._sep_j, self._ev_mask_j, self._eye_j

    def _loops_device(self):
        sig = tuple(self.loops)   # cheap: at most a handful of loops
        if sig != self._loops_sig:
            li, lj, mask = pack_loops(self.loops)
            self._li_j, self._lj_j, self._mask_j = jnp.asarray(li), jnp.asarray(lj), jnp.asarray(mask)
            self._loops_sig = sig
        return self._li_j, self._lj_j, self._mask_j

    def _types_device(self):
        sig = self.types.tobytes()
        if sig != self._types_sig:
            self._types_j = jnp.asarray(self.types, dtype=jnp.float64)
            self._types_sig = sig
        return self._types_j

    def _params_device(self):
        # SimParams is a frozen dataclass replaced wholesale on any change
        # (apply_param/set_params), so identity alone tells us "changed".
        if self._params_j is None or self.p is not self._params_id:
            self._params_j = jnp.asarray(pack_params(self.p))
            self._params_id = self.p
        return self._params_j

    def step(self, n_steps: int = 1) -> None:
        if not HAVE_JAX:
            return super().step(n_steps)

        sep_j, ev_mask_j, eye_j = self._jax_arrays()
        li, lj, mask = self._loops_device()
        types_j = self._types_device()
        params_j = self._params_device()
        if self.grab is not None:
            gi, target = self.grab
            grab_active, grab_idx = 1.0, int(gi)
            grab_target = jnp.asarray(target, dtype=jnp.float64)
        else:
            grab_active, grab_idx, grab_target = 0.0, 0, jnp.zeros(3)

        pos_out, self._key = _integrate(
            jnp.asarray(self.pos), self._key, int(n_steps),
            types_j, li, lj, mask,
            grab_idx, grab_target, grab_active,
            params_j, float(self.box),
            sep_j, ev_mask_j, eye_j)

        pos_out = np.asarray(pos_out)
        if not np.all(np.isfinite(pos_out)):
            raise SimulationUnstable(
                "positions diverged -- check dt, k_bond, gamma and force strengths")
        self.pos = pos_out

    def warm_up(self) -> None:
        """Pay the one-off JIT-compile cost for this (N, steps_per_frame)
        pair right now (call this while a loading screen is already
        showing) instead of on the first real frame of play, where it
        would show up as a single stutter. Compiles for the actual
        steps_per_frame the session will call every frame -- the cache key
        includes n_steps, so warming up with a different value would miss."""
        if not HAVE_JAX:
            return
        try:
            self.step(self.p.steps_per_frame)
        except SimulationUnstable:
            pass
