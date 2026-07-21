"""Coarse-grained chromatin model.

A chain of N beads evolves under overdamped Langevin (Brownian) dynamics:

    r(t+dt) = r(t) + F(r)/gamma * dt + sqrt(2 kT dt / gamma) * xi,   xi ~ N(0,1)

`gamma` is the friction coefficient. It is a pure time rescale: raising it does
not change the equilibrium ensemble, it only makes every step advance less --
which is exactly what we want for a calm picture. Compensate for it by taking
more `steps_per_frame`.

Energy terms, in units of the bond length b0 and kT:

  1. Backbone   U = 1/2 k_bond (|r_i - r_i+1| - b0)^2         stiff, unbreakable
  2. Bending    U = k_angle (1 - cos phi_i)                   Kratky-Porod
  3. Loops      U = 1/2 k_loop (|r_i - r_j| - b0)^2           |i - j| >= 3
  4. Copolymer  U = -eps_ij exp(-d^2 / 2 sigma^2)             Gaussian, same-type
                   eps_AA weak   (red,  E1 > 0, euchromatin)
                   eps_BB strong (blue, E1 < 0, heterochromatin)
                   eps_AB ~ 0    (Flory incompatibility)
  5. Excl. vol. U = E0 (1 - d/rc)^2   for d < rc              soft-core (crossable)
  6. Wall       U = 1/2 k_wall (|x| - L)^2 per axis, |x| > L

Everything is O(N^2) and fully vectorised. N is small (20-100) by design: the
whole point of the game is that you can follow every bead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_TYPE = 1    # red  -- positive E1 -- weak self-attraction   (euchromatin)
B_TYPE = -1   # blue -- negative E1 -- strong self-attraction (heterochromatin)

MIN_LOOP_SPAN = 3       # |i - j| must be at least this: no loop onto a neighbour
N_MIN, N_MAX = 10, 1000  # small polymers only, chosen for interactive feel


@dataclass(frozen=True)
class SimParams:
    dt: float = 0.001
    b0: float = 1.0
    kT: float = 0.5
    gamma: float = 5.0          # friction: higher = calmer, and proportionally slower

    k_bond: float = 400.0       # stiff backbone     (dt*k/gamma = 0.25, stable)
    k_angle: float = 10.0        # bending rigidity -> smooth, worm-like chain
    k_loop: float = 40.0
    k_grab: float = 80.0

    ev_eps: float = 200.0      # soft-core height (kT): tall enough to keep apart
    ev_rc: float = 1.0         # exclusion diameter

    sigma: float = 2.0         # Gaussian attraction width  (>= ev_rc)
    eps_AA: float = 0.5        # red / red   -- weak attraction
    eps_BB: float = 3.0        # blue / blue -- strong attraction
    eps_AB: float = -0.25       # red / blue  -- slight incompatibility (Flory chi)
    eps_domain: float = 0.20    # extra cohesion inside a loop (extrusion-like TAD)

    k_wall: float = 90.0

    contact_rc: float = 1.0    # contact called below this separation
    contact_w: float = 0.15     # softness of the contact call

    steps_per_frame: int = 60   # was 25 -- gamma=4 slows dynamics 4x, take more steps


def box_half_side(n: int) -> float:
    """Confinement half-side. Scales as N^(1/3) so density is N-independent."""
    return float(1.35 * n ** (1.0 / 3.0))


class Polymer:
    """Chain state + integrator. Mutations set a dirty flag; energetics are cached."""

    def __init__(self, n: int, params: SimParams | None = None, seed: int = 0):
        self.n = int(np.clip(n, N_MIN, N_MAX))
        self.p = params or SimParams()
        self.rng = np.random.default_rng(seed)
        self.box = box_half_side(self.n)
        self.types = np.full(self.n, A_TYPE, dtype=np.int8)
        self.loops: list[tuple[int, int]] = []
        self.grab: tuple[int, np.ndarray] | None = None
        self.pos = self._seed_positions()

        idx = np.arange(self.n)
        self._sep = np.abs(idx[:, None] - idx[None, :])
        self._eye = np.eye(self.n, dtype=bool)
        self._ev_mask = self._sep >= 2          # backbone neighbours skip excl. volume
        self._eps: np.ndarray | None = None
        self._dirty = True

    # ---------------------------------------------------------------- setup
    def _seed_positions(self) -> np.ndarray:
        """A stiff-ish random walk, centred and shrunk to fit the box."""
        pos = np.zeros((self.n, 3))
        d = np.array([1.0, 0.0, 0.0])
        for i in range(1, self.n):
            d = d + 0.5 * self.rng.normal(size=3)  # a touch stiffer to match k_angle
            d /= np.linalg.norm(d) + 1e-12
            pos[i] = pos[i - 1] + d * self.p.b0
        pos -= pos.mean(axis=0)
        span = np.abs(pos).max()
        if span > self.box * 0.75:
            pos *= (self.box * 0.75) / span
        return pos
    
    def set_params(self, params: SimParams) -> None:
        self.p = params
        self._dirty = True

    def clone_config(self):
        return self.types.copy(), list(self.loops)

    def load_config(self, types, loops) -> None:
        self.types = np.asarray(types, dtype=np.int8).copy()
        self.loops = [tuple(l) for l in loops]
        self._dirty = True

    # ------------------------------------------------------------- mutation
    def set_type(self, i: int, t: int) -> None:
        if self.types[i] != t:
            self.types[i] = t
            self._dirty = True

    def flip_type(self, i: int) -> None:
        self.set_type(i, B_TYPE if self.types[i] == A_TYPE else A_TYPE)

    def has_loop(self, i: int, j: int) -> bool:
        a, b = (i, j) if i < j else (j, i)
        return (a, b) in self.loops

    def valid_loop(self, i: int, j: int) -> bool:
        return abs(i - j) >= MIN_LOOP_SPAN

    def toggle_loop(self, i: int, j: int) -> str:
        """Returns 'added', 'removed' or 'invalid'."""
        if not self.valid_loop(i, j):
            return "invalid"
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in self.loops:
            self.loops.remove((a, b))
            self._dirty = True
            return "removed"
        self.loops.append((a, b))
        self._dirty = True
        return "added"

    def clear_loops(self) -> None:
        self.loops.clear()
        self._dirty = True

    # ------------------------------------------------------------ energetics
    def eps_matrix(self) -> np.ndarray:
        if self._eps is not None and not self._dirty:
            return self._eps
        t = self.types
        is_a = (t == A_TYPE)
        aa = is_a[:, None] & is_a[None, :]
        bb = (~is_a)[:, None] & (~is_a)[None, :]
        eps = np.where(aa, self.p.eps_AA, np.where(bb, self.p.eps_BB, self.p.eps_AB))
        eps = eps.astype(np.float64)
        # A loop brings the enclosed segment together: the classic extrusion TAD.
        for (i, j) in self.loops:
            eps[i:j + 1, i:j + 1] += self.p.eps_domain
        eps[self._sep < 2] = 0.0
        self._eps = eps
        self._dirty = False
        return eps

    def forces(self, pos: np.ndarray) -> np.ndarray:
        p = self.p
        f = np.zeros_like(pos)

        # 1 -- backbone
        dv = pos[1:] - pos[:-1]
        dist = np.linalg.norm(dv, axis=1) + 1e-12
        mag = p.k_bond * (dist - p.b0)
        fb = (mag / dist)[:, None] * dv
        f[:-1] += fb
        f[1:] -= fb

        # 2 -- bending: U = k_angle (1 - cos phi) between consecutive bonds
        if p.k_angle > 0.0 and self.n >= 3:
            uh = dv / dist[:, None]
            a, b = uh[:-1], uh[1:]
            c = np.einsum("ij,ij->i", a, b)
            Gu = p.k_angle * (b - c[:, None] * a) / dist[:-1, None]
            Gv = p.k_angle * (a - c[:, None] * b) / dist[1:, None]
            f[:-2] -= Gu
            f[1:-1] += Gu - Gv
            f[2:] += Gv

        # 3 -- loop anchors
        if self.loops:
            li = np.fromiter((l[0] for l in self.loops), dtype=np.intp)
            lj = np.fromiter((l[1] for l in self.loops), dtype=np.intp)
            dv = pos[lj] - pos[li]
            dist = np.linalg.norm(dv, axis=1) + 1e-12
            mag = p.k_loop * (dist - p.b0)
            fl = (mag / dist)[:, None] * dv
            np.add.at(f, li, fl)
            np.add.at(f, lj, -fl)

        # dense O(N^2) pair block
        d = pos[:, None, :] - pos[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        np.fill_diagonal(r2, 1.0)
        r = np.sqrt(r2)
        u = d / r[:, :, None]

        # 4 -- soft excluded volume (finite at contact => chains may cross)
        close = (r < p.ev_rc) & self._ev_mask & ~self._eye
        rep = np.where(close, (2.0 * p.ev_eps / p.ev_rc) * (1.0 - r / p.ev_rc), 0.0)

        # 5 -- Gaussian block-copolymer attraction
        eps = self.eps_matrix()
        att = -eps * (r / (p.sigma ** 2)) * np.exp(-r2 / (2.0 * p.sigma ** 2))

        fmag = rep + att
        f += np.einsum("ij,ijk->ik", fmag, u)

        # 6 -- soft box walls
        over = np.abs(pos) - self.box
        f -= p.k_wall * np.where(over > 0.0, over, 0.0) * np.sign(pos)

        # 7 -- the player's hand
        if self.grab is not None:
            gi, target = self.grab
            f[gi] += p.k_grab * (target - pos[gi])
        return f

    # ----------------------------------------------------------- integrator
    def step(self, n_steps: int = 1) -> None:
        p = self.p
        mob = p.dt / p.gamma                     # mobility * dt
        amp = np.sqrt(2.0 * p.kT * p.dt / p.gamma)
        pos = self.pos
        for _ in range(int(n_steps)):
            f = self.forces(pos)
            pos = pos + f * mob + amp * self.rng.normal(size=pos.shape)
            np.clip(pos, -self.box * 1.12, self.box * 1.12, out=pos)
        self.pos = pos

    # -------------------------------------------------------------- readout
    def contacts(self) -> np.ndarray:
        """Smooth contact indicator in [0, 1] for the current conformation.

        A sigmoid rather than a hard cutoff: far less sampling noise for the
        same number of frames.
        """
        d = self.pos[:, None, :] - self.pos[None, :, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", d, d) + 1e-12)
        c = 1.0 / (1.0 + np.exp((r - self.p.contact_rc) / self.p.contact_w))
        np.fill_diagonal(c, 1.0)
        return c

    def radius_of_gyration(self) -> float:
        c = self.pos - self.pos.mean(axis=0)
        return float(np.sqrt((c ** 2).sum(axis=1).mean()))


def simulate_ensemble(poly: Polymer, burn_in: int, n_samples: int, sample_every: int,
                      progress=None) -> np.ndarray:
    """Average contact map over an equilibrated trajectory."""
    total = burn_in + n_samples * sample_every
    done = 0
    chunk = 250
    while done < burn_in:
        k = min(chunk, burn_in - done)
        poly.step(k)
        done += k
        if progress:
            progress(done / total)
    acc = np.zeros((poly.n, poly.n))
    for s in range(n_samples):
        poly.step(sample_every)
        acc += poly.contacts()
        done += sample_every
        if progress and s % 20 == 0:
            progress(done / total)
    if progress:
        progress(1.0)
    return acc / max(1, n_samples)