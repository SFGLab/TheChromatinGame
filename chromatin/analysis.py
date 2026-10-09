"""The same analysis pipeline a Hi-C paper would run, at toy scale.

    contact map -> observed/expected -> correlation matrix -> E1 (compartments)
                -> SCC, loop F1 -> score

Pure NumPy; no scipy.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# ------------------------------------------------------------------ helpers
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    if den < EPS:
        return 0.0
    return float(np.clip((x * y).sum() / den, -1.0, 1.0))


def _rank(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks."""
    a = np.asarray(a, dtype=float).ravel()
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(a.size, dtype=float)
    # average ties
    sa = a[order]
    i = 0
    while i < sa.size:
        j = i
        while j + 1 < sa.size and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(_rank(x), _rank(y))


def upper_mask(n: int, min_sep: int = 2) -> np.ndarray:
    idx = np.arange(n)
    sep = idx[None, :] - idx[:, None]
    return sep >= min_sep


# ------------------------------------------------------- the Hi-C pipeline
def observed_over_expected(P: np.ndarray) -> np.ndarray:
    """Divide each diagonal by its mean, removing the P(s) distance decay."""
    n = P.shape[0]
    oe = np.ones_like(P, dtype=float)
    for d in range(0, n):
        vals = np.diagonal(P, d)
        m = vals.mean()
        if m < EPS:
            continue
        r = np.arange(n - d)
        oe[r, r + d] = vals / m
        oe[r + d, r] = vals / m
    return oe


def correlation_matrix(oe: np.ndarray) -> np.ndarray:
    """Pearson correlation between rows of log(O/E): the checkerboard view."""
    lo = np.log(np.clip(oe, 1e-4, None))
    lo = lo - lo.mean(axis=1, keepdims=True)
    sd = np.sqrt((lo ** 2).sum(axis=1))
    sd = np.where(sd < EPS, 1.0, sd)
    C = (lo @ lo.T) / np.outer(sd, sd)
    return np.clip(C, -1.0, 1.0)


def eigenvector_1(C: np.ndarray, orient_by: np.ndarray | None = None) -> np.ndarray:
    """First eigenvector of the correlation matrix, scaled to unit max |value|.

    Sign is arbitrary in the maths, so it is fixed the way every Hi-C paper
    fixes it: by correlating against a known active-chromatin track. Here that
    track is the bead colouring the player themselves assigned, so red ends up
    positive -- exactly the GC-content convention.
    """
    n = C.shape[0]
    Cc = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    Cc = Cc - Cc.mean(axis=0, keepdims=True)
    try:
        w, v = np.linalg.eigh((Cc + Cc.T) / 2.0)
    except np.linalg.LinAlgError:
        return np.zeros(n)
    e1 = v[:, int(np.argmax(w))]
    if orient_by is not None and np.std(orient_by) > EPS:
        if pearson(e1, orient_by) < 0:
            e1 = -e1
    peak = np.abs(e1).max()
    return e1 / peak if peak > EPS else e1


def pipeline(P: np.ndarray, orient_by: np.ndarray | None = None):
    """contact map -> (O/E, correlation matrix, E1)."""
    oe = observed_over_expected(P)
    C = correlation_matrix(oe)
    e1 = eigenvector_1(C, orient_by)
    return oe, C, e1


# ------------------------------------------------------------------ metrics
def scc(P1: np.ndarray, P2: np.ndarray, min_sep: int = 2, max_sep: int | None = None) -> float:
    """Stratum-adjusted correlation: per-diagonal Pearson, variance-weighted.

    This is the metric Hi-C people actually use to compare two maps, because a
    plain Pearson is dominated by the distance decay that every map shares.
    """
    n = P1.shape[0]
    if max_sep is None:
        max_sep = max(min_sep, n // 2)
    num = 0.0
    den = 0.0
    for d in range(min_sep, min(max_sep, n - 1) + 1):
        x = np.diagonal(P1, d)
        y = np.diagonal(P2, d)
        if x.size < 3:
            continue
        sx, sy = x.std(), y.std()
        if sx < 1e-9 or sy < 1e-9:
            continue
        w = x.size * sx * sy
        num += w * pearson(x, y)
        den += w
    return float(num / den) if den > EPS else 0.0


def loop_f1(placed: list[tuple[int, int]], target: list[tuple[int, int]],
            tol: int = 1) -> tuple[float, float, float]:
    """Precision / recall / F1 of anchor placement, with +/- tol bins slack."""
    if not target and not placed:
        return 1.0, 1.0, 1.0
    if not placed or not target:
        return 0.0, 0.0, 0.0
    used = set()
    hits = 0
    for p in placed:
        best, bd = None, 10 ** 9
        for k, t in enumerate(target):
            if k in used:
                continue
            d = max(abs(p[0] - t[0]), abs(p[1] - t[1]))
            if d < bd:
                bd, best = d, k
        if best is not None and bd <= tol:
            used.add(best)
            hits += 1
    prec = hits / len(placed)
    rec = hits / len(target)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > EPS else 0.0
    return float(prec), float(rec), float(f1)


def checkerboard_corr(C_sim: np.ndarray, C_tgt: np.ndarray, min_sep: int = 2) -> float:
    m = upper_mask(C_sim.shape[0], min_sep)
    return pearson(C_sim[m], C_tgt[m])


# ------------------------------------------------------------------- report
class Report(dict):
    """Metric bundle. Keys are stable; the UI reads them by name."""


def evaluate(P_sim: np.ndarray, P_tgt: np.ndarray,
             C_sim: np.ndarray, C_tgt: np.ndarray,
             e1_sim: np.ndarray, e1_tgt: np.ndarray,
             loops_placed: list[tuple[int, int]],
             loops_tgt: list[tuple[int, int]],
             calib: dict | None = None) -> Report:
    """`calib`, when given, is {"loop_score"/"comp_score"/"total": (floor, ceiling)}:
    raw scores from a deliberately-wrong replicate and a flawless one, measured
    against this specific target (see levels.py: _calibrate). Two independently
    sampled ensembles of identical physics never agree perfectly, so without
    this a "perfect" run reads well under 100 -- worse the bigger the polymer.
    (Same idea as replicate reproducibility in real Hi-C.) Headline scores are
    rescaled between floor and ceiling so 0 = nothing right, ~100 = as good as
    physically achievable on this target."""
    n = P_sim.shape[0]
    m = upper_mask(n, 2)
    lp_s = np.log(np.clip(P_sim[m], 1e-4, None))
    lp_t = np.log(np.clip(P_tgt[m], 1e-4, None))

    r = Report()
    r["pearson"] = pearson(lp_s, lp_t)
    r["spearman"] = spearman(lp_s, lp_t)
    r["scc"] = scc(P_sim, P_tgt)
    r["scc_short"] = scc(P_sim, P_tgt, 2, max(4, n // 3))
    r["eig_r"] = pearson(e1_sim, e1_tgt)
    r["checker"] = checkerboard_corr(C_sim, C_tgt)
    p, rc, f1 = loop_f1(loops_placed, loops_tgt)
    r["loop_prec"] = p
    r["loop_rec"] = rc
    r["loop_f1"] = f1

    # APA (loop corner enrichment) used to feed loop_score/total, but the toy
    # physics rarely builds up real loop contacts in a turn's time, so it was
    # dropped instead of silently reading ~0 -- weights below are the old
    # ones rescaled to fill the gap it left.
    pos = lambda x: max(0.0, x)  # noqa: E731
    r["loop_score"] = 100.0 * (0.60 * r["loop_f1"] + 0.40 * pos(r["scc_short"]))
    r["comp_score"] = 100.0 * (0.60 * pos(r["eig_r"]) + 0.40 * pos(r["checker"]))
    r["total"] = 100.0 * (0.40 * pos(r["scc"]) + 0.25 * pos(r["pearson"]) +
                          0.35 * pos(r["eig_r"]))

    if calib:
        for key in ("loop_score", "comp_score", "total"):
            lo, hi = calib[key]
            r[key] = 100.0 * np.clip((r[key] - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    return r
