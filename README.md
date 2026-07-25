# The Chromatin Game

> **Build the fold. Match the map.**

An experimental, interactive Python game about one of the deepest unsolved
problems in molecular biology: how does a two-metre strand of DNA fold itself
into a nucleus four micrometres across — and how can we reconstruct that
three-dimensional structure from the indirect, population-averaged measurements
that experiments give us?

```bash
pip install -r requirements.txt
python main.py
```

Requires Python 3.10+, `pygame-ce`, `numpy`, `scipy`, and `matplotlib`.
The 3D renderer, the Hi-C analysis pipeline, and the colourmaps are all
written from scratch — no OpenGL, no pandas, no scikit-learn.

---

## Inspiration — the inverse problem

Most of physics runs forward: you know the rules, you predict what happens.
Chromatin folding runs the other way. Hi-C experiments give you a contact
frequency matrix — a heatmap where entry (i, j) encodes how often genomic
loci i and j were found close together in space — and you must infer what
three-dimensional structure produced it.

This is an **inverse problem**, and inverse problems are hard by construction.
A well-posed problem (Hadamard, 1902) has a solution that exists, is unique,
and changes continuously with the data. Inverse problems fail on all three
counts. For chromatin in particular:

- Hi-C is an **ensemble average** over millions of cells, each with a slightly
  different conformation. The same heatmap is consistent with many completely
  different structural ensembles.
- The measurement is **lossy**: all phase information is destroyed, and only
  pairwise proximity is recorded, not absolute positions.
- **Noise and systematic biases** in the sequencing mean that small changes in
  the map can correspond to wildly different inferred structures.

There is no algorithm that reliably solves this inverse problem in full
generality. What researchers do instead — and what this game asks you to do —
is constrain the space of solutions with a physical polymer model and ask which
parameter choices make the simulated contact map best match the experimental
one.

The game was built because the chromatin field draws together physicists,
biologists, mathematicians, and computer scientists, many of whom have deep
expertise in one area and limited intuition for the others. Biologists who work
with Hi-C data daily may never have seen a polymer simulation; polymer
physicists may have little sense of what a contact map actually encodes
biologically. The game is an **experimental framework for building that
intuition interactively** — not reading about it, but doing it, watching the
heatmap change in real time as you place loops and paint compartments.

---

## The physics

The chromatin fibre is modelled as a chain of N beads evolving under
**overdamped Langevin dynamics** — the correct model for a polymer in a
viscous environment where inertia is negligible compared to friction:

```
dr/dt  =  F(r) / γ  +  √(2kT/γ) · ξ(t)
```

γ is the friction coefficient, kT is thermal energy, and ξ(t) is Gaussian
white noise whose amplitude is fixed by the fluctuation-dissipation theorem.
Raising γ slows the dynamics without changing which structures are
thermodynamically stable. Integration uses the Euler-Maruyama scheme with
timestep dt; stability requires `dt · k_bond / γ < 0.5`.

The total potential energy is a sum of six terms:

### 1. Backbone bonds
Stiff harmonic springs between consecutive beads with equilibrium length b₀
keep the chain connected. Without other forces this gives a freely-jointed
random walk. The stiffness k_bond is large enough that bonds stay near b₀
at all times.

### 2. Bending stiffness (Kratky-Porod)
A bending penalty on the angle φ between consecutive bond vectors gives the
chain a persistence length and makes it look like a smooth worm rather than
a crumpled string. Larger k_angle → stiffer, more rod-like chain.

```
U_bend  =  k_angle · Σᵢ (1 − cos φᵢ)
```

### 3. Excluded volume
A soft-core repulsion between all non-bonded pairs (|i−j| ≥ 2) keeps beads
apart while remaining **crossable** — chains can pass through each other.
This matches the biology (topoisomerases resolve tangles in the cell) and
keeps the simulation fast. The softness is controlled by ε_EV and diameter r_c.

### 4. Loop bonds
Each placed loop (m, n) adds a harmonic spring between two non-consecutive
beads with |m−n| ≥ 3, mimicking a cohesin ring stalled at two CTCF sites.
The spring pulls all beads between m and n into a compact domain — a **TAD**
— visible as a bright square on the diagonal of the contact map with a corner
dot at (m, n).

### 5. Block-copolymer attraction
A Gaussian pairwise attraction whose well depth depends on the epigenetic
types of the two beads drives **phase separation**:

```
U_cp(i,j)  =  −ε(tᵢ, tⱼ) · exp(−|rᵢ − rⱼ|² / 2σ²)
```

- **ε_BB** (blue-blue, strong): heterochromatin collapses into a dense core
- **ε_AA** (red-red, weak): euchromatin stays more open and accessible  
- **ε_AB** (red-blue, negative): Flory-Huggins incompatibility sharpens the boundary

This single term is responsible for the entire **plaid checkerboard** pattern
in Hi-C — loci in the same compartment enrich each other, loci across
the boundary are depleted.

### 6. Confinement
Soft harmonic walls confine the chain to a cube of half-side L ∝ N^(1/3),
keeping bead density N-independent as you change chain length. Only beads
outside the box feel a force; inside it is invisible.

### Validation
These are not free parameters chosen to make the game look good. With the
ground-truth colouring and loop placement, the simulated E1 eigenvector
correlates with bead types at **r ≈ 0.96**, and a correct loop gives a
corner contact probability of **~0.93** against a **~0.2** background —
numbers consistent with what real coarse-grained chromatin models achieve
in the literature.

---

## The analysis pipeline

`chromatin/analysis.py` implements a miniature but genuine Hi-C pipeline:

```
raw contacts  →  O/E normalisation  →  Pearson correlation matrix  →  E1
```

**O/E normalisation** divides each entry P(i,j) by the genome-distance-averaged
mean, removing the distance decay and revealing compartment structure.

**The correlation matrix** C(i,j) is the Pearson correlation between rows i and
j of the O/E matrix — positive when two loci have similar contact profiles
(same compartment), negative when they do not.

**E1** is the first eigenvector of C, sign-oriented by bead type exactly as
real pipelines orient by GC content. Positive E1 = A compartment (red);
negative E1 = B compartment (blue).

Scoring uses five metrics:

| Metric | What it measures |
|---|---|
| **SCC** | stratum-adjusted correlation — the headline score, robust to distance decay |
| **E1 r** | Pearson correlation of eigenvectors — directly rewards correct compartments |
| **r_cb** | checkerboard correlation — rewards fine compartment structure |
| **APA** | aggregate peak analysis — rewards loop enrichment at correct positions |
| **F1** | anchor F1 — rewards precise loop placement, penalises spurious loops |

Your map is always an **ensemble average over simulation time**. One structure
is never a Hi-C map — pressing Enter runs a long measurement (520 conformations)
and that ensemble average is what gets scored.

---

## Gameplay

### Single player
You control both loops and compartments, trying to maximise the total score.
The simulation runs live in 3D; pressing Enter freezes it, measures the ensemble
average, and reports your score.

### Two-player versus
One player controls loops, the other controls compartments — on **the same polymer**.
They are scored independently on different features of the same contact map.
The scores are genuinely orthogonal: correct loops with random colours scores
LOOP 80 / COMP 7; correct colours with no loops scores LOOP 14 / COMP 81.

### Hard mode
The target's loop anchors and E1 track are hidden. You must read the structure
from the raw contact map alone — as a real analyst would.

### Chromatin MiNI-Lab
A free-play sandbox with live sliders for every force-field parameter. Change
γ, k_bond, ε_BB, σ, contact cutoff — anything — and watch the polymer and
heatmaps respond in real time. Includes a live structural analysis panel
(Rg, R_ee, asphericity, compaction index, contact count, bond length) and
full interactive loop and compartment editing.

---

## Controls

| Key / action | Effect |
|---|---|
| **L / C** | loops mode / compartments mode |
| **click a map cell (i,j)** | tie or untie that loop |
| **click or drag the colour ribbon** | paint A/B compartments |
| click bead i then bead j in 3D | tie a loop (alternative) |
| drag a bead | grab and pull the polymer |
| drag empty space / scroll wheel | orbit / zoom the 3D view |
| **Enter** | measure ensemble and lock in a score |
| **A** | open / close the structural analysis panel |
| V | cycle map view: contact / O/E correlation |
| Space · F · R | pause · fast-forward · reset camera |
| H or F1 | in-game help |
| M · N · +/- | mute · next track · volume |

---

## Music

The soundtrack is original piano music composed and performed by
**Sebastian Korsak** (also known as **BlackPianoCat**). The recordings
live in the `music/` folder and are played on shuffle-free rotation.
Drop any additional `.mp3`, `.ogg`, `.wav`, or `.flac` files into that
folder and they will be picked up automatically at the next launch.

`M` mutes, `N` skips to the next track, `+`/`-` adjust volume.
The game runs correctly with no music files and with no audio device.

---

## Code layout

```
chromatin/
  physics.py        overdamped Langevin: backbone, bending, loops,
                    excluded volume, copolymer, confinement
  analysis.py       O/E, correlation matrix, E1, SCC, APA, F1, scoring
  levels.py         level definitions; targets built by the game's own
                    physics and cached in .cache/
  render3d.py       software 3D renderer: camera, Lambert sphere sprites,
                    painter's algorithm — no OpenGL
  widgets.py        heatmaps, E1 track, colour ribbon, metric table,
                    sliders, text fields
  colormaps.py      fall and coolwarm LUTs, no matplotlib dependency
  theme.py          palette, typography, font scaling
  audio.py          music transport
  app.py            game loop, states, event routing
  draw.py           all draw_* methods
  interact.py       mouse and keyboard interaction handlers
  session.py        Session and LabSession state classes
  constants.py      shared state constants and parameter groups
  metrics.py        live structural metrics (Rg, asphericity, etc.)
  analysis_panel.py live time-series plot grid
  manual_panel.py   in-game manual renderer
  manual_content.py manual text, equations, and section structure
  backend.py        optional numba / GPU acceleration for force kernel
```

Targets are generated by running the game's own simulator on a hidden
ground truth, so every level is provably reachable. The first launch of
a level spends a few seconds building it; the result is cached in `.cache/`
and is instant afterwards. Solo records land in `records.json`.

---

## Dependencies

```
pygame-ce >= 2.4    # rendering and audio
numpy    >= 1.24    # all numerics
scipy    >= 1.11    # KD-tree neighbour lists for large N
matplotlib >= 3.7   # equation rendering in the manual
Pillow   >= 9.0     # tight-crop of rendered equations
numba    >= 0.58    # optional: multi-core force kernel
```

GPU acceleration (CUDA) is supported via `cupy` if a compatible device is
detected at runtime; the game falls back to numba or numpy otherwise.
