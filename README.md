# The Chromatin Game

> **Build the fold. Match the map.**

An experimental, interactive Python game about one of the most delightfully
stubborn puzzles in molecular biology: a two-metre strand of DNA somehow
folds itself into a nucleus four micrometres across, and all we ever get to
see of the result is an indirect, population-averaged shadow of it. Can you
work backwards from that shadow to the shape that cast it?

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
Chromatin folding hands you the puzzle backwards — you get the measurement
first, and your job is to recover the structure that produced it. Hi-C
experiments give you a contact frequency matrix, a heatmap where entry
(i, j) encodes how often genomic loci i and j were found close together in
space, and from that alone you have to infer the three-dimensional shape
responsible.

That's called an **inverse problem**, and nature is not shy about making
them hard. A well-posed problem (Hadamard, 1902) has a solution that exists,
is unique, and changes continuously with the data. Inverse problems love to
break all three rules. For chromatin in particular:

- Hi-C is an **ensemble average** over millions of cells, each folded
  slightly differently. The same heatmap is consistent with many completely
  different structural ensembles — there's room for more than one right answer.
- The measurement is **lossy**: all phase information is destroyed, and only
  pairwise proximity is recorded, not absolute positions.
- **Noise and systematic biases** in the sequencing mean that small changes in
  the map can correspond to wildly different inferred structures.

There is no algorithm that reliably solves this inverse problem in full
generality. What researchers do instead — and what this game asks you to do —
is constrain the space of solutions with a physical polymer model and ask
which parameter choices make the simulated contact map best match the
experimental one.

The game exists because the chromatin field is a happy mix of physicists,
biologists, mathematicians, and computer scientists who don't always speak
the same dialect. Biologists who work with Hi-C data daily may never have
watched a polymer simulation wiggle; polymer physicists may have never had
to explain what a contact map means biologically. This game is a shared,
hands-on place to build that intuition together — not by reading about it,
but by doing it: placing loops, painting compartments, and watching the
heatmap talk back in real time.

---

## The physics

The chromatin fibre is modelled as a chain of N beads evolving under
**overdamped Langevin dynamics** — a marble dropped in honey, not a ball
in free flight. Friction from the surrounding medium is so large that
inertia barely matters, so it's velocity, not acceleration, that's
proportional to the force:

```
dr/dt  =  F(r) / γ  +  √(2kT/γ) · ξ(t)
```

γ is the friction coefficient, kT is thermal energy (the gentle background
hum of heat), and ξ(t) is Gaussian white noise whose amplitude is fixed by
the fluctuation-dissipation theorem — the system always settles at the
correct thermal equilibrium no matter how large γ gets. Raising γ just
slows the dance down without changing which structures are thermodynamically
stable. Integration uses the Euler-Maruyama scheme with timestep dt;
stability requires `dt · k_bond / γ < 0.5`.

The total potential energy is a sum of six terms, each with its own small
physical story:

### 1. Backbone bonds
The most basic rule in the whole game: consecutive beads have to stay
connected. Stiff harmonic springs with equilibrium length b₀ do the job —
strip away every other force and the backbone alone gives you a
freely-jointed random walk, plain old spaghetti. The stiffness k_bond is
large enough that bonds stay close to b₀ at all times.

### 2. Bending stiffness (Kratky-Porod)
Real chromatin isn't loose spaghetti either — it resists sharp bends over
short stretches. A bending penalty on the angle φ between consecutive bond
vectors gives the chain a persistence length and makes it read as a smooth
worm rather than a crumpled string. Larger k_angle → stiffer, more rod-like
chain.

```
U_bend  =  k_angle · Σᵢ (1 − cos φᵢ)
```

### 3. Excluded volume
A soft-core repulsion between all non-bonded pairs (|i−j| ≥ 2) keeps beads
from piling on top of each other, while staying **crossable** — chains can
still pass through one another. That matches the biology (topoisomerases
resolve tangles in the living cell) and keeps the simulation fast. The
softness is controlled by ε_EV and diameter r_c.

### 4. Loop bonds
Each placed loop (m, n) adds a harmonic spring between two non-consecutive
beads with |m−n| ≥ 3 — a cohesin ring, caught in the act of reeling in DNA
until it stalls at two CTCF sites. The spring pulls every bead between m
and n into a compact domain: a **TAD**, visible as a bright square on the
diagonal of the contact map with a corner dot at (m, n).

### 5. Block-copolymer attraction
A Gaussian pairwise attraction whose well depth depends on the epigenetic
types of the two beads drives **phase separation** — guests clustering
around the same snack table:

```
U_cp(i,j)  =  −ε(tᵢ, tⱼ) · exp(−|rᵢ − rⱼ|² / 2σ²)
```

- **ε_BB** (blue-blue, strong): heterochromatin collapses into a dense core
- **ε_AA** (red-red, weak): euchromatin stays more open and accessible  
- **ε_AB** (red-blue, negative): Flory-Huggins incompatibility sharpens the boundary

This one term is responsible for the entire **plaid checkerboard** pattern
in Hi-C — loci sharing a compartment enrich each other, loci across the
boundary get the cold shoulder.

### 6. Confinement
Soft harmonic walls confine the chain to a cube of half-side L ∝ N^(1/3),
keeping bead density N-independent as you change chain length. Only beads
outside the box feel anything; inside it is invisible.

### Validation
These aren't free parameters tuned just to make the game look good. With
the ground-truth colouring and loop placement, the simulated E1 eigenvector
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
| **T** | cycle 3D colour mode: compartment / rainbow / loop domains |
| **G** | cycle 3D shape: spheres / rigid chain / ribbon |
| Space · F · R | pause · fast-forward · reset camera |
| H or F1 | in-game help |
| **M** | start music, then toggle mute |
| N · +/- | next track · volume |

The 3D viewport also has **Colour** and **Shape** buttons in its top-right
corner for the mouse -- same toggles as T and G, available in both normal
play and the MiNI-Lab.

**3D colour modes (T):** *compartment* tints each bead by its A/B type
(pink/blue), fading toward grey where the E1 signal is weak; *rainbow*
gives every bead a unique hue by its index along the chain, with a
colour-bar legend; *loop* shows orange for beads outside any tied loop
and green for beads inside one, i.e. the stretch a loop's extra
attraction (`eps_domain` in the physics) actually pulls together.

**3D shapes (G):** *spheres* is the original ball-and-stick look; *rigid
chain* draws the backbone as one continuous, glossy tube with rounded
joints; *ribbon* traces a smooth Catmull-Rom spline through the backbone
and shades it with a soft, rounded highlight across its width, in the
style of a ChimeraX or PyMOL cartoon ribbon, rather than a flat painted
strip. All three use the same colour modes above and the same camera,
picking and loop/anchor overlays -- only how the backbone's body is drawn
changes.

A small **axis gizmo** sits in the bottom-right corner of the 3D viewport
at all times, showing the X/Y/Z orientation of the camera as three
colour-coded arms (red/green/blue) that rotate along with the view -- a
quick way to tell which way the structure is actually facing.

---

## Music

The soundtrack is original piano music composed and performed by
**Sebastian Korsak** (also known as **BlackPianoCat**). The recordings
live in the `music/` folder and are played on shuffle-free rotation.
Drop any additional `.mp3`, `.ogg`, `.wav`, or `.flac` files into that
folder and they will be picked up automatically at the next launch.

Music never starts on its own -- press **M** at any time, or use the
**Play music** button on the Settings screen, to start the first track.
Press `M` again any time after that to mute or unmute. `N` skips to the
next track, `+`/`-` adjust volume.
The game runs correctly with no music files and with no audio device.

---

## Appearance

Settings has a **theme** picker: 5 named palettes, 3 dark and 2 light,
switchable at any time with no restart needed.

| Theme | Kind |
|---|---|
| **Daylight** (default) | light |
| Paper | light |
| Midnight | dark |
| Carbon | dark |
| Nebula | dark |

Scientific colour conventions (compartment red/blue, loop green, the
Hi-C contact/correlation colour ramps) are the same in every theme --
only the surrounding chrome (panels, text, rules, accents) re-skins.
The choice is saved in `settings.json`.

UI text is set in **Quicksand** (bundled under `chromatin/fonts/`,
SIL Open Font License), chosen for a smoother, more modern look that
doesn't depend on what's installed on the machine running the game.
Numbers and aligned data keep a monospace font, resolved from whatever
coding font is available on the system.

---

## Code layout

```
chromatin/
  physics.py        overdamped Langevin: backbone, bending, loops,
                    excluded volume, copolymer, confinement
  physics_jax.py    same equations, JIT-fused into one compiled update via
                    JAX so many steps run per frame without re-entering
                    Python; falls back to physics.py if JAX isn't installed
  analysis.py       O/E, correlation matrix, E1, SCC, APA, F1, scoring
  levels.py         level definitions; targets built by the game's own
                    physics and cached in .cache/
  polymer_view.py   picks the 3D renderer: GPU (render3d_gpu.py) if
                    Panda3D can open a graphics context, else software
  render3d.py       software 3D renderer: camera, Lambert sphere sprites,
                    painter's algorithm, axis orientation gizmo — no OpenGL
  render3d_gpu.py   Panda3D renderer: same camera/picking math and axis
                    gizmo, beads and bonds drawn with the GPU, composited
                    into the pygame window every frame
  beadcolor.py      the three bead colour modes (compartment/rainbow/loop),
                    shared by both renderers so they always match
  chainshape.py     the three backbone shapes (spheres/chain/ribbon):
                    Catmull-Rom spline subdivision and rounded
                    cross-section shading for the ribbon's ChimeraX-style
                    look, shared by both renderers so they always match
  widgets.py        heatmaps, E1 track, colour ribbon, metric table,
                    sliders, text fields
  colormaps.py      fall, coolwarm and rainbow LUTs, no matplotlib dependency
  theme.py          5 palettes (3 dark + 2 light) + set_theme(), typography,
                    font scaling -- bundled Quicksand is the default UI font
  fonts/            bundled Quicksand TTFs (Regular/Medium/Bold) + OFL.txt
  audio.py          music transport (no autoplay -- explicit start only)
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

## Resetting progress & clearing the cache

The game writes three things next to `main.py`, all plain files/folders you
can delete by hand at any time -- nothing about the code itself depends on
them, and the game quietly recreates whatever it needs on the next launch.

| Path | What it holds | Effect of deleting it |
|---|---|---|
| `.cache/` | Cached level targets (`target_<hash>.npz`, one per level + seed) | Safe any time. The next time that level loads it spends a few seconds re-running the hidden ground-truth simulation; the result is identical either way since targets are fully reproducible from their seed. |
| `records.json` | Your best score per level, separately for easy and hard mode, with the seed that earned it | Clears all personal records back to empty -- next play on any level is a fresh "new record". |
| `settings.json` | Volume, theme, resolution, fullscreen, font size | Resets to the defaults (Daylight theme, small font, 1600x900 windowed, fullscreen off, full volume). |

To wipe everything and start completely fresh, close the game and run
(from the project root, where `main.py` lives):

```bash
rm -rf .cache records.json settings.json
```

Deleting only `.cache/` is the handy one if a level ever seems stuck on a
stale or corrupt target -- the game already falls back to rebuilding it
automatically if the cached file fails to load, so this is mostly for
reclaiming disk space or forcing a clean rebuild by hand.

---

## Dependencies

```
pygame-ce >= 2.4    # rendering and audio
numpy    >= 1.24    # all numerics
scipy    >= 1.11    # KD-tree neighbour lists for large N
matplotlib >= 3.7   # equation rendering in the manual
Pillow   >= 9.0     # tight-crop of rendered equations
numba    >= 0.58    # optional: multi-core force kernel
jax      >= 0.4     # optional: JIT-fused CPU physics stepping
jaxlib   >= 0.4      #           (physics_jax.py; falls back to plain numpy)
panda3d  >= 1.10     # optional: GPU-accelerated 3D view (render3d_gpu.py)
```

GPU acceleration (CUDA) is supported via `cupy` if a compatible device is
detected at runtime; the game falls back to numba or numpy otherwise.
JAX and Panda3D are both optional in the same spirit: if either fails to
import or to initialise a working context on the current machine, the
game quietly drops back to the original plain-numpy physics loop and/or
the original software 3D renderer — nothing about install order or a
missing GPU driver can stop the game from starting.

The UI font (Quicksand) ships as TTF files in `chromatin/fonts/` under the
SIL Open Font License, so no extra install step or system font is needed.
