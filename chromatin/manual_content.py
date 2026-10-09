"""Manual content.

Body text uses plain Unicode symbols for readability.
Standalone equations use {"type": "equation", "latex": "..."}.
"""

SECTIONS = [
    {
        "title": "Inspiration",
        "icon": "I",
        "blocks": [
            {"type": "heading", "text": "The Inverse Problem"},
            {"type": "body", "text":
                "Most of physics runs forward: you're given the rules, and you predict "
                "the outcome. Chromatin folding hands you the puzzle backwards — you "
                "get the measurement first, and your job is to recover the "
                "structure that produced it. That's called an inverse problem, "
                "and nature is not shy about making them hard. This whole game "
                "is built around that one delightful difficulty."},
            {"type": "subheading", "text": "Macrostates and microstates"},
            {"type": "body", "text":
                "A microstate is the full, dizzying detail of a system — "
                "every atom's position, every molecule's conformation. "
                "A macrostate is the coarse, friendlier thing you can actually "
                "observe. Temperature is the macrostate of jittery molecular "
                "velocities; a Hi-C heatmap is the macrostate "
                "of millions of individual chromatin fibres, all folded differently, "
                "averaged into one picture. "
                "Going from micro to macro is the easy direction — just add things up. "
                "Going backwards is where it gets interesting, because many "
                "different microstates can produce the exact same macrostate."},
            {"type": "subheading", "text": "Hi-C as an ensemble average"},
            {"type": "body", "text":
                "Hi-C cross-links DNA, snips it up, and counts how often "
                "pairs of loci end up glued together — a clever proxy for "
                "spatial proximity. The contact matrix P, where P(i,j) is the "
                "probability that loci i and j were close in space, is "
                "measured across millions of cells at once. "
                "So the heatmap you see is an ensemble average, never a "
                "single snapshot. Many different crowds of structures can "
                "produce the same heatmap — which is part of why this is fun "
                "rather than merely frustrating: there's room for more than "
                "one right answer."},
            {"type": "subheading", "text": "Ill-posed by definition"},
            {"type": "body", "text":
                "A well-posed problem (Hadamard, 1902) has a solution "
                "that exists, is unique, and changes smoothly with "
                "the data. Inverse problems love to break all three rules. "
                "For chromatin: not every heatmap corresponds to a "
                "realisable polymer; many polymers can produce the same "
                "heatmap; and small measurement errors can snowball into "
                "wildly different reconstructions. What we *can* do is "
                "constrain the problem with a physical model and ask "
                "which parameters make the simulated map best match "
                "the observed one — which is exactly the game you're about "
                "to play."},
            {"type": "subheading", "text": "Why we built this"},
            {"type": "body", "text":
                "The chromatin field is a happy mix of physicists, biologists, "
                "mathematicians, and computer scientists, and they don't always "
                "speak the same dialect. Plenty of people working with Hi-C data "
                "have never watched a polymer simulation wiggle; plenty of polymer "
                "physicists have never had to explain what a contact map means "
                "biologically. This game exists to build that shared intuition, "
                "hands-on and a little bit playfully. "
                "Place loops, paint compartments, and watch the heatmap talk back. "
                "The physics underneath is real — the same models and "
                "analysis pipeline used in the actual research literature. "
                "The goal was never to *solve* the inverse problem once and for all, "
                "but to let you feel, in your own hands, why it's hard and why it matters."},
        ]
    },
    {
        "title": "Polymer States",
        "icon": "P",
        "blocks": [
            {"type": "heading", "text": "Chromatin as a Spatial Graph"},
            {"type": "body", "text":
                "Picture the chromatin fibre as a chain of N beads on a string, "
                "connected by a backbone. Each bead i carries two "
                "independent personality traits: a loop state "
                "describing which long-range bonds it's tangled up in, "
                "and an epigenetic state s_i describing its local "
                "chromatin chemistry. Together they define a spatial "
                "graph, and the 3D shape that graph folds into is what "
                "produces the contact map you're trying to match."},
            {"type": "subheading", "text": "Cohesin and loop extrusion"},
            {"type": "body", "text":
                "Loops are long-range friendships mediated by cohesin, "
                "a ring-shaped protein complex with a neat trick. Cohesin "
                "encircles the chromatin fibre and acts like a tiny "
                "molecular motor: it reels in a growing loop of DNA until it "
                "stalls at a CTCF boundary element, like a drawstring "
                "pulled tight and tied off. The result is a stable loop "
                "anchored at two genomic positions (m_k, n_k). "
                "The full loop state L of the polymer is just the set "
                "of all such pairs, with each anchor pair required to "
                "sit at least 3 beads apart:"},
            {"type": "equation",
             "latex": r"\mathcal{L} = \{(m_k, n_k) : 0 \leq m_k < n_k \leq N-1,\; n_k - m_k \geq 3\}",
             "label": "loop microstate"},
            {"type": "body", "text":
                "Each pair (m_k, n_k) shows up as a bright corner dot "
                "in the Hi-C heatmap, with a whole square of "
                "elevated contact frequency gathered around it — that's your TAD."},
            {"type": "subheading", "text": "Epigenetic marks and compartments"},
            {"type": "body", "text":
                "Every bead i also carries an epigenetic state s_i in {A, B} — "
                "think of it as the bead's mood. Epigenetic marks are "
                "chemical tags on histones that change how tightly chromatin is "
                "packed and which proteins come to visit. Active euchromatin "
                "(A, red) is gene-rich, busy transcribing, and likes to hang "
                "out near nuclear pores. Inactive heterochromatin (B, blue) "
                "is gene-poor, quiet, and prefers the nuclear "
                "lamina, out near the edges."},
            {"type": "equation",
             "latex": r"s_i \in \{A,\, B\}, \quad i = 0, 1, \ldots, N-1",
             "label": "epigenetic microstate"},
            {"type": "body", "text":
                "Chromatin segments sharing an epigenetic state attract "
                "each other, because they recruit the same proteins — "
                "a bit like guests clustering around the same snack table, "
                "and not unlike liquid-liquid phase separation. "
                "This is captured by a block-copolymer interaction "
                "with well depths eps_AA, eps_BB, and eps_AB:"},
            {"type": "equation",
             "latex": r"\varepsilon_{s_i s_j}: \quad \varepsilon_{AA} > 0,\quad \varepsilon_{BB} \gg \varepsilon_{AA},\quad \varepsilon_{AB} < 0",
             "label": "epigenetic interaction strengths"},
            {"type": "body", "text":
                "The strong B-B attraction (eps_BB) pulls blue beads "
                "into a dense little core. The negative eps_AB is "
                "Flory-Huggins incompatibility at work — A and B beads would "
                "rather not touch, which sharpens the separation between phases. "
                "The visible result is the plaid checkerboard in Hi-C: "
                "loci sharing a compartment get elevated P(i,j), "
                "loci of opposite type get the cold shoulder."},
            {"type": "subheading", "text": "The full microstate"},
            {"type": "body", "text":
                "So the complete description of the polymer at any instant "
                "is the triplet of bead positions {r_i}, the loop "
                "state L, and the epigenetic state {s_i}:"},
            {"type": "equation",
             "latex": r"\Omega = (\{\mathbf{r}_i\}_{i=0}^{N-1},\;\mathcal{L},\;\{s_i\}_{i=0}^{N-1})",
             "label": "full polymer microstate"},
            {"type": "body", "text":
                "The positions {r_i} evolve under Langevin dynamics, "
                "jostled by thermal noise. The loop state L and epigenetic "
                "state {s_i} are your two moves — you set them, and the "
                "simulation does the honest work of exploring every "
                "conformation consistent with your choices."},
        ]
    },
    {
        "title": "Force Field",
        "icon": "F",
        "blocks": [
            {"type": "heading", "text": "The Force Field"},
            {"type": "body", "text":
                "A force field is simply a set of rules that assigns a potential "
                "energy U to every possible arrangement of the system. "
                "From Newton's second law, the force on each particle "
                "is the negative gradient of that energy — nature's way of "
                "saying \"roll downhill.\" "
                "Where the energy landscape slopes, the force "
                "nudges the particle along. Where it's flat, nothing happens. "
                "The particle always drifts toward lower energy, while "
                "thermal fluctuations jiggle it around and let it "
                "explore nearby configurations instead of getting stuck."},
            {"type": "body", "text":
                "In classical molecular dynamics this gives Newton's "
                "equation, mass times acceleration equals minus the "
                "gradient of U. But for a polymer in a thick, "
                "viscous environment like the cell nucleus, inertia barely "
                "matters — the friction from the surrounding medium "
                "is so large that the particle forgets its velocity "
                "almost instantly, like a marble dropped in honey. "
                "The equation of motion simplifies "
                "to the overdamped Langevin equation, where it's "
                "velocity, not acceleration, that's proportional to "
                "the force:"},
            {"type": "equation",
             "latex": r"\frac{d\mathbf{r}}{dt} = \frac{-\nabla U(\mathbf{r})}{\gamma} + \sqrt{\frac{2k_BT}{\gamma}}\;\boldsymbol{\xi}(t)",
             "label": "overdamped Langevin"},
            {"type": "body", "text":
                "Here gamma is the friction coefficient — it sets how "
                "quickly the system responds to forces. kT is thermal "
                "energy, the gentle background hum of heat, and xi(t) is "
                "Gaussian white noise standing in for the constant "
                "random bombardment of solvent molecules. The elegant part "
                "is that the noise amplitude isn't a free dial: "
                "it's locked to gamma and kT by the "
                "fluctuation-dissipation theorem, so the system always "
                "settles at the correct thermal equilibrium no matter "
                "how large gamma gets. Raising gamma just slows the dance down "
                "without changing which structures are the stable ones."},
            {"type": "body", "text":
                "The total potential energy U is a sum of six terms, "
                "each one its own little physical story. "
                "We'll walk through each below — its mathematical form, "
                "what it represents biologically, and what happens when "
                "you turn its dial up or down."},

            {"type": "subheading", "text": "1. Backbone bonds"},
            {"type": "body", "text":
                "The most basic rule in the whole game: consecutive beads "
                "have to stay connected. The backbone bond is a harmonic "
                "spring between bead i and bead i+1, with a preferred "
                "length b_0 = 1 and stiffness k_bond:"},
            {"type": "equation",
             "latex": r"U_{\mathrm{bond}} = \frac{k_{\mathrm{bond}}}{2} \sum_{i=0}^{N-2} \left(|\mathbf{r}_{i+1}-\mathbf{r}_i| - b_0\right)^2",
             "label": "backbone"},
            {"type": "body", "text":
                "It pulls back when stretched and pushes back when "
                "squeezed — exactly like a rubber band with a favourite "
                "length. k_bond needs to be stiff enough that bonds "
                "stay close to b_0 at all times, but not so stiff "
                "that the timestep dt goes unstable underneath it. "
                "The safe zone is dt · k_bond / gamma < 0.5. "
                "Intuition: strip away every other force, and the "
                "backbone alone gives you a freely-jointed "
                "chain — plain old random-walk spaghetti."},

            {"type": "subheading", "text": "2. Bending stiffness"},
            {"type": "body", "text":
                "Real chromatin isn't loose spaghetti, either — it has a "
                "persistence length of roughly 50 nm (about 150 bp), "
                "meaning it resists sharp bends over short stretches. "
                "The Kratky-Porod bending term penalises kinks "
                "between consecutive bond vectors, where phi_i is the "
                "angle between bond (i, i+1) and bond (i+1, i+2):"},
            {"type": "equation",
             "latex": r"U_{\mathrm{bend}} = k_{\mathrm{angle}} \sum_{i=0}^{N-3} (1 - \cos\varphi_i)",
             "label": "Kratky-Porod bending"},
            {"type": "body", "text":
                "When phi = 0 the chain runs perfectly straight and "
                "U_bend = 0 — no complaints. When phi = pi it's folded "
                "all the way back on itself, and U_bend = 2·k_angle, "
                "the maximum penalty it can levy. The persistence length "
                "in bead units works out to roughly k_angle / kT. A larger "
                "k_angle gives you a stiffer, smoother chain that looks "
                "more like a real chromatin fibre and less like a "
                "crumpled receipt. "
                "Intuition: this is the dial that decides whether your "
                "chain reads as a calm worm or a tangled mess."},

            {"type": "subheading", "text": "3. Excluded volume"},
            {"type": "body", "text":
                "Two beads simply can't sit on top of each other. "
                "The excluded volume term is a soft, polite repulsion "
                "between all non-bonded bead pairs i and j with "
                "|i-j| >= 2, switching on only once their distance r "
                "dips below the exclusion diameter r_c:"},
            {"type": "equation",
             "latex": r"U_{\mathrm{EV}}(r) = \varepsilon_{\mathrm{EV}}\!\left(1 - \frac{r}{r_c}\right)^{\!2}, \quad r < r_c",
             "label": "soft excluded volume"},
            {"type": "body", "text":
                "The softness is a deliberate choice: beads can still "
                "overlap a little if thermal energy pushes hard enough, "
                "which means the chain isn't topologically locked — "
                "strands are free to slip through one another. That "
                "matches real biology nicely (topoisomerases are busy "
                "untangling chromatin in the cell all the time), and it "
                "makes the simulation run far faster than a strict "
                "hard-core model would. "
                "Intuition: eps_EV controls how firmly beads shove each "
                "other apart. Too low and everything collapses into a "
                "point; too high and the chain flies apart. The sweet "
                "spot gives a tidy, well-spaced, self-avoiding-looking "
                "polymer where every bead gets its own personal space."},

            {"type": "subheading", "text": "4. Loop bonds"},
            {"type": "body", "text":
                "Every loop you place adds a harmonic spring between "
                "two non-consecutive beads (m_k, n_k) with |m_k-n_k| >= 3 — "
                "your hand-placed stand-in for a cohesin complex "
                "clamping two distant genomic loci together:"},
            {"type": "equation",
             "latex": r"U_{\mathrm{loop}} = \frac{k_{\mathrm{loop}}}{2} \sum_{(m,n)\in\mathcal{L}} \left(|\mathbf{r}_m - \mathbf{r}_n| - b_0\right)^2",
             "label": "loop bonds"},
            {"type": "body", "text":
                "The loop spring shares the same equilibrium length b_0 "
                "as the backbone, so it tries to bring anchors m and "
                "n into direct contact. Everything in between gets "
                "pulled along for the ride, forming a tidy domain — your TAD. "
                "On the contact map this shows up as a bright square "
                "on the diagonal, with a corner dot at (m, n) marking "
                "the spot. "
                "Intuition: placing a loop is like pinching two beads "
                "of a necklace together — everything between them "
                "loops out into its own little bubble."},

            {"type": "subheading", "text": "5. Block-copolymer attraction"},
            {"type": "body", "text":
                "This is the engine behind compartmentalisation, the "
                "big one. A Gaussian pairwise attraction acts between "
                "all non-bonded bead pairs, with a well depth eps that "
                "depends on the epigenetic types of the two beads involved:"},
            {"type": "equation",
             "latex": r"U_{\mathrm{cp}}(i,j) = -\varepsilon_{t_i t_j}\exp\!\left(-\frac{|\mathbf{r}_i-\mathbf{r}_j|^2}{2\sigma^2}\right)",
             "label": "block-copolymer"},
            {"type": "bullet", "items": [
                "eps_BB  (B-B, strong)  ->  blue beads huddle into a "
                "dense heterochromatin core near the nuclear lamina",
                "eps_AA  (A-A, weak)  ->  red beads cluster loosely "
                "in the interior, staying open and easy to reach",
                "eps_AB  (A-B, negative)  ->  Flory-Huggins "
                "incompatibility: mixed pairs give each other a gentle "
                "shove, sharpening the border between phases",
                "eps_domain  (inside a loop)  ->  a little extra cohesion "
                "for beads enclosed by a loop, reinforcing the TAD boundary",
            ]},
            {"type": "body", "text":
                "Being Gaussian, the attraction has a natural "
                "range sigma: beads closer than sigma feel a strong tug, "
                "beads farther than 2x or 3x that range barely notice each other. "
                "Put strong B-B attraction together with repulsive A-B, "
                "and you get liquid-liquid phase separation for free: a "
                "dense blue core wrapped in a looser red shell. On the "
                "contact map this produces the plaid checkerboard — the "
                "unmistakable signature of A/B compartmentalisation. "
                "Intuition: this is the force you're steering with the "
                "colour ribbon. Paint a bead blue and it starts angling "
                "toward the B phase; paint it red and it drifts back "
                "toward A."},

            {"type": "subheading", "text": "6. Confinement"},
            {"type": "body", "text":
                "The chain has to stay inside the nucleus — it can't just "
                "wander off. Soft harmonic walls confine each bead to a "
                "cube of half-side L ~ N^(1/3), keeping bead density "
                "roughly constant no matter how long the chain gets. "
                "Only beads that stray outside the box feel anything; "
                "inside, the walls are completely invisible:"},
            {"type": "equation",
             "latex": r"U_{\mathrm{wall}} = \frac{k_w}{2} \sum_{\alpha \in \{x,y,z\}} \max(0,\,|x_\alpha|-L)^2",
             "label": "soft confinement"},
            {"type": "body", "text":
                "Being soft, the walls let a very energetic bead make a "
                "brief escape — the same forgiving spirit as the soft "
                "excluded volume above. In practice beads rarely wander "
                "outside the box at normal settings. "
                "Intuition: think of the box as the nuclear envelope. "
                "Switch it off (k_w = 0 in the MiNI-Lab) and the chain "
                "drifts freely into open space, where compartment "
                "formation gets noticeably weaker — there's no confining "
                "pressure left to squeeze the phases together."},
        ]
    },
    {
        "title": "Validation",
        "icon": "V",
        "blocks": [
            {"type": "heading", "text": "Comparing Simulation to Experiment"},
            {"type": "body", "text":
                "When you press Measure, the game lets the simulation "
                "run for a good long while and averages the contact map "
                "over hundreds of conformations. That ensemble average — "
                "never any single snapshot — is what gets compared to "
                "the experimental target. One structure is never a "
                "Hi-C map; Hi-C is always a population telling its story "
                "together."},
            {"type": "body", "text":
                "The comparison pipeline mirrors exactly what real Hi-C "
                "analysts do: normalise away the distance decay, work out "
                "the correlation structure, pull out the compartment "
                "eigenvector, and measure how enriched the loops are. "
                "Each step peels back a different layer of the signal, "
                "and each one earns its own score. The final score is a "
                "weighted sum built so that nailing compartments while "
                "ignoring loops, or the reverse, simply isn't "
                "enough — you need both working together."},

            {"type": "subheading", "text": "Step 1 — Observed over expected (O/E)"},
            {"type": "body", "text":
                "The raw contact map P has one loud, boring feature: "
                "nearby loci are always in contact simply because "
                "they're close along the chain. That distance decay drowns "
                "out the interesting biology, so the first step divides "
                "each entry P(i,j) by the genome-distance-averaged mean "
                "contact frequency at separation |i-j|:"},
            {"type": "equation",
             "latex": r"OE_{ij} = \frac{P_{ij}}{\langle P_{|i-j|}\rangle}",
             "label": "O/E normalisation"},
            {"type": "body", "text":
                "After O/E normalisation, a value of 1.0 means the "
                "contact frequency is exactly what you'd expect "
                "by chance at that distance — nothing to see here. Values "
                "above 1.0 are the interesting bit: enrichment, two loci "
                "sitting closer than chance would allow, either thanks to "
                "a loop or a shared compartment. Values below 1.0 signal "
                "depletion — loci actively keeping their distance across "
                "an A/B boundary. "
                "Intuition: O/E is the Hi-C equivalent of subtracting "
                "the background noise. It's what finally makes the "
                "checkerboard and the TAD squares pop into view."},

            {"type": "subheading", "text": "Step 2 — Pearson correlation matrix"},
            {"type": "body", "text":
                "With the distance decay out of the way, the next step "
                "computes the Pearson correlation between every pair "
                "of rows in the OE matrix. Two loci i and j score a "
                "high correlation when their contact profiles look "
                "alike — they're keeping company with the same crowd of "
                "other loci across the genome:"},
            {"type": "equation",
             "latex": r"C_{ij} = \mathrm{corr}(OE_{i\cdot},\; OE_{j\cdot})",
             "label": "Pearson correlation matrix"},
            {"type": "body", "text":
                "This is where the plaid pattern really starts to pop. "
                "Loci sharing an A or B compartment end up with nearly "
                "identical contact profiles — same friends, same "
                "avoidances. So C(i,j) runs high when i and j share "
                "a compartment, and dips negative when they don't. The "
                "resulting matrix looks like a chessboard "
                "of positive and negative squares — the compartment "
                "structure laid bare for all to see. "
                "Intuition: if O/E removes the distance effect, the "
                "correlation matrix removes the individual noise "
                "and reveals the collective A/B structure underneath."},

            {"type": "subheading", "text": "Step 3 — First eigenvector E1"},
            {"type": "body", "text":
                "The dominant pattern hiding in the correlation matrix C "
                "is captured by its first eigenvector, E1 — a single "
                "number per locus summing up which compartment it "
                "belongs to. Positive E1 means an A-type bead (red, "
                "active); negative means B-type (blue, inactive). "
                "The sign gets anchored using GC content in real data, "
                "or bead type here in the simulation."},
            {"type": "body", "text":
                "E1 is the single most readable output of the whole "
                "Hi-C pipeline. When you look at the E1 track under "
                "the heatmap, you're looking straight at the genome-wide "
                "compartment landscape — the boundary between red and "
                "blue is the boundary between euchromatin and "
                "heterochromatin. Matching your E1 to the target's E1 is "
                "the heart of the compartment move. "
                "Intuition: E1 is the Hi-C equivalent of colouring in "
                "the genome by hand — it tells you, locus by locus, how "
                "A-like or B-like its neighbourhood really is."},

            {"type": "subheading", "text": "Score 1 — Stratum-adjusted correlation (SCC)"},
            {"type": "body", "text":
                "SCC is the headline score, and the one that's hardest "
                "to fake. A naive Pearson correlation between the full "
                "P_sim and P_exp matrices would be dominated by the "
                "distance decay alone — two completely random polymers "
                "would score suspiciously well just by sharing the right "
                "diagonal shape. SCC sidesteps this by computing "
                "correlations separately at each genomic distance "
                "(each stratum), then combining them with weights "
                "proportional to the variance at each one:"},
            {"type": "equation",
             "latex": r"\mathrm{SCC} = \sum_s w_s\,\rho_s(P_{\mathrm{sim}},\,P_{\mathrm{exp}}),\quad w_s \propto \sqrt{\mathrm{var}_s(P_{\mathrm{sim}})\cdot\mathrm{var}_s(P_{\mathrm{exp}})}",
             "label": "SCC"},
            {"type": "body", "text":
                "A stratum with high variance earns a big weight — "
                "it's information-rich and worth getting right. "
                "A stratum where both maps are nearly flat earns barely "
                "any weight — there's nothing much to compare there anyway. "
                "SCC ranges from -1 (perfectly backwards) "
                "to +1 (a flawless match). In practice, a solid polymer "
                "model lands around SCC ~ 0.6-0.9 on real Hi-C data, so "
                "don't feel bad if you're not hitting 1.0. "
                "Intuition: SCC asks whether the texture of "
                "your map matches the texture of the target — at "
                "every distance scale, all at once."},

            {"type": "subheading", "text": "Score 2 — E1 correlation"},
            {"type": "body", "text":
                "Even a high SCC can hide a wrong compartment pattern "
                "underneath. E1 correlation checks directly "
                "whether your eigenvector matches the target's:"},
            {"type": "equation",
             "latex": r"r_{E1} = \mathrm{corr}(E1_{\mathrm{sim}},\; E1_{\mathrm{exp}})",
             "label": "compartment score"},
            {"type": "body", "text":
                "r_E1 = +1 means a perfect A/B match. "
                "r_E1 = -1 means you nailed the block structure but "
                "swapped every colour — close, but inverted. "
                "r_E1 ~ 0 means your compartment landscape has nothing "
                "to do with the target's. "
                "Intuition: this is the score that answers directly to "
                "the colour ribbon. Get the block boundaries right and "
                "r_E1 climbs fast. Get them wrong and the heatmap can "
                "still look convincing (it'll still show a checkerboard) "
                "while r_E1 quietly sits low, or even negative."},

            {"type": "subheading", "text": "Score 3 — Checkerboard correlation"},
            {"type": "body", "text":
                "A companion compartment score that works directly "
                "on the correlation matrix C rather than squeezing it "
                "down to one eigenvector. It compares C_sim and C_exp "
                "on every off-diagonal entry |i-j| > 1, capturing the "
                "whole plaid pattern — not just its headline component:"},
            {"type": "equation",
             "latex": r"r_{\mathrm{cb}} = \mathrm{corr}(C_{\mathrm{sim}}[\mathrm{off}],\; C_{\mathrm{exp}}[\mathrm{off}])",
             "label": "checkerboard"},
            {"type": "body", "text":
                "This score catches fine compartment structure that "
                "E1 alone might miss — sub-compartments, gradients "
                "within an A or B domain, how sharp the A/B boundary "
                "really is. "
                "Intuition: if r_E1 is the coarse read of compartments, "
                "r_cb is the fine print. You'll want both running high "
                "for full marks on the compartment side."},

            {"type": "subheading", "text": "Score 4 — Anchor F1"},
            {"type": "body", "text":
                "Anchor F1 checks loop placement accuracy head-on, "
                "comparing your loop anchors to the true anchors "
                "within ±1 bead of tolerance:"},
            {"type": "equation",
             "latex": r"F_1 = \frac{2\,\cdot\,\mathrm{precision}\,\cdot\,\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}",
             "label": "anchor placement accuracy"},
            {"type": "body", "text":
                "Precision is the fraction of your loops that land "
                "near a true anchor. Recall is the fraction of true "
                "anchors you actually managed to cover. F1 is their "
                "harmonic mean, so it only climbs when both precision "
                "and recall do. Scatter 50 correct loops among 50 wrong "
                "ones and precision tanks; find only 2 of the 10 true "
                "anchors and recall tanks instead. "
                "Intuition: F1 punishes both scattering loops everywhere "
                "and being too timid to place enough. Read the green "
                "dots carefully, and place with intent."},

            {"type": "subheading", "text": "The total score"},
            {"type": "body", "text":
                "All four scores combine into one weighted "
                "sum, scaled to run from 0 to 100. The weights are "
                "chosen so compartments and loops pull roughly equal "
                "weight — you can't coast by ignoring either one:"},
            {"type": "equation",
             "latex": r"\mathrm{Score} = w_1\cdot\mathrm{SCC} + w_2\cdot r_{E1} + w_3\cdot r_{\mathrm{cb}} + w_4\cdot F_1",
             "label": "weighted total (0-100)"},
            {"type": "body", "text":
                "In two-player mode the scores go their separate ways: "
                "the loop player is scored only on F1; the "
                "compartment player only on r_E1 and r_cb. That keeps "
                "the competition genuinely independent — a perfect loop "
                "layout with random colours scores well on loops and "
                "near zero on compartments, and vice versa. The two "
                "mechanisms are separate in the biology, and they stay "
                "separate in the scoring too."},
        ]
    },
    {
        "title": "Structural",
        "icon": "S",
        "blocks": [
            {"type": "heading", "text": "Live Structural Metrics"},
            {"type": "body", "text":
                "The analysis panel tracks six quantities in real time "
                "as the simulation runs, computed straight from the bead "
                "positions — no contact map required — and refreshed "
                "every few frames. Together they give you an instant "
                "physical read on what the polymer is actually doing, "
                "independent of how well it matches the target. "
                "Think of them as the polymer's vital signs on a monitor."},

            {"type": "subheading", "text": "Radius of gyration  Rg"},
            {"type": "equation",
             "latex": r"R_g = \sqrt{\frac{1}{N}\sum_i |\mathbf{r}_i - \bar{\mathbf{r}}|^2}",
             "label": "compactness"},
            {"type": "body", "text":
                "Rg is the RMS distance of every bead from the chain's "
                "centre of mass — arguably the single most useful "
                "number on the panel. Small Rg means a compact, globular "
                "chain (attractions are winning); large Rg means an open, "
                "stretched-out one (excluded volume or backbone stiffness "
                "is winning instead). "
                "At thermal equilibrium, polymer theory predicts "
                "Rg ~ N^nu, with nu ~ 1/3 for a collapsed globule "
                "and nu ~ 3/5 for a swollen, self-avoiding coil. "
                "Watch Rg drop as you paint on more B beads — the "
                "strong B-B attraction reels the chain in — and "
                "climb again as you raise the excluded volume diameter."},

            {"type": "subheading", "text": "End-to-end distance  R_ee"},
            {"type": "equation",
             "latex": r"R_{ee} = |\mathbf{r}_{N-1} - \mathbf{r}_0|",
             "label": "chain extent"},
            {"type": "body", "text":
                "Simply the straight-line distance between the very "
                "first and very last bead. For a freely jointed ideal "
                "chain, R_ee^2 = N · b_0² on average. Add loops and "
                "attractions, and R_ee can shrink dramatically — a loop "
                "tying bead 0 to bead N-1 would pull it almost to zero. "
                "Unlike Rg, R_ee jitters around quite a bit frame to "
                "frame as the two ends wander, so its time-series plot "
                "will look noisier than the others. Watch its overall "
                "trend, not every little spike."},

            {"type": "subheading", "text": "Mean bond length <b>"},
            {"type": "equation",
             "latex": r"\langle b \rangle = \frac{1}{N-1}\sum_i |\mathbf{r}_{i+1}-\mathbf{r}_i|",
             "label": "integrator health"},
            {"type": "body", "text":
                "The average length across every backbone bond. It "
                "should hover close to the equilibrium value b_0 = 1 at "
                "all times — this is the first gauge to check if the "
                "simulation starts looking off. If <b> drifts past "
                "about 1.2, the integrator is coming unglued — "
                "forces have grown so large each step is overshooting. "
                "The fix is always the same trio: lower dt, raise gamma, or "
                "lower k_bond, until dt · k_bond / gamma < 0.5 again. "
                "Intuition: a healthy simulation has bonds that "
                "wobble gently around b_0. An unstable one has "
                "bonds stretching further and further until positions "
                "turn to NaN and the simulation quietly gives up."},

            {"type": "subheading", "text": "Contact count  Nc"},
            {"type": "equation",
             "latex": r"N_c = |\{(i,j):\,|i-j| \geq 2,\;P_{ij}>0.5\}|",
             "label": "crowding"},
            {"type": "body", "text":
                "The number of non-bonded bead pairs whose sigmoid "
                "contact probability crosses 0.5 — a clean, binary "
                "reading of how crowded the chain has gotten. An "
                "extended chain has O(N) contacts (mostly between "
                "neighbours); a collapsed globule racks up O(N²), since "
                "nearly every bead ends up near every other. "
                "Nc climbs quickly as you add B beads or crank up the "
                "copolymer attraction, and falls as you raise the "
                "excluded volume height. It's the most direct way to "
                "catch a sudden collapse in the act — more immediate "
                "than Rg for spotting it the moment it happens."},

            {"type": "subheading", "text": "Asphericity (delta)"},
            {"type": "equation",
             "latex": r"\delta = \frac{(\lambda_1-\lambda_2)^2+(\lambda_2-\lambda_3)^2+(\lambda_1-\lambda_3)^2}{2(\lambda_1+\lambda_2+\lambda_3)^2}",
             "label": "shape  [0 = sphere,  1 = rod]"},
            {"type": "body", "text":
                "Asphericity measures how far the chain's overall shape "
                "strays from a perfect sphere, using the three "
                "eigenvalues lambda_1 <= lambda_2 <= lambda_3 of the 3×3 gyration "
                "tensor — the polymer's own version of the inertia "
                "tensor. Equal eigenvalues mean a spherically symmetric "
                "cloud of beads and delta = 0. All the mass strung out "
                "along one axis, like a rod, and delta = 1. "
                "A typical looped, compartmentalised chromatin "
                "model lands around delta = 0.3-0.5 — not quite a sphere, "
                "not quite a rod, more of a gently flattened ellipsoid. "
                "High delta tends to show up when long loops stretch the "
                "chain along one direction, or when a stiff backbone "
                "dominates before the chain has had time to settle."},

            {"type": "subheading", "text": "Compaction index (kappa)"},
            {"type": "equation",
             "latex": r"\kappa = \frac{R_g}{R_g^{\mathrm{ideal}}}, \qquad R_g^{\mathrm{ideal}} = b_0\sqrt{N/6}",
             "label": "vs ideal Gaussian chain"},
            {"type": "body", "text":
                "The compaction index kappa puts Rg into perspective by "
                "comparing it to what a Gaussian (ideal, non-interacting) "
                "chain of the same length would give. kappa = 1 means your "
                "chain is exactly as spread out as an ideal random "
                "walk — no better, no worse. kappa < 1 means it's more "
                "compact, with attractions calling the shots. kappa > 1 "
                "means it's more swollen, with excluded volume or "
                "backbone stiffness in charge instead. For biological "
                "chromatin at the scale this game plays with, kappa usually "
                "sits around 0.3-0.7. "
                "Watching kappa over time is a quick, scale-free way to tell "
                "whether a parameter change is collapsing the chain "
                "(kappa falls) or opening it back up (kappa rises)."},
        ]
    },
]
