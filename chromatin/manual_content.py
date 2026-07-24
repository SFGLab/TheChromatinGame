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
                "Most of physics runs forward: given the rules, predict "
                "the outcome. Chromatin folding runs the other way — you "
                "are handed the measurement and asked to recover the "
                "structure that produced it. This is an inverse problem, "
                "and they are notoriously hard."},
            {"type": "subheading", "text": "Macrostates and microstates"},
            {"type": "body", "text":
                "A microstate is the full detailed description of a system "
                "— every atom's position, every molecule's conformation. "
                "A macrostate is what you can actually observe: a coarse, "
                "averaged projection. Temperature is the macrostate of "
                "molecular velocities; a Hi-C heatmap is the macrostate "
                "of millions of individual chromatin fibres. "
                "The forward direction micro → macro is usually easy. "
                "The reverse is where the trouble begins, because many "
                "different microstates can produce the same macrostate."},
            {"type": "subheading", "text": "Hi-C as an ensemble average"},
            {"type": "body", "text":
                "Hi-C cross-links DNA, fragments it, and counts how often "
                "pairs of loci end up ligated — a proxy for spatial "
                "proximity. The contact matrix P, where P(i,j) is the "
                "probability that loci i and j were close in space, is "
                "measured across millions of cells simultaneously. "
                "The heatmap is therefore an ensemble average, not a "
                "snapshot. Many different ensembles of structures can "
                "produce the same heatmap. The problem has no unique "
                "answer."},
            {"type": "subheading", "text": "Ill-posed by definition"},
            {"type": "body", "text":
                "A well-posed problem (Hadamard, 1902) has a solution "
                "that exists, is unique, and changes continuously with "
                "the data. Inverse problems fail on all three counts. "
                "For chromatin: not every heatmap corresponds to a "
                "realisable polymer; many polymers produce the same "
                "heatmap; and small measurement errors can lead to "
                "wildly different reconstructions. What we can do is "
                "constrain the problem with a physical model and ask "
                "which parameters make the simulated map best match "
                "the observed one — which is exactly what this game "
                "asks you to do."},
            {"type": "subheading", "text": "Why we built this"},
            {"type": "body", "text":
                "The chromatin field draws in physicists, biologists, "
                "mathematicians, and computer scientists. Many people "
                "working with Hi-C data have never seen a polymer "
                "simulation; many polymer physicists have limited "
                "intuition for what a contact map encodes biologically. "
                "This game exists to build that intuition interactively. "
                "Place loops, paint compartments, watch the heatmap "
                "respond. The physics is real — the same models and "
                "analysis pipeline used in the research literature. "
                "The goal is not to solve the inverse problem, "
                "but to feel why it is hard and why it matters."},
        ]
    },
    {
        "title": "Polymer States",
        "icon": "P",
        "blocks": [
            {"type": "heading", "text": "Chromatin as a Spatial Graph"},
            {"type": "body", "text":
                "The chromatin fibre is modelled as a chain of N beads "
                "connected by a backbone. Each bead i carries two "
                "independent biological properties: a loop state "
                "describing which long-range bonds it participates in, "
                "and an epigenetic state sᵢ describing its local "
                "chromatin chemistry. Together they define a spatial "
                "graph whose 3D embedding produces the contact map."},
            {"type": "subheading", "text": "Cohesin and loop extrusion"},
            {"type": "body", "text":
                "Loops are long-range interactions mediated by cohesin, "
                "a ring-shaped protein complex. Cohesin encircles the "
                "chromatin fibre and acts like a molecular motor: it "
                "extrudes a growing loop of DNA until it stalls at a "
                "CTCF boundary element. The result is a stable loop "
                "anchored at two genomic positions (mₖ, nₖ). "
                "The full loop state ℒ of the polymer is the set "
                "of all such pairs, where each anchor must be at "
                "least 3 beads apart:"},
            {"type": "equation",
             "latex": r"\mathcal{L} = \{(m_k, n_k) : 0 \leq m_k < n_k \leq N-1,\; n_k - m_k \geq 3\}",
             "label": "loop microstate"},
            {"type": "body", "text":
                "Each pair (mₖ, nₖ) appears as a bright corner dot "
                "in the Hi-C heatmap, surrounded by a square of "
                "elevated contact frequency — the TAD."},
            {"type": "subheading", "text": "Epigenetic marks and compartments"},
            {"type": "body", "text":
                "Every bead i carries an epigenetic state sᵢ ∈ {A, B}. "
                "Epigenetic marks are chemical modifications to histones "
                "that alter how tightly chromatin is packed and which "
                "proteins are recruited. Active euchromatin (A, red) "
                "is gene-rich, transcriptionally active, and associates "
                "with nuclear pores. Inactive heterochromatin (B, blue) "
                "is gene-poor, silenced, and clusters near the nuclear "
                "lamina."},
            {"type": "equation",
             "latex": r"s_i \in \{A,\, B\}, \quad i = 0, 1, \ldots, N-1",
             "label": "epigenetic microstate"},
            {"type": "body", "text":
                "Chromatin segments in the same epigenetic state attract "
                "each other because they recruit the same proteins — "
                "a process analogous to liquid-liquid phase separation. "
                "This is captured by a block-copolymer interaction "
                "with well depths ε_AA, ε_BB, and ε_AB:"},
            {"type": "equation",
             "latex": r"\varepsilon_{s_i s_j}: \quad \varepsilon_{AA} > 0,\quad \varepsilon_{BB} \gg \varepsilon_{AA},\quad \varepsilon_{AB} < 0",
             "label": "epigenetic interaction strengths"},
            {"type": "body", "text":
                "The strong B-B attraction (ε_BB) collapses blue beads "
                "into a dense core. The negative ε_AB encodes "
                "Flory-Huggins incompatibility — A and B beads repel "
                "each other, reinforcing phase separation. The "
                "macroscopic result is the plaid checkerboard in Hi-C: "
                "loci in the same compartment have elevated P(i,j), "
                "loci of opposite type are depleted."},
            {"type": "subheading", "text": "The full microstate"},
            {"type": "body", "text":
                "The complete description of the polymer at any instant "
                "is the triplet of bead positions {rᵢ}, the loop "
                "state ℒ, and the epigenetic state {sᵢ}:"},
            {"type": "equation",
             "latex": r"\Omega = (\{\mathbf{r}_i\}_{i=0}^{N-1},\;\mathcal{L},\;\{s_i\}_{i=0}^{N-1})",
             "label": "full polymer microstate"},
            {"type": "body", "text":
                "The positions {rᵢ} evolve under Langevin dynamics. "
                "The loop state ℒ and epigenetic state {sᵢ} are your "
                "two moves — you set them, and the simulation explores "
                "the conformational space consistent with your choices."},
        ]
    },
    {
        "title": "Force Field",
        "icon": "F",
        "blocks": [
            {"type": "heading", "text": "The Force Field"},
            {"type": "body", "text":
                "A force field is a set of rules that assigns a potential "
                "energy U to every possible configuration of the system. "
                "From Newton's second law, the force on each particle "
                "is the negative gradient of that energy. "
                "Where the energy landscape slopes downhill, the force "
                "pushes the particle in that direction. Where the "
                "landscape is flat, there is no force. The particle "
                "always tends to roll toward lower energy, while "
                "thermal fluctuations kick it around and allow it to "
                "explore nearby configurations."},
            {"type": "body", "text":
                "In classical molecular dynamics this gives Newton's "
                "equation m·ẍ = −∇U. For a polymer in a viscous "
                "environment like the cell nucleus, inertia is "
                "negligible — the friction from the surrounding medium "
                "is so large that the particle forgets its velocity "
                "almost instantly. The equation of motion simplifies "
                "to the overdamped Langevin equation, where the "
                "velocity (not the acceleration) is proportional to "
                "the force:"},
            {"type": "equation",
             "latex": r"\frac{d\mathbf{r}}{dt} = \frac{-\nabla U(\mathbf{r})}{\gamma} + \sqrt{\frac{2k_BT}{\gamma}}\;\boldsymbol{\xi}(t)",
             "label": "overdamped Langevin"},
            {"type": "body", "text":
                "Here γ is the friction coefficient — it sets how "
                "quickly the system responds to forces. kT is thermal "
                "energy and ξ(t) is Gaussian white noise representing "
                "random collisions with the solvent. The key insight "
                "is that the noise amplitude is not a free parameter: "
                "it is fixed by γ and kT through the "
                "fluctuation-dissipation theorem, ensuring the system "
                "reaches the correct thermal equilibrium regardless "
                "of how large γ is. Raising γ slows the dynamics "
                "without changing which structures are stable."},
            {"type": "body", "text":
                "The total potential energy U is a sum of six terms, "
                "each capturing a different physical interaction. "
                "We describe each one below — its mathematical form, "
                "what it models biologically, and what happens when "
                "you turn it up or down."},

            {"type": "subheading", "text": "1. Backbone bonds"},
            {"type": "body", "text":
                "The most fundamental constraint: consecutive beads "
                "must stay connected. The backbone bond is a harmonic "
                "spring between bead i and bead i+1, with equilibrium "
                "length b₀ = 1 and stiffness k_bond:"},
            {"type": "equation",
             "latex": r"U_{\mathrm{bond}} = \frac{k_{\mathrm{bond}}}{2} \sum_{i=0}^{N-2} \left(|\mathbf{r}_{i+1}-\mathbf{r}_i| - b_0\right)^2",
             "label": "backbone"},
            {"type": "body", "text":
                "This term is always attractive when bonds are "
                "stretched and repulsive when compressed — it acts "
                "like a rubber band with a preferred length. "
                "The stiffness k_bond must be large enough that bonds "
                "stay close to b₀ at all times, but not so large "
                "that the timestep dt becomes unstable. "
                "The stability condition is dt · k_bond / γ < 0.5. "
                "Intuition: if you could remove all other forces, "
                "the backbone alone would give you a freely-jointed "
                "chain — a random walk in 3D."},

            {"type": "subheading", "text": "2. Bending stiffness"},
            {"type": "body", "text":
                "Real chromatin is not freely jointed — it has a "
                "persistence length of roughly 50 nm (about 150 bp), "
                "meaning it resists bending on short length scales. "
                "The Kratky-Porod bending term penalises sharp kinks "
                "between consecutive bond vectors, where φᵢ is the "
                "angle between bond (i, i+1) and bond (i+1, i+2):"},
            {"type": "equation",
             "latex": r"U_{\mathrm{bend}} = k_{\mathrm{angle}} \sum_{i=0}^{N-3} (1 - \cos\varphi_i)",
             "label": "Kratky-Porod bending"},
            {"type": "body", "text":
                "When φ = 0 the chain is perfectly straight and "
                "U_bend = 0. When φ = π the chain folds back on "
                "itself and U_bend = 2·k_angle — the maximum penalty. "
                "The persistence length in bead units is approximately "
                "k_angle / kT. Larger k_angle gives a stiffer, "
                "smoother chain that looks more like a real chromatin "
                "fibre and less like a crumpled string. "
                "Intuition: this is what makes the chain look like "
                "a worm rather than a tangled mess."},

            {"type": "subheading", "text": "3. Excluded volume"},
            {"type": "body", "text":
                "Two beads cannot occupy the same point in space. "
                "The excluded volume term is a soft-core repulsion "
                "between all non-bonded bead pairs i and j with "
                "|i−j| ≥ 2, active only when their distance r "
                "is less than the exclusion diameter r_c:"},
            {"type": "equation",
             "latex": r"U_{\mathrm{EV}}(r) = \varepsilon_{\mathrm{EV}}\!\left(1 - \frac{r}{r_c}\right)^{\!2}, \quad r < r_c",
             "label": "soft excluded volume"},
            {"type": "body", "text":
                "The softness is intentional: beads can overlap "
                "if the thermal energy is large enough, which means "
                "the chain is not topologically constrained — strands "
                "can pass through each other. This matches the biology "
                "(topoisomerases actively resolve chromatin tangles "
                "in the cell) and makes the simulation far faster "
                "than a hard-core model would be. "
                "Intuition: ε_EV controls how much beads push each "
                "other apart. Too low and beads collapse into a "
                "point; too high and the chain explodes. The sweet "
                "spot gives a well-separated, self-avoiding-like "
                "polymer where you can distinguish individual beads."},

            {"type": "subheading", "text": "4. Loop bonds"},
            {"type": "body", "text":
                "Each loop you place adds a harmonic spring between "
                "two non-consecutive beads (mₖ, nₖ) with |mₖ−nₖ| ≥ 3. "
                "This mimics the mechanical effect of a cohesin "
                "complex holding two distant genomic loci together:"},
            {"type": "equation",
             "latex": r"U_{\mathrm{loop}} = \frac{k_{\mathrm{loop}}}{2} \sum_{(m,n)\in\mathcal{L}} \left(|\mathbf{r}_m - \mathbf{r}_n| - b_0\right)^2",
             "label": "loop bonds"},
            {"type": "body", "text":
                "The loop spring has the same equilibrium length b₀ "
                "as the backbone, so it tries to bring anchors m and "
                "n into direct contact. All beads between them are "
                "pulled inward, forming a compact domain — the TAD. "
                "In the contact map this appears as a bright square "
                "on the diagonal, with a corner dot at (m, n) marking "
                "the anchors. "
                "Intuition: adding a loop is like pinching two points "
                "of a necklace together — everything between them "
                "loops out into a separate bubble."},

            {"type": "subheading", "text": "5. Block-copolymer attraction"},
            {"type": "body", "text":
                "This is the compartmentalisation engine. A Gaussian "
                "pairwise attraction acts between all non-bonded bead "
                "pairs, with a well depth ε that depends on the "
                "epigenetic types of the two beads:"},
            {"type": "equation",
             "latex": r"U_{\mathrm{cp}}(i,j) = -\varepsilon_{t_i t_j}\exp\!\left(-\frac{|\mathbf{r}_i-\mathbf{r}_j|^2}{2\sigma^2}\right)",
             "label": "block-copolymer"},
            {"type": "bullet", "items": [
                "ε_BB  (B-B, strong)  →  blue beads collapse into a "
                "dense heterochromatin core near the nuclear lamina",
                "ε_AA  (A-A, weak)  →  red beads cluster loosely "
                "in the interior, staying more open and accessible",
                "ε_AB  (A-B, negative)  →  Flory-Huggins "
                "incompatibility: mixed pairs are slightly repulsive, "
                "sharpening the interface between phases",
                "ε_domain  (inside a loop)  →  extra cohesion for "
                "beads enclosed by a loop, reinforcing the TAD boundary",
            ]},
            {"type": "body", "text":
                "The Gaussian form means the attraction has a natural "
                "range σ: beads closer than σ feel a strong pull, "
                "beads farther than 2σ or 3σ feel almost nothing. "
                "The combination of strong B-B and negative A-B "
                "drives liquid-liquid phase separation: a dense blue "
                "core surrounded by a looser red shell. In the contact "
                "map this produces the plaid checkerboard — the "
                "hallmark of A/B compartmentalisation. "
                "Intuition: this is the term you control with the "
                "colour ribbon. Painting a bead blue makes it want "
                "to join the B phase; painting it red pulls it toward "
                "the A phase."},

            {"type": "subheading", "text": "6. Confinement"},
            {"type": "body", "text":
                "The chain must stay inside the nucleus. Soft harmonic "
                "walls confine each bead to a cube of half-side "
                "L ∝ N^(1/3), so that bead density stays "
                "N-independent as you change the chain length. "
                "Only beads that venture outside the box feel a force; "
                "inside the box the walls are invisible:"},
            {"type": "equation",
             "latex": r"U_{\mathrm{wall}} = \frac{k_w}{2} \sum_{\alpha \in \{x,y,z\}} \max(0,\,|x_\alpha|-L)^2",
             "label": "soft confinement"},
            {"type": "body", "text":
                "The softness means a very energetic bead can briefly "
                "escape — the same rationale as the soft excluded "
                "volume. In practice beads rarely leave the box "
                "at normal parameter values. "
                "Intuition: the box is the nuclear envelope. "
                "Removing it (k_w = 0 in the MiNI-Lab) lets the chain "
                "expand freely into space, which makes compartment "
                "formation much weaker because there is no confinement "
                "pressure forcing the phases together."},
        ]
    },
    {
        "title": "Validation",
        "icon": "V",
        "blocks": [
            {"type": "heading", "text": "Comparing Simulation to Experiment"},
            {"type": "body", "text":
                "When you press Measure, the game runs the simulation "
                "for a long time and averages the contact map over "
                "hundreds of conformations. This ensemble average — "
                "not any single snapshot — is what gets compared to "
                "the experimental target. One structure is never a "
                "Hi-C map; Hi-C is always a population average."},
            {"type": "body", "text":
                "The comparison pipeline mirrors what real Hi-C "
                "analysts do: normalise for distance decay, compute "
                "the correlation structure, extract the compartment "
                "eigenvector, and measure loop enrichment. Each step "
                "peels away a different layer of the signal, and each "
                "produces a score. The final score is a weighted sum "
                "designed so that getting compartments right without "
                "loops, or loops right without compartments, is "
                "not enough — you need both."},

            {"type": "subheading", "text": "Step 1 — Observed over expected (O/E)"},
            {"type": "body", "text":
                "The raw contact map P has a strong distance decay: "
                "nearby loci are always in contact simply because "
                "they are close along the chain. This obscures the "
                "interesting biology. The first step divides each "
                "entry P(i,j) by the genome-distance-averaged mean "
                "contact frequency at separation |i−j|:"},
            {"type": "equation",
             "latex": r"OE_{ij} = \frac{P_{ij}}{\langle P_{|i-j|}\rangle}",
             "label": "O/E normalisation"},
            {"type": "body", "text":
                "After O/E normalisation, a value of 1.0 means the "
                "contact frequency is exactly what you would expect "
                "by chance at that distance. Values above 1.0 signal "
                "enrichment — two loci are closer than expected, "
                "either because of a loop or because they share a "
                "compartment. Values below 1.0 signal depletion — "
                "loci that actively avoid each other across the "
                "A/B boundary. "
                "Intuition: O/E is the Hi-C equivalent of subtracting "
                "the background. It is what makes the checkerboard "
                "and the TAD squares visible."},

            {"type": "subheading", "text": "Step 2 — Pearson correlation matrix"},
            {"type": "body", "text":
                "Once the distance decay is removed, the next step "
                "computes the Pearson correlation between every pair "
                "of rows in the OE matrix. Two loci i and j get a "
                "high correlation if they have similar contact "
                "profiles — they interact with the same set of other "
                "loci across the genome:"},
            {"type": "equation",
             "latex": r"C_{ij} = \mathrm{corr}(OE_{i\cdot},\; OE_{j\cdot})",
             "label": "Pearson correlation matrix"},
            {"type": "body", "text":
                "This is where the plaid pattern becomes striking. "
                "Loci within the same A or B compartment have nearly "
                "identical contact profiles — they all interact with "
                "the same partners and avoid the same loci. So "
                "C(i,j) is high when i and j are in the same "
                "compartment, and negative when they are in different "
                "ones. The resulting matrix looks like a chessboard "
                "of positive and negative squares — the compartment "
                "structure laid bare. "
                "Intuition: if OE removes the distance effect, the "
                "correlation matrix removes the individual variation "
                "and reveals the collective A/B structure."},

            {"type": "subheading", "text": "Step 3 — First eigenvector E1"},
            {"type": "body", "text":
                "The dominant pattern in the correlation matrix C is "
                "captured by its first eigenvector E1. This is a "
                "single number per locus that summarises which "
                "compartment it belongs to. Positive E1 values "
                "correspond to A-type beads (red, active); negative "
                "values to B-type beads (blue, inactive). "
                "The sign is oriented using GC content in real data, "
                "or bead type in the simulation."},
            {"type": "body", "text":
                "E1 is the single most interpretable output of the "
                "Hi-C pipeline. When you look at the E1 track under "
                "the heatmap, you are looking at the genome-wide "
                "compartment landscape — the boundary between red and "
                "blue is the boundary between euchromatin and "
                "heterochromatin. Getting your E1 to match the target "
                "E1 is the core challenge of the compartment move. "
                "Intuition: E1 is the Hi-C equivalent of colouring "
                "the genome — it tells you, for each locus, how "
                "A-like or B-like its contact environment is."},

            {"type": "subheading", "text": "Score 1 — Stratum-adjusted correlation (SCC)"},
            {"type": "body", "text":
                "SCC is the headline score and the hardest to fool. "
                "A naive Pearson correlation between the full "
                "P_sim and P_exp matrices would be dominated by the "
                "distance decay — two random polymers would score "
                "well just by having the right diagonal structure. "
                "SCC avoids this by computing correlations separately "
                "at each genomic distance (stratum), then combining "
                "them with weights proportional to the variance at "
                "each stratum:"},
            {"type": "equation",
             "latex": r"\mathrm{SCC} = \sum_s w_s\,\rho_s(P_{\mathrm{sim}},\,P_{\mathrm{exp}}),\quad w_s \propto \sqrt{\mathrm{var}_s(P_{\mathrm{sim}})\cdot\mathrm{var}_s(P_{\mathrm{exp}})}",
             "label": "SCC"},
            {"type": "body", "text":
                "A stratum with high variance gets a large weight — "
                "it is informationally rich and worth matching. "
                "A stratum where both maps are nearly constant gets "
                "a small weight — there is nothing to compare. "
                "SCC ranges from -1 (perfectly anti-correlated) "
                "to +1 (perfect match). In practice, a good polymer "
                "model achieves SCC ~ 0.6-0.9 on real Hi-C data. "
                "Intuition: SCC is asking whether the texture of "
                "your map matches the texture of the target — at "
                "every distance scale simultaneously."},

            {"type": "subheading", "text": "Score 2 — E1 correlation"},
            {"type": "body", "text":
                "Even if SCC is high, the compartment pattern could "
                "still be wrong. E1 correlation directly checks "
                "whether your eigenvector matches the target:"},
            {"type": "equation",
             "latex": r"r_{E1} = \mathrm{corr}(E1_{\mathrm{sim}},\; E1_{\mathrm{exp}})",
             "label": "compartment score"},
            {"type": "body", "text":
                "r_E1 = +1 means your A/B pattern is a perfect match. "
                "r_E1 = -1 means it is perfectly inverted — you have "
                "the right block structure but every colour is wrong. "
                "r_E1 ≈ 0 means your compartment landscape is "
                "uncorrelated with the target. "
                "Intuition: this is the score that responds directly "
                "to the colour ribbon. Get the block boundaries right "
                "and r_E1 rises quickly. Get them wrong and the "
                "heatmap may still look plausible (it will still have "
                "a checkerboard) but r_E1 will be low or negative."},

            {"type": "subheading", "text": "Score 3 — Checkerboard correlation"},
            {"type": "body", "text":
                "A complementary compartment score that works directly "
                "on the correlation matrix C rather than its "
                "eigenvector. It compares C_sim and C_exp on all "
                "off-diagonal entries |i−j| > 1, capturing whether "
                "the full plaid pattern — not just its dominant "
                "component — is reproduced:"},
            {"type": "equation",
             "latex": r"r_{\mathrm{cb}} = \mathrm{corr}(C_{\mathrm{sim}}[\mathrm{off}],\; C_{\mathrm{exp}}[\mathrm{off}])",
             "label": "checkerboard"},
            {"type": "body", "text":
                "This score is sensitive to fine compartment structure "
                "that E1 alone might miss — sub-compartments, "
                "gradients within an A or B domain, and the "
                "sharpness of the A/B boundary. "
                "Intuition: if r_E1 is the coarse-grained compartment "
                "score, r_cb is the fine-grained one. Both need to "
                "be high for full marks on the compartment side."},

            {"type": "subheading", "text": "Score 4 — Aggregate peak analysis (APA)"},
            {"type": "body", "text":
                "APA measures loop enrichment. For each loop anchor "
                "pair (m, n) in the target, a small sub-matrix of "
                "P is extracted centred on (m, n) and all these "
                "sub-matrices are averaged. A correct loop produces "
                "a bright pixel at the centre surrounded by a "
                "dimmer background:"},
            {"type": "equation",
             "latex": r"\mathrm{APA} = \frac{P_{\mathrm{centre}}}{\langle P_{\mathrm{background}}\rangle}",
             "label": "loop enrichment"},
            {"type": "body", "text":
                "APA > 1 means there is loop enrichment — your "
                "polymer is being brought into contact at the right "
                "positions. APA ≈ 1 means no enrichment — the "
                "contacts at loop positions are no different from "
                "background. "
                "Intuition: APA is the score that rewards you for "
                "clicking the right cells in the heatmap. It does "
                "not care where exactly your loops are — only whether "
                "the positions that should be enriched in the target "
                "are also enriched in your simulation."},

            {"type": "subheading", "text": "Score 5 — Anchor F1"},
            {"type": "body", "text":
                "APA tells you about enrichment at target positions, "
                "but it does not penalise spurious loops — loops "
                "placed in completely wrong positions. Anchor F1 "
                "checks placement accuracy directly, comparing your "
                "loop anchors to the true anchors within ±1 bead "
                "tolerance:"},
            {"type": "equation",
             "latex": r"F_1 = \frac{2\,\cdot\,\mathrm{precision}\,\cdot\,\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}",
             "label": "anchor placement accuracy"},
            {"type": "body", "text":
                "Precision is the fraction of your loops that land "
                "near a true anchor. Recall is the fraction of true "
                "anchors that you have covered. F1 is the harmonic "
                "mean — it is only high if both precision and recall "
                "are high. Placing 50 correct loops and 50 wrong ones "
                "gives a low precision; placing 2 correct loops out "
                "of 10 gives a low recall. "
                "Intuition: F1 is the score that punishes both over- "
                "and under-placement. It is why scattering loops "
                "randomly across the heatmap does not work — you need "
                "to read the green dots carefully and place loops "
                "precisely."},

            {"type": "subheading", "text": "The total score"},
            {"type": "body", "text":
                "All five scores are combined into a single weighted "
                "sum, normalised to a 0-100 scale. The weights are "
                "chosen so that compartments and loops contribute "
                "roughly equally — neither can be ignored:"},
            {"type": "equation",
             "latex": r"\mathrm{Score} = w_1\cdot\mathrm{SCC} + w_2\cdot r_{E1} + w_3\cdot r_{\mathrm{cb}} + w_4\cdot\mathrm{APA} + w_5\cdot F_1",
             "label": "weighted total (0-100)"},
            {"type": "body", "text":
                "In two-player mode the scores split: the loop player "
                "is scored on APA and F1 only; the compartment player "
                "on r_E1 and r_cb only. This makes the competition "
                "genuinely orthogonal — a perfect loop configuration "
                "with random colours scores high for loops and near "
                "zero for compartments, and vice versa. The two "
                "mechanisms are independent in the biology and "
                "independent in the scoring."},
        ]
    },
    {
        "title": "Structural",
        "icon": "S",
        "blocks": [
            {"type": "heading", "text": "Live Structural Metrics"},
            {"type": "body", "text":
                "The analysis panel tracks six quantities in real time "
                "as the simulation runs. They are computed directly "
                "from the bead positions — no contact map needed — "
                "and update every few frames. Together they give you "
                "an instant physical picture of what the polymer is "
                "doing, independent of how well it matches the target. "
                "Think of them as the polymer's vital signs."},

            {"type": "subheading", "text": "Radius of gyration  Rg"},
            {"type": "equation",
             "latex": r"R_g = \sqrt{\frac{1}{N}\sum_i |\mathbf{r}_i - \bar{\mathbf{r}}|^2}",
             "label": "compactness"},
            {"type": "body", "text":
                "Rg is the RMS distance of all beads from the chain's "
                "centre of mass. It is the single most informative "
                "structural number: small Rg means the chain is "
                "compact and globular (attractions dominate), large "
                "Rg means it is open and extended (excluded volume "
                "or backbone stiffness dominates). "
                "At thermal equilibrium, polymer theory predicts "
                "Rg ~ N^ν, where ν ≈ 1/3 for a collapsed globule "
                "and ν ≈ 3/5 for a self-avoiding swollen coil. "
                "Watch Rg fall when you paint more B beads — the "
                "strong B-B attraction collapses the chain — and "
                "rise when you increase the excluded volume diameter."},

            {"type": "subheading", "text": "End-to-end distance  R_ee"},
            {"type": "equation",
             "latex": r"R_{ee} = |\mathbf{r}_{N-1} - \mathbf{r}_0|",
             "label": "chain extent"},
            {"type": "body", "text":
                "The straight-line distance between the first and "
                "last bead. For a freely jointed ideal chain, "
                "R_ee² = N · b₀² on average. In the presence of "
                "loops and attractions, R_ee can be far smaller — "
                "a loop between bead 0 and bead N-1 would collapse "
                "it almost to zero. Unlike Rg, R_ee fluctuates "
                "strongly from frame to frame as the chain ends "
                "wander, so the time-series plot is noisier than "
                "the others. Watch for its mean value and trend "
                "rather than individual spikes."},

            {"type": "subheading", "text": "Mean bond length  ⟨b⟩"},
            {"type": "equation",
             "latex": r"\langle b \rangle = \frac{1}{N-1}\sum_i |\mathbf{r}_{i+1}-\mathbf{r}_i|",
             "label": "integrator health"},
            {"type": "body", "text":
                "The average length of all backbone bonds. This should "
                "stay very close to the equilibrium value b₀ = 1 at "
                "all times. It is the first thing to check when the "
                "simulation looks wrong: if ⟨b⟩ drifts upward beyond "
                "about 1.2, the integrator is becoming unstable — "
                "forces are so large that each step overshoots. "
                "The fix is always to lower dt, raise γ, or lower "
                "k_bond until dt · k_bond / γ < 0.5. "
                "Intuition: a stable simulation has bonds that "
                "oscillate tightly around b₀. An unstable one has "
                "bonds that stretch further and further until the "
                "positions become NaN and the simulation dies."},

            {"type": "subheading", "text": "Contact count  Nc"},
            {"type": "equation",
             "latex": r"N_c = |\{(i,j):\,|i-j| \geq 2,\;P_{ij}>0.5\}|",
             "label": "crowding"},
            {"type": "body", "text":
                "The number of non-bonded bead pairs whose sigmoid "
                "contact probability exceeds 0.5 — a clean binary "
                "measure of how crowded the chain is. An extended "
                "chain has O(N) contacts (mostly between neighbours); "
                "a collapsed globule has O(N²) because every bead "
                "is near every other. "
                "Nc rises quickly when you add B beads or increase "
                "the copolymer attraction, and falls when you raise "
                "the excluded volume height. It is the most direct "
                "measure of how 3D-compact the chain is at any "
                "given moment — more direct than Rg for detecting "
                "sudden collapse events."},

            {"type": "subheading", "text": "Asphericity  δ"},
            {"type": "equation",
             "latex": r"\delta = \frac{(\lambda_1-\lambda_2)^2+(\lambda_2-\lambda_3)^2+(\lambda_1-\lambda_3)^2}{2(\lambda_1+\lambda_2+\lambda_3)^2}",
             "label": "shape  [0 = sphere,  1 = rod]"},
            {"type": "body", "text":
                "Asphericity measures how far the chain's shape "
                "departs from a perfect sphere, using the three "
                "eigenvalues λ₁ ≤ λ₂ ≤ λ₃ of the 3×3 gyration "
                "tensor — the polymer equivalent of the inertia "
                "tensor. If all three eigenvalues are equal the "
                "cloud of beads is spherically symmetric and δ = 0. "
                "If all mass is along one axis (a rod), δ = 1. "
                "A typical looped and compartmentalised chromatin "
                "model sits around δ = 0.3-0.5 — not quite a sphere, "
                "not quite a rod, more like a flattened ellipsoid. "
                "High δ values appear when long loops stretch the "
                "chain along one axis, or when stiff backbone "
                "dominates and the chain has not yet equilibrated."},

            {"type": "subheading", "text": "Compaction index  κ"},
            {"type": "equation",
             "latex": r"\kappa = \frac{R_g}{R_g^{\mathrm{ideal}}}, \qquad R_g^{\mathrm{ideal}} = b_0\sqrt{N/6}",
             "label": "vs ideal Gaussian chain"},
            {"type": "body", "text":
                "Compaction index κ puts Rg in context by comparing "
                "it to what a Gaussian (ideal, non-interacting) chain "
                "of the same length would give. κ = 1 means the "
                "chain is exactly as extended as an ideal random "
                "walk. κ < 1 means it is more compact — attractive "
                "interactions are winning. κ > 1 means it is more "
                "swollen — excluded volume or backbone stiffness are "
                "winning. For biological chromatin at the scale of "
                "this game, κ typically sits around 0.3-0.7. "
                "Watching κ over time tells you immediately whether "
                "a parameter change is collapsing the chain (κ falls) "
                "or expanding it (κ rises), without needing to know "
                "the absolute scale of the system."},
        ]
    },
]