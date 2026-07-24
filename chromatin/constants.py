# chromatin/constants.py
"""Shared constants imported by app.py, draw.py and interact.py.

Kept in a separate module so none of those three files need to import
each other, which would create circular dependencies.
"""
from __future__ import annotations
from . import theme

# Game states
MENU, SETTINGS, LOADING, PLAY, SETTLE, RESULTS, LAB = (
    "menu", "settings", "loading", "play", "settle", "results", "lab")

# Player indices (versus mode)
P_LOOP, P_COMP = 0, 1
PLAYER_NAME   = ["Loop player", "Compartment player"]
PLAYER_SHORT  = ["LOOPS", "COMPARTMENTS"]
PLAYER_ACCENT = [theme.GREEN, theme.MAGENTA]   # needs theme, so can't be in a plain data file

# Measurement ensemble
MEAS_BURN, MEAS_SAMPLES, MEAS_EVERY = 2500, 520, 20
LIVE_ALPHA = 0.006

# Available resolutions in Settings
RESOLUTIONS = [(1280, 800), (1600, 900), (1920, 1080)]

# Lab parameter groups (SimParams field, label, lo, hi, step, fmt, is_int)
LAB_PARAM_GROUPS = [
    ("Langevin integrator", [
        ("dt",             "timestep  dt",      0.001, 0.02,  0.001, "{:.3f}", False),
        ("gamma",          "friction  γ",        0.1,  10.0,  0.1,  "{:.2f}", False),
        ("kT",             "temperature  kT",    0.1,   3.0,  0.1,  "{:.2f}", False),
        ("steps_per_frame","steps / frame",      5,   200,    5,    "{:d}",   True),
    ]),
    ("Backbone & bending", [
        ("b0",      "bond length  b0",    0.5,   2.0,  0.05, "{:.2f}", False),
        ("k_bond",  "bond stiffness",     20,  500,    5,    "{:.0f}", False),
        ("k_angle", "bending  k_angle",   0.0, 100.0,  5,   "{:.1f}", False),
    ]),
    ("Loops & the player's hand", [
        ("k_loop",  "loop stiffness",     5,   150,    5,    "{:.0f}", False),
        ("k_grab",  "hand stiffness",     10,  200,    5,    "{:.0f}", False),
    ]),
    ("Excluded volume", [
        ("ev_eps",  "EV height  ε",       5,   500,    5,    "{:.0f}", False),
        ("ev_rc",   "EV diameter  rc",    0.5,   4.0,  0.05, "{:.2f}", False),
    ]),
    ("Compartments (copolymer)", [
        ("sigma",      "attraction range σ", 0.5,  2.5, 0.1,  "{:.2f}", False),
        ("eps_AA",     "A–A  ε",             0.0,  4.0, 0.25, "{:.2f}", False),
        ("eps_BB",     "B–B  ε",             0.0,  4.0, 0.25, "{:.2f}", False),
        ("eps_AB",     "A–B  ε",            -1.0,  1.0, 0.25, "{:.2f}", False),
        ("eps_domain", "loop-domain bonus",  0.0,  1.0, 0.25, "{:.2f}", False),
    ]),
    ("Confinement & contact call", [
        ("k_wall",    "wall stiffness",    5,   200,   5,    "{:.0f}", False),
        ("contact_rc","contact cutoff",    0.8,   3.0, 0.1,  "{:.2f}", False),
        ("contact_w", "contact softness",  0.02,  0.5, 0.05, "{:.2f}", False),
    ]),
]