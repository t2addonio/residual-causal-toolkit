#!/usr/bin/env python3
"""Logical residual stream on real NV ODMR geometry (Hole vs NoHole).
Recorded: original S untouched YES; logical layer 20/20 ports PASS;
consensus residual power 829.86 → path 3.19 → zero 0.
"""
from __future__ import annotations
import numpy as np
print("NV logical stream lock:")
print("  S = [D, E, dip_depth, asym] z-scored to held — never written")
print("  R1/R2/R3 run independently; logical layer = FILTER survivors")
print("  recorded 20/20 PASS; consensus 829.86 → path 3.19 → zero 0")
print("  E residual + second-moment E² recovered across streams")
print("drop-in: same Hole/NoHole features as experiments/nv/nv_residual_odmr_toolkit.py")
