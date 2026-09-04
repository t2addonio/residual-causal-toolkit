"""Residual Causal Toolkit primitives for GPT-2 residual streams."""
from .pure_quad import pure_quadratic_14d, second_moment_features
from .causal_filters import residual_power, apply_causal_filters, extract_and_filter_ports
from .hooks import ResidualStreamHook, get_residual_at_layer
from . import interferometer
