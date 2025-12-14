#!/usr/bin/env python3
"""
Information-Time Structure Theory
Quick Start: Temporal Interference Demo (Conceptual Only)

What this does
--------------
This script demonstrates a *detector sensitivity test*:

1) Generate a null time series (past-origin baseline): AR(1) + white noise
2) Generate a pseudo future-constrained variant by imposing a terminal boundary condition
3) Compute simple indices that react to "time-directional distortion":
   - Delta_TR : time-reversal asymmetry (MSE vs reversed series)
   - f_tau    : scale-dependent variance anomaly (variance across window variances)
   - I_TI     : composite score (weighted sum)

Important
---------
- This does NOT prove the future, retrocausality, or any physical time reversal.
- This is NOT a real "future signal receiver".
- It is a clean, runnable demo that shows the detector reacts to a boundary-condition-like structure.

Install
-------
Requires only NumPy.
Optional: Matplotlib for plotting.

    pip install numpy
    pip install matplotlib   # optional

Run
---
    python conceptual_demo.py
    python conceptual_demo.py --plot
    python conceptual_demo.py --seed 0 --C 15 --decay 80 --plot

Outputs
-------
- Higher I_TI means stronger time-directional distortion in this synthetic setup.
- Expect I_TI(pseudo_future) > I_TI(null) often, but not always (randomness).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# -----------------------------
# Core generators
# -----------------------------

def generate_null_ar1(T: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a null (past-origin baseline) AR(1) series:
        x[t] = phi * x[t-1] + noise[t]
    """
    noise = rng.normal(0.0, 1.0, T)
    x = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def apply_pseudo_future_constraint(
    x_base: np.ndarray,
    C: float,
    decay: float
) -> np.ndarray:
    """
    Create a pseudo future-constrained series by adding a terminal boundary condition.

    Mechanism:
    - Compute the terminal discrepancy: (C - x[T-1])
    - Propagate a decaying influence backward in time:
        x[T-k] += exp(-k/decay) * (C - x[T-1])

    Notes:
    - This is a mathematical boundary condition, not a physical future signal.
    - Smaller decay => stronger / more localized interference.
    """
    T = len(x_base)
    x = x_base.copy()
    terminal_delta = C - x[T - 1]

    # Backward influence
    for k in range(1, T):
        alpha = np.exp(-k / float(decay))
        x[T - k] += alpha * terminal_delta

    return x


# -----------------------------
# Detection measures
# -----------------------------

def time_reversal_asymmetry_mse(x: np.ndarray) -> float:
    """
    Time-reversal asymmetry via mean squared error between x and reversed(x).

    A perfectly time-reversal-symmetric *sequence* would yield small values here.
    """
    xr = x[::-1]
    return float(np.mean((x - xr) ** 2))


def window_variances(x: np.ndarray, tau: int) -> np.ndarray:
    """
    Split x into non-overlapping windows of size tau and return per-window variances.
    """
    if tau <= 1 or tau >= len(x):
        return np.array([], dtype=np.float64)

    n = (len(x) // tau) * tau
    x2 = x[:n]
    windows = x2.reshape(-1, tau)
    return np.var(windows, axis=1)


def scale_anomaly_f_tau(x: np.ndarray, scales: List[int]) -> Tuple[float, Dict[int, float]]:
    """
    Scale anomaly score: compute mean variance within each scale's windowing,
    then take variance across scales.

    Returns:
      f_tau : float
      per_scale_mean_var : {tau: mean(window_variance)}
    """
    per_scale: Dict[int, float] = {}
    vals: List[float] = []

    for tau in scales:
        wv = window_variances(x, tau)
        if wv.size == 0:
            continue
        m = float(np.mean(wv))
        per_scale[int(tau)] = m
        vals.append(m)

    if len(vals) < 2:
        return 0.0, per_scale

    f_tau = float(np.var(np.array(vals, dtype=np.float64)))
    return f_tau, per_scale


@dataclass
class Indices:
    I_TI: float
    Delta_TR: float
    f_tau: float


def compute_indices(
    x: np.ndarray,
    scales: List[int],
    w1: float = 0.4,
    w2: float = 0.6
) -> Tuple[Indices, Dict[int, float]]:
    """
    Compute composite temporal interference indices.

    I_TI = w1 * Delta_TR + w2 * f_tau
    """
    Delta_TR = time_reversal_asymmetry_mse(x)
    f_tau, per_scale = scale_anomaly_f_tau(x, scales)
    I_TI = float(w1 * Delta_TR + w2 * f_tau)
    return Indices(I_TI=I_TI, Delta_TR=Delta_TR, f_tau=f_tau), per_scale


# -----------------------------
# CLI + presentation
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick Start conceptual demo: null vs pseudo future-constrained time series."
    )
    p.add_argument("--T", type=int, default=5000, help="Length of time series")
    p.add_argument("--phi", type=float, default=0.8, help="AR(1) coefficient for null data")
    p.add_argument("--C", type=float, default=5.0, help="Terminal constraint value for pseudo-future")
    p.add_argument("--decay", type=float, default=200.0, help="Decay constant for backward influence")
    p.add_argument("--scales", type=str, default="10,50,100,200", help="Comma-separated window sizes")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--plot", action="store_true", help="Plot series and basic diagnostics (requires matplotlib)")
    return p.parse_args()


def fmt_scales(s: str) -> List[int]:
    out: List[int] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def print_summary(
    title: str,
    idx: Indices,
    per_scale: Dict[int, float]
) -> None:
    print(f"--- {title} ---")
    print(f"I_TI     = {idx.I_TI:.6g}")
    print(f"Delta_TR = {idx.Delta_TR:.6g}")
    print(f"f_tau    = {idx.f_tau:.6g}")
    if per_scale:
        items = ", ".join([f"{k}:{v:.6g}" for k, v in sorted(per_scale.items())])
        print(f"scale_mean_var = {{{items}}}")
    print()


def maybe_plot(x_past: np.ndarray, x_future: np.ndarray, scales: List[int]) -> None:
    try:
        import matplotlib.pyplot as plt  # optional
    except Exception:
        print("Plotting requested but matplotlib is not installed.")
        print("Install with: pip install matplotlib")
        return

    # Plot the two series (first N points for readability)
    N = min(1200, len(x_past))

    plt.figure()
    plt.title("Null (past-origin) vs Pseudo future-constrained (first segment)")
    plt.plot(x_past[:N], label="null")
    plt.plot(x_future[:N], label="pseudo_future")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.show()

    # Plot per-scale mean variances for both
    _, ps_past = compute_indices(x_past, scales)
    _, ps_future = compute_indices(x_future, scales)

    taus = sorted(set(ps_past.keys()) | set(ps_future.keys()))
    if taus:
        past_vals = [ps_past.get(t, np.nan) for t in taus]
        future_vals = [ps_future.get(t, np.nan) for t in taus]

        plt.figure()
        plt.title("Scale mean variance (per window size)")
        plt.plot(taus, past_vals, marker="o", label="null")
        plt.plot(taus, future_vals, marker="o", label="pseudo_future")
        plt.legend()
        plt.xlabel("window size (tau)")
        plt.ylabel("mean(window variance)")
        plt.show()


def main() -> None:
    args = parse_args()
    scales = fmt_scales(args.scales)

    rng = np.random.default_rng(args.seed)

    # 1) Null data
    x_past = generate_null_ar1(T=args.T, phi=args.phi, rng=rng)

    # 2) Pseudo future-constrained data (derived from the same baseline)
    x_future = apply_pseudo_future_constraint(x_base=x_past, C=args.C, decay=args.decay)

    # 3) Compute indices
    idx_past, ps_past = compute_indices(x_past, scales)
    idx_future, ps_future = compute_indices(x_future, scales)

    print("=" * 72)
    print("Temporal Interference Demo (Conceptual Only)")
    print("=" * 72)
    print(f"T={args.T}, phi={args.phi}, C={args.C}, decay={args.decay}, scales={scales}, seed={args.seed}")
    print()

    print_summary("NULL (past-origin baseline)", idx_past, ps_past)
    print_summary("PSEUDO FUTURE-CONSTRAINED", idx_future, ps_future)

    print("Δ(I_TI) = I_TI(pseudo_future) - I_TI(null) =", f"{(idx_future.I_TI - idx_past.I_TI):.6g}")
    print()

    if args.plot:
        maybe_plot(x_past, x_future, scales)


if __name__ == "__main__":
    main()
