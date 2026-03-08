#!/usr/bin/env python3
"""
Generate synthetic engineering stress-strain curves for steels with different
impurity concentrations.

This is a toy model for onboarding, demonstrations, and ML dataset generation.
It is not a calibrated constitutive law and should not be used for design.

Outputs:
- synthetic_steel_stress_strain_curves.csv
- synthetic_steel_curve_summary.csv

Usage:
    python generate_synthetic_steel_curves.py

Optional:
    python generate_synthetic_steel_curves.py --plot
    python generate_synthetic_steel_curves.py --n-random 200 --seed 42
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SteelComposition:
    label: str
    C: float   # carbon, wt%
    Mn: float  # manganese, wt%
    P: float   # phosphorus, wt%
    S: float   # sulfur, wt%


def synthetic_steel_curve(
    strain: np.ndarray,
    C: float,
    Mn: float,
    P: float,
    S: float,
    E: float = 210_000.0,
) -> dict:
    """
    Build a synthetic engineering stress-strain curve.

    Parameters
    ----------
    strain : np.ndarray
        Engineering strain values.
    C, Mn, P, S : float
        Impurity concentrations in wt%.
    E : float
        Young's modulus in MPa.

    Returns
    -------
    dict
        Contains stress array and summary properties.
    """
    # Heuristic property model:
    # - C strongly raises strength and slightly reduces ductility
    # - Mn raises strength and can preserve ductility better
    # - P and S embrittle and reduce fracture strain
    sigma_y = 220.0 + 900.0 * np.sqrt(max(C, 1e-9)) + 35.0 * Mn + 1200.0 * P + 700.0 * S
    sigma_uts = sigma_y + 180.0 + 250.0 * C + 45.0 * Mn - 80.0 * (P + S)

    uniform_strain = 0.11 + 0.010 * Mn - 0.020 * np.sqrt(max(C, 1e-9)) - 0.90 * P - 0.60 * S
    fracture_strain = 0.24 + 0.020 * Mn - 0.040 * np.sqrt(max(C, 1e-9)) - 2.2 * P - 1.6 * S

    uniform_strain = float(np.clip(uniform_strain, 0.03, 0.18))
    fracture_strain = float(np.clip(fracture_strain, uniform_strain + 0.01, 0.35))
    sigma_uts = float(max(sigma_uts, sigma_y + 40.0))

    eps_y = sigma_y / E
    eps_luders = min(eps_y + 0.008 + 0.030 * C, uniform_strain - 0.01)

    # Small upper-yield effect, enhanced by C
    sigma_upper = sigma_y * (1.0 + 0.03 + 0.15 * C)
    sigma_lower = sigma_y

    stress = np.full_like(strain, np.nan, dtype=float)

    for i, eps in enumerate(strain):
        if eps <= eps_y:
            # Elastic
            stress[i] = E * eps
        elif eps <= eps_luders:
            # Lüders-like plateau
            t = (eps - eps_y) / max(eps_luders - eps_y, 1e-12)
            stress[i] = sigma_upper - (sigma_upper - sigma_lower) * (1.0 - np.exp(-8.0 * t))
        elif eps <= uniform_strain:
            # Strain hardening
            t = (eps - eps_luders) / max(uniform_strain - eps_luders, 1e-12)
            stress[i] = sigma_lower + (sigma_uts - sigma_lower) * (t ** 0.55)
        elif eps <= fracture_strain:
            # Necking in engineering stress
            t = (eps - uniform_strain) / max(fracture_strain - uniform_strain, 1e-12)
            stress[i] = sigma_uts * (1.0 - 0.35 * (t ** 1.25))
        else:
            stress[i] = np.nan

    return {
        "stress": stress,
        "yield_strength_MPa": sigma_y,
        "ultimate_tensile_strength_MPa": sigma_uts,
        "uniform_strain": uniform_strain,
        "fracture_strain": fracture_strain,
        "youngs_modulus_MPa": E,
    }


def make_default_cases() -> list[SteelComposition]:
    return [
        SteelComposition("Base steel", C=0.05, Mn=0.50, P=0.010, S=0.010),
        SteelComposition("Higher carbon", C=0.25, Mn=0.50, P=0.010, S=0.010),
        SteelComposition("Low impurity", C=0.02, Mn=0.30, P=0.003, S=0.003),
        SteelComposition("Phosphorus-rich", C=0.05, Mn=0.50, P=0.040, S=0.010),
        SteelComposition("Mn-strengthened", C=0.05, Mn=1.50, P=0.010, S=0.010),
    ]


def make_random_cases(n_random: int, seed: int) -> list[SteelComposition]:
    rng = np.random.default_rng(seed)
    cases: list[SteelComposition] = []

    for i in range(n_random):
        cases.append(
            SteelComposition(
                label=f"Random_{i:04d}",
                C=float(rng.uniform(0.005, 0.30)),
                Mn=float(rng.uniform(0.10, 2.00)),
                P=float(rng.uniform(0.001, 0.050)),
                S=float(rng.uniform(0.001, 0.040)),
            )
        )
    return cases


def make_random_compositions_array(n_random: int, seed: int) -> np.ndarray:
    """
    Generate N random steel compositions.

    Returns
    -------
    np.ndarray
        Array of shape (n_random, 4) with [C, Mn, P, S] values in wt%.
    """
    rng = np.random.default_rng(seed)
    # Generate C, Mn, P, S
    c = rng.uniform(0.005, 0.30, size=n_random)
    mn = rng.uniform(0.10, 2.00, size=n_random)
    p = rng.uniform(0.001, 0.050, size=n_random)
    s = rng.uniform(0.001, 0.040, size=n_random)

    return np.stack([c, mn, p, s], axis=1)


def curves_to_dataframes(
    cases: list[SteelComposition],
    strain: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    summary_rows = []
    # Collect all stress columns in a list to stack them later
    stress_arrays = [strain]

    for case in cases:
        result = synthetic_steel_curve(
            strain=strain,
            C=case.C,
            Mn=case.Mn,
            P=case.P,
            S=case.S,
        )

        stress_arrays.append(result["stress"])

        summary_rows.append(
            {
                "label": case.label,
                "C_wt_percent": case.C,
                "Mn_wt_percent": case.Mn,
                "P_wt_percent": case.P,
                "S_wt_percent": case.S,
                "yield_strength_MPa": result["yield_strength_MPa"],
                "ultimate_tensile_strength_MPa": result["ultimate_tensile_strength_MPa"],
                "uniform_strain": result["uniform_strain"],
                "fracture_strain": result["fracture_strain"],
                "youngs_modulus_MPa": result["youngs_modulus_MPa"],
            }
        )

    # curves_array: (n_cases + 1, n_points)
    # The first row is the strain array.
    curves_array = np.stack(stress_arrays, axis=0)
    summary_df = pd.DataFrame(summary_rows)
    return curves_array, summary_df


def save_outputs(
    curves_array: np.ndarray,
    summary_df: pd.DataFrame,
    output_dir: Path,
    compositions_array: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    curves_path = output_dir / "synthetic_steel_stress_strain_curves.npy"
    summary_path = output_dir / "synthetic_steel_curve_summary.csv"

    np.save(curves_path, curves_array)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved curves to:  {curves_path}")
    print(f"Saved summary to: {summary_path}")

    if compositions_array is not None:
        comp_path = output_dir / "random_steel_compositions.npy"
        np.save(comp_path, compositions_array)
        print(f"Saved compositions array to: {comp_path}")


def plot_curves(curves_array: np.ndarray, labels: list[str]) -> None:
    import matplotlib.pyplot as plt

    strain = curves_array[0, :]

    plt.figure(figsize=(8, 5))
    for i in range(1, curves_array.shape[0]):
        plt.plot(strain, curves_array[i, :], label=labels[i-1].replace("_", " "))

    plt.xlabel("Engineering strain")
    plt.ylabel("Engineering stress (MPa)")
    plt.title("Synthetic stress-strain curves for steels")
    plt.xlim(0.0, float(np.nanmax(strain)))
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic steel stress-strain curves.")
    parser.add_argument(
        "--n-points",
        type=int,
        default=601,
        help="Number of strain points between 0 and max strain.",
    )
    parser.add_argument(
        "--max-strain",
        type=float,
        default=0.30,
        help="Maximum engineering strain.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display a plot of the generated curves.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=0,
        help="Generate N random compositions instead of only the default example cases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --n-random.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strain = np.linspace(0.0, args.max_strain, args.n_points)

    cases = make_default_cases()
    compositions_array = None
    if args.n_random > 0:
        cases.extend(make_random_cases(args.n_random, args.seed))
        compositions_array = make_random_compositions_array(args.n_random, args.seed)

    curves_array, summary_df = curves_to_dataframes(cases, strain)
    save_outputs(curves_array, summary_df, args.output_dir, compositions_array)

    if args.plot:
        labels = [c.label for c in cases]
        plot_curves(curves_array, labels)


if __name__ == "__main__":
    main()