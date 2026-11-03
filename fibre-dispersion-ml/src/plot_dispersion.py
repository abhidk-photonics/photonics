import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fibre_model import calculate_dispersion_curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=3.8)
    parser.add_argument("--delta-n", type=float, default=0.004)
    parser.add_argument(
        "--output",
        default="results/physics_model_dispersion_curves.png",
    )
    args = parser.parse_args()

    wavelengths = np.linspace(1300, 1600, 31)
    dispersion = calculate_dispersion_curve(
        args.radius,
        args.delta_n,
        wavelengths,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, dispersion, marker="o", markersize=3)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Chromatic dispersion [ps/(nm·km)]")
    plt.title(
        f"Physics-model dispersion (a = {args.radius:.1f} µm, "
        f"Δn = {args.delta_n:.4f})"
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"Saved {args.output}")
