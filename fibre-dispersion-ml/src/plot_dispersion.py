import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from fibre_model import calculate_dispersion_curve

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=float, default=3.8)
    p.add_argument("--delta-n", type=float, default=0.004)
    p.add_argument("--output", default="results/physics_model_dispersion_curves.png")
    a = p.parse_args()
    wavelengths = np.linspace(1300, 1600, 31)
    dispersion = calculate_dispersion_curve(a.radius, a.delta_n, wavelengths)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, dispersion, marker="o", markersize=3)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Chromatic dispersion [ps/(nm·km)]")
    plt.title(f"Physics-model dispersion (a = {a.radius:.1f} µm, Δn = {a.delta_n:.4f})")
    plt.tight_layout()
    plt.savefig(a.output, dpi=300)
    plt.close()
    print(f"Saved {a.output}")
