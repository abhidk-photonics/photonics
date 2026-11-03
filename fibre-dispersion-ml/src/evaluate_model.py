import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dataset import split_by_design
from train_model import FEATURES, TARGET

def plot_design_curves(test, path, n_designs=4):
    """Physics curve (line) and MLP prediction (markers) for a few test designs."""
    # Pick designs evenly across the core-radius range so the curves are not
    # all from one corner of the design space.
    designs = (test.drop_duplicates("design_id").sort_values("core_radius_um")
               .iloc[np.linspace(0, test.design_id.nunique() - 1, n_designs).astype(int)])
    plt.figure(figsize=(8, 5))
    for i, (_, d) in enumerate(designs.iterrows()):
        g = test[test.design_id == d.design_id].sort_values("wavelength_nm")
        color = f"C{i}"
        plt.plot(g.wavelength_nm, g[TARGET], color=color,
                 label=f"a = {d.core_radius_um:.2f} µm, Δn = {d.delta_n:.4f}")
        plt.plot(g.wavelength_nm, g.predicted, "o", color=color, markersize=4, mfc="none")
    plt.axhline(0, linestyle="--", linewidth=1, color="gray")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Chromatic dispersion [ps/(nm·km)]")
    plt.title("Test designs: physics model (line) vs. MLP (markers)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_error_histogram(test, path):
    plt.figure(figsize=(7, 5))
    plt.hist(test.error, bins=50)
    plt.axvline(0, linestyle="--", linewidth=1, color="gray")
    plt.xlabel("Prediction error (MLP − physics) [ps/(nm·km)]")
    plt.ylabel("Number of test samples")
    plt.title("Test-set prediction error")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_error_vs_wavelength(test, path):
    """Signed errors per sample, with the mean absolute error at each wavelength."""
    plt.figure(figsize=(8, 5))
    plt.scatter(test.wavelength_nm, test.error, s=8, alpha=0.35, label="Individual samples")
    mae = test.groupby("wavelength_nm").error.apply(lambda e: e.abs().mean())
    plt.plot(mae.index, mae.values, "k-o", markersize=3, label="MAE at each wavelength")
    plt.axhline(0, linestyle="--", linewidth=1, color="gray")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Prediction error [ps/(nm·km)]")
    plt.title("Prediction error vs. wavelength (test set)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def evaluate(data_path, model_path, output_dir, seed=42):
    data = pd.read_csv(data_path)
    # The seed must match the one used in training, otherwise the test designs
    # would overlap with the training designs.
    _, _, test = split_by_design(data, seed)
    bundle = joblib.load(model_path)
    X = bundle["x_scaler"].transform(test[FEATURES])
    pred_scaled = bundle["model"].predict(X).reshape(-1, 1)
    pred = bundle["y_scaler"].inverse_transform(pred_scaled).ravel()
    actual = test[TARGET].to_numpy()
    r2 = r2_score(actual, pred)
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"reference_dispersion": actual, "predicted_dispersion": pred}).to_csv(
        out/"test_set_predictions.csv", index=False
    )
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, pred, s=18, alpha=0.65)
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("Physics-model dispersion [ps/(nm·km)]")
    plt.ylabel("MLP-predicted dispersion [ps/(nm·km)]")
    plt.title("Predicted vs. reference chromatic dispersion")
    plt.tight_layout()
    plt.savefig(out/"predicted_vs_reference_dispersion.png", dpi=300)
    plt.close()

    test = test.assign(predicted=pred, error=pred - actual)
    plot_design_curves(test, out/"test_fibre_design_dispersion_comparison.png")
    plot_error_histogram(test, out/"dispersion_prediction_error_distribution.png")
    plot_error_vs_wavelength(test, out/"dispersion_prediction_error_vs_wavelength.png")

    print(f"Test designs: {test.design_id.nunique()} | samples: {len(test)}")
    print(f"R² = {r2:.6f}; MAE = {mae:.6f} ps/(nm·km); RMSE = {rmse:.6f} ps/(nm·km)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/fibre_dispersion_dataset.csv")
    p.add_argument("--model", default="models/fibre_dispersion_mlp.joblib")
    p.add_argument("--output", default="results")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    evaluate(a.data, a.model, a.output, a.seed)
