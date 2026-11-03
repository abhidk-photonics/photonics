import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from dataset import split_by_design
from train_model import FEATURES, TARGET


def plot_design_curves(test, path, n_designs=4):
    """Compare physics-model curves and MLP predictions for selected test designs."""
    designs = (
        test.drop_duplicates("design_id")
        .sort_values("core_radius_um")
        .iloc[
            np.linspace(
                0,
                test.design_id.nunique() - 1,
                n_designs,
            ).astype(int)
        ]
    )

    plt.figure(figsize=(8, 5))

    for i, (_, design) in enumerate(designs.iterrows()):
        group = test[test.design_id == design.design_id].sort_values(
            "wavelength_nm"
        )
        color = f"C{i}"

        plt.plot(
            group.wavelength_nm,
            group[TARGET],
            color=color,
            label=(
                f"a = {design.core_radius_um:.2f} µm, "
                f"Δn = {design.delta_n:.4f}"
            ),
        )
        plt.plot(
            group.wavelength_nm,
            group.predicted,
            "o",
            color=color,
            markersize=4,
            mfc="none",
        )

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
    """Plot signed prediction errors and wavelength-resolved mean absolute error."""
    plt.figure(figsize=(8, 5))
    plt.scatter(
        test.wavelength_nm,
        test.error,
        s=8,
        alpha=0.35,
        label="Individual samples",
    )

    mae = test.groupby("wavelength_nm").error.apply(
        lambda errors: errors.abs().mean()
    )
    plt.plot(
        mae.index,
        mae.values,
        "k-o",
        markersize=3,
        label="MAE at each wavelength",
    )

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

    # Use the training seed to reproduce the same held-out test designs.
    _, _, test = split_by_design(data, seed)

    bundle = joblib.load(model_path)
    X = bundle["x_scaler"].transform(test[FEATURES])
    pred_scaled = bundle["model"].predict(X).reshape(-1, 1)
    pred = bundle["y_scaler"].inverse_transform(pred_scaled).ravel()
    actual = test[TARGET].to_numpy()

    r2 = r2_score(actual, pred)
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "reference_dispersion": actual,
            "predicted_dispersion": pred,
        }
    ).to_csv(output_path / "test_set_predictions.csv", index=False)

    lower = min(actual.min(), pred.min())
    upper = max(actual.max(), pred.max())

    plt.figure(figsize=(7, 6))
    plt.scatter(actual, pred, s=18, alpha=0.65)
    plt.plot([lower, upper], [lower, upper], "--")
    plt.xlabel("Physics-model dispersion [ps/(nm·km)]")
    plt.ylabel("MLP-predicted dispersion [ps/(nm·km)]")
    plt.title("Predicted vs. reference chromatic dispersion")
    plt.tight_layout()
    plt.savefig(
        output_path / "predicted_vs_reference_dispersion.png",
        dpi=300,
    )
    plt.close()

    test = test.assign(predicted=pred, error=pred - actual)

    plot_design_curves(
        test,
        output_path / "test_fibre_design_dispersion_comparison.png",
    )
    plot_error_histogram(
        test,
        output_path / "dispersion_prediction_error_distribution.png",
    )
    plot_error_vs_wavelength(
        test,
        output_path / "dispersion_prediction_error_vs_wavelength.png",
    )

    print(
        f"Test designs: {test.design_id.nunique()} | samples: {len(test)}"
    )
    print(
        f"R² = {r2:.6f}; MAE = {mae:.6f} ps/(nm·km); "
        f"RMSE = {rmse:.6f} ps/(nm·km)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fibre_dispersion_dataset.csv")
    parser.add_argument("--model", default="models/fibre_dispersion_mlp.joblib")
    parser.add_argument("--output", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(args.data, args.model, args.output, args.seed)
