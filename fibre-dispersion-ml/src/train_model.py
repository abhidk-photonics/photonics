import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from dataset import split_by_design


FEATURES = ["core_radius_um", "delta_n", "wavelength_nm"]
TARGET = "dispersion_ps_nm_km"


def train(data_path, model_path, seed=42):
    data = pd.read_csv(data_path)
    train_df, val_df, test_df = split_by_design(data, seed)

    # Standardisation improves conditioning; scalers are fitted on training designs only.
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(train_df[FEATURES])
    y_train = y_scaler.fit_transform(train_df[[TARGET]]).ravel()

    # Early stopping uses a random row-wise subset of training data, not the held-out validation designs.
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )
    mlp.fit(X_train, y_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": mlp,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
        },
        model_path,
    )

    print(
        f"Train: {len(train_df)} | Validation: {len(val_df)} | "
        f"Held-out test: {len(test_df)}"
    )
    print(f"Iterations: {mlp.n_iter_}; saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fibre_dispersion_dataset.csv")
    parser.add_argument("--model", default="models/fibre_dispersion_mlp.joblib")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args.data, args.model, args.seed)
