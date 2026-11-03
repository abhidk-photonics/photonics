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
    # Features differ by orders of magnitude (delta_n ~ 1e-3, wavelength ~ 1e3),
    # so both inputs and target are standardised. Scalers are fitted on the
    # training designs only and saved with the model.
    xs, ys = StandardScaler(), StandardScaler()
    X_train = xs.fit_transform(train_df[FEATURES])
    y_train = ys.fit_transform(train_df[[TARGET]]).ravel()

    # Settings retained from the earlier project notes.
    # Note: early_stopping holds out a random 15% of training rows, not whole
    # designs, and the validation designs from the split are not used here.
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64, 32), activation="relu", solver="adam",
        learning_rate_init=0.001, max_iter=500, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=20, random_state=seed
    )
    mlp.fit(X_train, y_train)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": mlp, "x_scaler": xs, "y_scaler": ys}, model_path)
    print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Held-out test: {len(test_df)}")
    print(f"Iterations: {mlp.n_iter_}; saved model to {model_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/fibre_dispersion_dataset.csv")
    p.add_argument("--model", default="models/fibre_dispersion_mlp.joblib")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    train(a.data, a.model, a.seed)
