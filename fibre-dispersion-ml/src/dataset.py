import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fibre_model import calculate_dispersion_curve


WAVELENGTHS_NM = np.linspace(1300, 1600, 31)


def make_fibre_designs(n_designs=300, random_state=42):
    """Generate fibre designs using the reconstructed sampling procedure."""
    rng = np.random.default_rng(random_state)
    pairs, seen = [], set()

    while len(pairs) < n_designs:
        pair = (
            round(float(rng.uniform(3.2, 4.0)), 8),
            round(float(rng.uniform(0.0025, 0.0050)), 8),
        )
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    return pd.DataFrame(pairs, columns=["core_radius_um", "delta_n"])


def generate_dataset(n_designs=300, random_state=42):
    designs = make_fibre_designs(n_designs, random_state)
    rows = []

    for design_id, design in designs.iterrows():
        values = calculate_dispersion_curve(
            design.core_radius_um,
            design.delta_n,
            WAVELENGTHS_NM,
        )
        for wavelength, dispersion in zip(WAVELENGTHS_NM, values):
            rows.append([
                design_id,
                design.core_radius_um,
                design.delta_n,
                wavelength,
                dispersion,
            ])

    return pd.DataFrame(
        rows,
        columns=[
            "design_id",
            "core_radius_um",
            "delta_n",
            "wavelength_nm",
            "dispersion_ps_nm_km",
        ],
    )


def split_by_design(data, random_state=42):
    """Split 80/10/10 by fibre design to evaluate generalisation to unseen designs."""
    ids = data.design_id.drop_duplicates().to_numpy()
    train_ids, remaining = train_test_split(
        ids,
        test_size=0.20,
        random_state=random_state,
    )
    val_ids, test_ids = train_test_split(
        remaining,
        test_size=0.50,
        random_state=random_state,
    )

    return tuple(
        data[data.design_id.isin(group)].copy()
        for group in (train_ids, val_ids, test_ids)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/fibre_dispersion_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_dataset(random_state=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    train_df, val_df, test_df = split_by_design(df, args.seed)
    print(f"Generated {len(df)} samples from {df.design_id.nunique()} fibre designs.")
    print(
        f"Train: {len(train_df)} samples / "
        f"{train_df.design_id.nunique()} designs"
    )
    print(
        f"Validation: {len(val_df)} samples / "
        f"{val_df.design_id.nunique()} designs"
    )
    print(
        f"Test: {len(test_df)} samples / "
        f"{test_df.design_id.nunique()} designs"
    )
