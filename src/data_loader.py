import os
import yaml
import pandas as pd
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_psv_files(folder: str, max_files: int = None) -> pd.DataFrame:
# Adds patient id from filename
    files = sorted(Path(folder).glob("*.psv"))

    if max_files:
        files = files[:max_files]

    if not files:
        raise FileNotFoundError(f"No .psv files found in: {folder}")

    records = []
    for f in files:
        df = pd.read_csv(f, sep="|")
        df["patient_id"] = f.stem          # e.g. "p000001"
        records.append(df)

    return pd.concat(records, ignore_index=True)


def load_all_data(config: dict) -> pd.DataFrame:
    """
    Load and combine training_setA and training_setB
    Adds an 'iculos_hour' column as a clean integer ICU hour index
    iculos - icu lenght of stay
    """
    set_a = load_psv_files(config["data"]["raw_set_a"])
    set_b = load_psv_files(config["data"]["raw_set_b"])

    df = pd.concat([set_a, set_b], ignore_index=True)

    # Rename for consistency
    df = df.rename(columns={"ICULOS": "iculos_hour", "SepsisLabel": "sepsis_label"})

    # Ensure hour is integer
    df["iculos_hour"] = df["iculos_hour"].astype(int)

    print(f"Loaded {df['patient_id'].nunique():,} patients | {len(df):,} hourly records")
    print(f"Sepsis-positive hours: {df['sepsis_label'].sum():,} "
          f"({df['sepsis_label'].mean() * 100:.2f}%)")

    return df


def save_interim(df: pd.DataFrame, config: dict) -> None:
    out_path = config["data"]["interim_combined"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved interim data → {out_path}")


if __name__ == "__main__":
    config = load_config()
    df = load_all_data(config)
    save_interim(df, config)