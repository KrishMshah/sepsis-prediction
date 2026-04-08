import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
#  1. Shock Index                                                               
# --------------------------------------------------------------------------- #

def add_shock_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shock Index = HR / SBP
    Elevated SI (>1.0) indicates haemodynamic instability.
    Clinically validated early warning sign for septic shock.
    """
    df["shock_index"] = df["HR"] / df["SBP"].replace(0, np.nan)
    df["shock_index"] = df["shock_index"].fillna(0)
    return df


# --------------------------------------------------------------------------- #
#  2. Rolling statistics (per patient, no leakage)                             
# --------------------------------------------------------------------------- #

def add_rolling_features(df: pd.DataFrame, windows: list, cols: list) -> pd.DataFrame:
    """
    For each column and each window size, compute:
    - rolling mean  (trend direction)
    - rolling std   (volatility / instability)

    Uses vectorised groupby rolling — avoids slow Python-level transform(lambda).
    """
    df = df.sort_values(["patient_id", "iculos_hour"]).reset_index(drop=True)
    present_cols = [c for c in cols if c in df.columns]

    for w in windows:
        grouped = df.groupby("patient_id", sort=False)
        for col in present_cols:
            roll = grouped[col].rolling(w, min_periods=1)
            df[f"{col}_roll{w}h_mean"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{col}_roll{w}h_std"]  = roll.std().fillna(0).reset_index(level=0, drop=True)

    return df


# --------------------------------------------------------------------------- #
#  3. Delta features (rate of change between consecutive hours)                
# --------------------------------------------------------------------------- #

def add_delta_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Delta = value at hour t minus value at hour t-1, per patient.
    Captures how fast a patient is deteriorating — critical for early prediction.
    """
    df = df.sort_values(["patient_id", "iculos_hour"])
    present_cols = [c for c in cols if c in df.columns]

    for col in present_cols:
        df[f"{col}_delta"] = (
            df.groupby("patient_id")[col]
            .transform(lambda x: x.diff().fillna(0))
        )

    return df


# --------------------------------------------------------------------------- #
#  4. SOFA component proxies                                                   
# --------------------------------------------------------------------------- #

def add_sofa_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified SOFA sub-scores derived from available features
    These approximate the clinical scoring system used in Sepsis-3 definition.

    Renal : Creatinine > 1.2 → score 1, > 2.0 → score 2
    Liver : Bilirubin_total > 1.2 → score 1, > 2.0 → score 2
    Coag  : Platelets < 150 → score 1, < 100 → score 2
    Resp  : SpO2 < 95 → score 1
    """
    scores = pd.DataFrame(index=df.index)

    if "Creatinine" in df.columns:
        scores["sofa_renal"] = np.where(df["Creatinine"] > 2.0, 2,
            np.where(df["Creatinine"] > 1.2, 1, 0))

    if "Bilirubin_total" in df.columns:
        scores["sofa_liver"] = np.where(df["Bilirubin_total"] > 2.0, 2,
            np.where(df["Bilirubin_total"] > 1.2, 1, 0))

    if "Platelets" in df.columns:
        scores["sofa_coag"] = np.where(df["Platelets"] < 100, 2,
            np.where(df["Platelets"] < 150, 1, 0))

    if "O2Sat" in df.columns:
        scores["sofa_resp"] = np.where(df["O2Sat"] < 95, 1, 0)

    scores["sofa_proxy_total"] = scores.sum(axis=1)

    return pd.concat([df, scores], axis=1)


# --------------------------------------------------------------------------- #
#  5. Time in ICU feature                                                       
# --------------------------------------------------------------------------- #

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalised ICU stay progress between 0 and 1 per patient.
    Early ICU hours have very different risk profiles than later hours.
    Shakeri et al. (2021) confirmed feature importance shifts across time windows.
    """
    max_hours = df.groupby("patient_id")["iculos_hour"].transform("max")
    df["icu_stay_progress"] = df["iculos_hour"] / max_hours.replace(0, 1)
    return df


# --------------------------------------------------------------------------- #
#  6. Main pipeline                                                             
# --------------------------------------------------------------------------- #

def run(config: dict):
    print("Starting feature engineering...")

    # Reload pre-split interim data to engineer features before SMOTE
    df = pd.read_parquet(config["data"]["interim_combined"])

    # Replicate same preprocessing steps
    from preprocessing import drop_high_missingness, impute
    df = drop_high_missingness(df, config["preprocessing"]["missingness_drop_threshold"])
    df = impute(df, config["features"]["vitals"], config["features"]["labs"])

    windows    = config["features"]["rolling_windows"]
    roll_cols  = config["features"]["vitals"] + config["features"]["labs"]
    delta_cols = ["Lactate", "MAP", "HR", "Creatinine", "Platelets"]

    print("Adding shock index...")
    df = add_shock_index(df)

    print("Adding rolling features...")
    df = add_rolling_features(df, windows, roll_cols)

    print("Adding delta features...")
    df = add_delta_features(df, delta_cols)

    print("Adding SOFA proxies...")
    df = add_sofa_proxies(df)

    print("Adding time features...")
    df = add_time_features(df)

    total_features = len([c for c in df.columns
        if c not in {"patient_id", "iculos_hour", "sepsis_label"}])
    print(f"Total features after engineering: {total_features}")

    out_path = Path("data/interim/features.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    config = load_config()
    run(config)