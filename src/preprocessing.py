import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import SMOTE


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
#  1. Drop high-missingness columns                                            
# --------------------------------------------------------------------------- #

def drop_high_missingness(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop columns missing more than `threshold` fraction of values."""
    miss_rate = df.isnull().mean()
    drop_cols = miss_rate[miss_rate > threshold].index.tolist()

    protected = {"patient_id", "iculos_hour", "sepsis_label"}
    drop_cols = [c for c in drop_cols if c not in protected]

    print(f"Dropping {len(drop_cols)} columns (>{threshold*100:.0f}% missing): {drop_cols}")
    return df.drop(columns=drop_cols)


# --------------------------------------------------------------------------- #
#  2. Imputation                                                                
# --------------------------------------------------------------------------- #

def impute(df: pd.DataFrame, vital_cols: list, lab_cols: list) -> pd.DataFrame:
    """
    Vitals  → forward-fill per patient (no leakage)
    Labs    → per-patient median, then global median fallback
    All     → final global median fill to catch entirely empty columns
    """
    df = df.sort_values(["patient_id", "iculos_hour"])

    present_vitals = [c for c in vital_cols if c in df.columns]
    present_labs   = [c for c in lab_cols   if c in df.columns]

    df[present_vitals] = (
        df.groupby("patient_id")[present_vitals]
        .transform(lambda x: x.ffill())
    )

    df[present_labs] = (
        df.groupby("patient_id")[present_labs]
        .transform(lambda x: x.fillna(x.median()))
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude      = {"iculos_hour", "sepsis_label"}
    fill_cols    = [c for c in numeric_cols if c not in exclude]

    df[fill_cols] = df[fill_cols].fillna(df[fill_cols].median())
    df[fill_cols] = df[fill_cols].fillna(0)

    print(f"NaNs remaining after imputation: {df[fill_cols].isnull().sum().sum()}")
    return df


# --------------------------------------------------------------------------- #
#  3. Patient-wise train/test split                                             
# --------------------------------------------------------------------------- #

def split_patients(df: pd.DataFrame, test_size: float, random_state: int):
    """
    Split at patient level — all hours of one patient go to either
    train or test, never both. Prevents data leakage.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["patient_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {train_df['patient_id'].nunique():,} patients | {len(train_df):,} records")
    print(f"Test : {test_df['patient_id'].nunique():,} patients | {len(test_df):,} records")

    return train_df, test_df


# --------------------------------------------------------------------------- #
#  4. Save CSVs for review (pre-SMOTE, real patient data only)                 
# --------------------------------------------------------------------------- #

def save_csv_for_review(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Save pre-SMOTE train and test splits as CSV for review purposes.
    These reflect actual patient data, not synthetic SMOTE samples.
    """
    review_dir = Path("data/review")
    review_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(review_dir / "train_review.csv", index=False)
    test_df.to_csv(review_dir  / "test_review.csv",  index=False)

    print(f"Review CSVs saved → {review_dir}/")


# --------------------------------------------------------------------------- #
#  5. SMOTE — applied ONLY on training data                                    
# --------------------------------------------------------------------------- #

def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int):
    """
    Oversample minority class in training data only.
    Must never be applied before the train/test split.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    print(f"After SMOTE — positives: {y_res.sum():,} | "
        f"negatives: {(y_res == 0).sum():,}")

    return X_res, y_res


# --------------------------------------------------------------------------- #
#  6. Main pipeline                                                             
# --------------------------------------------------------------------------- #

def run(config: dict):
    print("Starting preprocessing...")
    df = pd.read_parquet(config["data"]["interim_combined"])

    df = drop_high_missingness(df, config["preprocessing"]["missingness_drop_threshold"])
    df = impute(df, config["features"]["vitals"], config["features"]["labs"])

    train_df, test_df = split_patients(
        df,
        test_size    = config["preprocessing"]["test_size"],
        random_state = config["model"]["random_state"]
    )

    # Save review CSVs before SMOTE (real patient data only)
    save_csv_for_review(train_df, test_df)

    exclude      = {"patient_id", "iculos_hour", "sepsis_label"}
    feature_cols = [c for c in train_df.columns if c not in exclude]

    X_train = train_df[feature_cols]
    y_train = train_df["sepsis_label"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["sepsis_label"]

    X_train, y_train = apply_smote(
        X_train, y_train,
        random_state=config["preprocessing"]["smote_random_state"]
    )

    Path(config["data"]["processed_train"]).parent.mkdir(parents=True, exist_ok=True)

    pd.concat([X_train, y_train], axis=1).to_parquet(
        config["data"]["processed_train"], index=False
    )
    pd.concat([X_test, y_test], axis=1).to_parquet(
        config["data"]["processed_test"], index=False
    )

    print(f"Saved → {config['data']['processed_train']}")
    print(f"Saved → {config['data']['processed_test']}")


if __name__ == "__main__":
    config = load_config()
    run(config)