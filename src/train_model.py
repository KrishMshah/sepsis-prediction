import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from preprocessing import drop_high_missingness, impute, split_patients

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --------------------------------------------------------------------------- #
#  Load engineered data, split, SMOTE                                          
# --------------------------------------------------------------------------- #

def load_splits(config: dict):
    df = pd.read_parquet("data/interim/features.parquet")

    train_df, test_df = split_patients(
        df,
        test_size    = config["preprocessing"]["test_size"],
        random_state = config["model"]["random_state"]
    )

    exclude      = {"patient_id", "iculos_hour", "sepsis_label"}
    feature_cols = [c for c in train_df.columns if c not in exclude]
    X_train = train_df[feature_cols]
    y_train = train_df["sepsis_label"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["sepsis_label"]
    # SMOTE on training only
    smote = SMOTE(random_state=config["preprocessing"]["smote_random_state"])
    X_train, y_train = smote.fit_resample(X_train, y_train)
    # Convert to float32 numpy — halves memory, fixes sklearn indexing crash
    X_train = X_train.astype(np.float32).values
    y_train = y_train.astype(np.int32).values
    X_test  = X_test.astype(np.float32).values
    y_test  = y_test.astype(np.int32).values
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, feature_cols

# --------------------------------------------------------------------------- #
#  Model definitions                                                            
# --------------------------------------------------------------------------- #

def get_models(config: dict) -> dict:
    rs  = config["model"]["random_state"]
    spw = config["model"]["scale_pos_weight"]

    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(
                max_iter     = 2000,
                class_weight = "balanced",
                solver       = "lbfgs",
                random_state = rs
            ))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators = 100,
            max_depth    = 10,
            class_weight = "balanced",
            random_state = rs,
            n_jobs       = 2
        ),
        "xgboost": XGBClassifier(
            tree_method       = "hist",       # memory-efficient histogram method
            scale_pos_weight  = spw,
            eval_metric       = "aucpr",
            random_state      = rs,
            n_jobs            = 2
        )
    }

# --------------------------------------------------------------------------- #
#  GridSearch tuning for XGBoost on a subset                                   
# --------------------------------------------------------------------------- #

def tune_xgboost(X: np.ndarray, y: np.ndarray, config: dict) -> XGBClassifier:
    """
    GridSearch on 400k subsample — enough to find optimal params reliably.
    Final model is then retrained on full training data with best params.
    """
    # Subsample for GridSearch
    idx   = np.random.default_rng(config["model"]["random_state"]).choice(
                len(X), size=400_000, replace=False
            )
    X_sub = X[idx]
    y_sub = y[idx]

    param_grid = {
        "n_estimators"  : [200, 400],
        "max_depth"     : [4, 6],
        "learning_rate" : [0.05, 0.1],
    }

    base = XGBClassifier(
        tree_method       = "hist",
        scale_pos_weight  = config["model"]["scale_pos_weight"],
        eval_metric       = "aucpr",
        random_state      = config["model"]["random_state"],
        n_jobs            = 2
    )

    search = GridSearchCV(
        estimator  = base,
        param_grid = param_grid,
        scoring    = "average_precision",
        cv         = 3,
        verbose    = 1,
        n_jobs     = 1           # no parallel jobs — avoids memory duplication
    )

    search.fit(X_sub, y_sub)
    print(f"Best params : {search.best_params_}")
    print(f"Best AUPRC  : {search.best_score_:.4f}")

    # Retrain best model on full training data
    print("Retraining best XGBoost on full training data...")
    best = XGBClassifier(
        tree_method       = "hist",
        scale_pos_weight  = config["model"]["scale_pos_weight"],
        eval_metric       = "aucpr",
        random_state      = config["model"]["random_state"],
        n_jobs            = 2,
        **search.best_params_
    )
    best.fit(X, y)
    return best

# --------------------------------------------------------------------------- #
#  Train all models                                                             
# --------------------------------------------------------------------------- #

def train_all(X_train: np.ndarray, y_train: np.ndarray, config: dict) -> dict:
    models  = get_models(config)
    trained = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == "logistic_regression":
            # Subsample for LR — scale-sensitive, slow on 2.4M rows
            idx   = np.random.default_rng(42).choice(
                        len(X_train), size=200_000, replace=False
                    )
            model.fit(X_train[idx], y_train[idx])
        elif name == "xgboost":
            model = tune_xgboost(X_train, y_train, config)
        else:
            model.fit(X_train, y_train)
        trained[name] = model
        print(f"{name} done.")
    return trained

# --------------------------------------------------------------------------- #
#  Save                                                                         
# --------------------------------------------------------------------------- #

def save_models(trained: dict, feature_cols: list, config: dict) -> None:
    out_dir = Path(config["paths"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model in trained.items():
        joblib.dump(model, out_dir / f"{name}.pkl")
        print(f"Saved → {out_dir}/{name}.pkl")
    pd.Series(feature_cols).to_csv(out_dir / "feature_cols.csv", index=False)
    print(f"Saved → {out_dir}/feature_cols.csv")

# --------------------------------------------------------------------------- #
#  Main                                                                         
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    config = load_config()
    X_train, y_train, X_test, y_test, feature_cols = load_splits(config)
    trained = train_all(X_train, y_train, config)
    save_models(trained, feature_cols, config)
    np.save("models/X_test.npy", X_test)
    np.save("models/y_test.npy", y_test)

    print("\nAll models trained and saved.")
    print("Run evaluate_model.py next.")
#  the 0.9989 AUPRC is on the SMOTE-balanced subset, not real patient data