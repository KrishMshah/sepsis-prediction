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
from preprocessing import split_patients

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --------------------------------------------------------------------------- #
#  Get top 30 features from SHAP importance CSV                                
# --------------------------------------------------------------------------- #

def get_top30_features() -> list:
    shap_df = pd.read_csv("outputs/results/shap_feature_importance.csv")
    top30   = shap_df.nlargest(30, "mean_abs_shap")["feature"].tolist()
    print(f"Top 30 features selected:")
    for i, f in enumerate(top30, 1):
        print(f"  {i:2}. {f}")
    return top30

# --------------------------------------------------------------------------- #
#  Load engineered data, filter to top 30, split, SMOTE                       
# --------------------------------------------------------------------------- #

def load_splits(config: dict, top30_cols: list):
    df = pd.read_parquet("data/interim/features.parquet")

    train_df, test_df = split_patients(
        df,
        test_size    = config["preprocessing"]["test_size"],
        random_state = config["model"]["random_state"]
    )

    X_train = train_df[top30_cols]
    y_train = train_df["sepsis_label"]
    X_test  = test_df[top30_cols]
    y_test  = test_df["sepsis_label"]

    smote = SMOTE(random_state=config["preprocessing"]["smote_random_state"])
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_train = X_train.astype(np.float32).values
    y_train = y_train.astype(np.int32).values
    X_test  = X_test.astype(np.float32).values
    y_test  = y_test.astype(np.int32).values
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

# --------------------------------------------------------------------------- #
#  Models                                                                       
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
            tree_method      = "hist",
            scale_pos_weight = spw,
            eval_metric      = "aucpr",
            random_state     = rs,
            n_jobs           = 2
        )
    }

# --------------------------------------------------------------------------- #
#  GridSearch for XGBoost                                                       
# --------------------------------------------------------------------------- #

def tune_xgboost(X: np.ndarray, y: np.ndarray, config: dict) -> XGBClassifier:  # noqa
    rng   = np.random.default_rng(config["model"]["random_state"])
    idx   = rng.choice(len(X), size=400_000, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    param_grid = {
        "n_estimators"  : [200, 400],
        "max_depth"     : [4, 6],
        "learning_rate" : [0.05, 0.1],
    }

    base = XGBClassifier(
        tree_method      = "hist",
        scale_pos_weight = config["model"]["scale_pos_weight"],
        eval_metric      = "aucpr",
        random_state     = config["model"]["random_state"],
        n_jobs           = 2
    )

    search = GridSearchCV(
        estimator  = base,
        param_grid = param_grid,
        scoring    = "average_precision",
        cv         = 3,
        verbose    = 1,
        n_jobs     = 1
    )

    search.fit(X_sub, y_sub)
    print(f"Best params : {search.best_params_}")
    print(f"Best AUPRC  : {search.best_score_:.4f}")

    print("Retraining best XGBoost on full training data...")
    best = XGBClassifier(
        tree_method      = "hist",
        scale_pos_weight = config["model"]["scale_pos_weight"],
        eval_metric      = "aucpr",
        random_state     = config["model"]["random_state"],
        n_jobs           = 2,
        **search.best_params_
    )
    best.fit(X, y)  # X, y = full training data passed into function
    return best

# --------------------------------------------------------------------------- #
#  Train all                                                                    
# --------------------------------------------------------------------------- #

def train_all(X_train: np.ndarray, y_train: np.ndarray, config: dict) -> dict:
    models  = get_models(config)
    trained = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == "logistic_regression":
            idx = np.random.default_rng(42).choice(
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

def save_models(trained: dict, top30_cols: list, config: dict) -> None:
    out_dir = Path(config["paths"]["models"]) / "top30"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, model in trained.items():
        joblib.dump(model, out_dir / f"{name}.pkl")
        print(f"Saved → {out_dir}/{name}.pkl")

    pd.Series(top30_cols).to_csv(out_dir / "feature_cols_top30.csv", index=False)
    print(f"Saved → {out_dir}/feature_cols_top30.csv")

# --------------------------------------------------------------------------- #
#  Main                                                                         
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    config     = load_config()
    top30_cols = get_top30_features()
    X_train, y_train, X_test, y_test = load_splits(config, top30_cols)

    # Save test data for top30 evaluation
    Path("models/top30").mkdir(parents=True, exist_ok=True)
    np.save("models/top30/X_test.npy", X_test)
    np.save("models/top30/y_test.npy", y_test)
    trained = train_all(X_train, y_train, config)
    save_models(trained, top30_cols, config)
    print("\nTop-30 models trained and saved.")
    print("Run evaluate_model_top30.py next.")