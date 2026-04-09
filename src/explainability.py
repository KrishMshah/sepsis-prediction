import numpy as np
import pandas as pd
import joblib
import yaml
import shap
import matplotlib.pyplot as plt
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --------------------------------------------------------------------------- #
#  Load model, test data, feature names                                        
# --------------------------------------------------------------------------- #

def load_artifacts(config: dict):
    model        = joblib.load("models/xgboost.pkl")
    X_test       = np.load("models/X_test.npy")
    feature_cols = pd.read_csv("models/feature_cols.csv").iloc[:, 0].tolist()

    # Use a subsample for SHAP — full 310k rows is very slow
    rng  = np.random.default_rng(config["model"]["random_state"])
    idx  = rng.choice(len(X_test), size=2000, replace=False)
    X_sample = X_test[idx]
    return model, X_sample, feature_cols

# --------------------------------------------------------------------------- #
#  1. Global SHAP — which features matter most across all patients             
# --------------------------------------------------------------------------- #

def global_shap(explainer, shap_values: np.ndarray,
                feature_cols: list, out_dir: Path) -> None:
    """
    Summary plot — shows top 20 features by mean absolute SHAP value.
    Each dot = one patient-hour. Red = high feature value, Blue = low.
    Tells us which features drive sepsis predictions across the whole cohort.
    """
    shap.summary_plot(
        shap_values, features=shap_values,
        feature_names=feature_cols,
        max_display=20,
        show=False
    )
    plt.title("Global SHAP — Top 20 Features")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → shap_global_summary.png")

    # Bar plot — mean absolute SHAP (cleaner for paper)
    shap.summary_plot(
        shap_values, features=shap_values,
        feature_names=feature_cols,
        plot_type="bar",
        max_display=20,
        show=False
    )
    plt.title("Global SHAP — Mean Absolute Importance")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → shap_global_bar.png")

# --------------------------------------------------------------------------- #
#  2. Local SHAP — why was THIS specific patient flagged                       
# --------------------------------------------------------------------------- #

def local_shap(explainer, model, shap_values: np.ndarray, X_sample: np.ndarray,
    feature_cols: list, out_dir: Path) -> None:
    """
    Waterfall plot for 3 patients — one low risk, one medium, one high risk.
    Shows exactly which features pushed the prediction up or down for that patient.
    This is the bedside-level explanation a clinician can actually use.
    """
    # Get predicted probabilities to pick representative patients
    probs = model.predict_proba(X_sample)[:, 1]

    # Pick one patient from each risk tier
    low_idx    = np.argmin(np.abs(probs - 0.05))   # ~5% risk
    medium_idx = np.argmin(np.abs(probs - 0.50))   # ~50% risk
    high_idx   = np.argmax(probs)                   # highest risk

    patients = {
        "low_risk"    : low_idx,
        "medium_risk" : medium_idx,
        "high_risk"   : high_idx
    }

    for label, idx in patients.items():
        shap.waterfall_plot(
            shap.Explanation(
                values    = shap_values[idx],
                base_values = explainer.expected_value,
                data      = X_sample[idx],
                feature_names = feature_cols
            ),
            max_display = 15,
            show        = False
        )
        plt.title(f"Local SHAP — {label.replace('_', ' ').title()} Patient")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_local_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → shap_local_{label}.png")

# --------------------------------------------------------------------------- #
#  3. Temporal SHAP — how feature importance shifts across ICU hours           
# --------------------------------------------------------------------------- #

def temporal_shap(config: dict, out_dir: Path) -> None:
    """
    Novel contribution — plots mean absolute SHAP for top 5 features
    across ICU hours (1-24). Shows that feature importance shifts over time,
    consistent with Shakeri et al. (2021) findings.
    Requires patient_id and iculos_hour — load from features.parquet directly.
    """
    model        = joblib.load("models/xgboost.pkl")
    feature_cols = pd.read_csv("models/feature_cols.csv").iloc[:, 0].tolist()
    df           = pd.read_parquet("data/interim/features.parquet")

    # Filter to first 24 ICU hours and subsample for speed
    df    = df[df["iculos_hour"] <= 24].copy()
    df    = df.groupby("iculos_hour").sample(
                n=min(50, df.groupby("iculos_hour").size().min()),
                random_state=config["model"]["random_state"]
            )

    X     = df[feature_cols].astype(np.float32).values
    hours = df["iculos_hour"].values

    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X)
    shap_df     = pd.DataFrame(np.abs(shap_vals), columns=feature_cols)
    shap_df["iculos_hour"] = hours

    # Top 5 features by overall mean SHAP
    top5 = shap_df[feature_cols].mean().nlargest(5).index.tolist()
    temporal = shap_df.groupby("iculos_hour")[top5].mean()
    plt.figure(figsize=(12, 6))
    for feat in top5:
        plt.plot(temporal.index, temporal[feat], marker="o", label=feat)

    plt.xlabel("ICU Hour")
    plt.ylabel("Mean |SHAP Value|")
    plt.title("Temporal SHAP — Feature Importance Across ICU Hours (1–24)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_temporal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → shap_temporal.png")

# --------------------------------------------------------------------------- #
#  Save top features table                                                      
# --------------------------------------------------------------------------- #

def save_top_features(shap_values: np.ndarray,
    feature_cols: list, results_dir: Path) -> None:
    mean_shap = pd.DataFrame({
        "feature"        : feature_cols,
        "mean_abs_shap"  : np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    mean_shap.to_csv(results_dir / "shap_feature_importance.csv", index=False)
    print("\nTop 10 features by SHAP importance:")
    print(mean_shap.head(10).to_string(index=False))
    print(f"\nSaved → {results_dir}/shap_feature_importance.csv")

# --------------------------------------------------------------------------- #
#  Main                                                                         
# --------------------------------------------------------------------------- #

def run(config: dict):
    out_dir     = Path(config["paths"]["figures"]) / "shap"
    results_dir = Path(config["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading artifacts...")
    model, X_sample, feature_cols = load_artifacts(config)
    xgb_model = joblib.load("models/xgboost.pkl")

    print("Computing SHAP values (2000 samples)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print("Generating global SHAP plots...")
    global_shap(explainer, shap_values, feature_cols, out_dir)

    print("Generating local SHAP plots...")
    local_shap(explainer, xgb_model, shap_values, X_sample, feature_cols, out_dir)

    print("Saving top features table...")
    save_top_features(shap_values, feature_cols, results_dir)

    print("Generating temporal SHAP plot...")
    temporal_shap(config, out_dir)

    print("\nExplainability complete")

if __name__ == "__main__":
    config = load_config()
    run(config)