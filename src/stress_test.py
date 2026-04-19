import numpy as np
import pandas as pd
import joblib
import yaml
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
#  Load artifacts                                                               #
# --------------------------------------------------------------------------- #

def load_artifacts(config: dict):
    model        = joblib.load("models/xgboost.pkl")
    feature_cols = pd.read_csv("models/feature_cols.csv").iloc[:, 0].tolist()
    df           = pd.read_parquet("data/interim/features.parquet")

    # Sample 200 patients for stress test cohort
    patients = df["patient_id"].unique()
    rng      = np.random.default_rng(config["model"]["random_state"])
    sampled  = rng.choice(patients, size=200, replace=False)

    # Take last hour of each patient — most clinically relevant point
    cohort = (
        df[df["patient_id"].isin(sampled)]
        .sort_values(["patient_id", "iculos_hour"])
        .groupby("patient_id")
        .last()
        .reset_index()
    )

    X = cohort[feature_cols].astype(np.float32).values
    return model, X, feature_cols, cohort


# --------------------------------------------------------------------------- #
#  Apply perturbation                                                           #
# --------------------------------------------------------------------------- #

def perturb(X: np.ndarray, feature_cols: list,
            level: str, config: dict) -> np.ndarray:
    """
    Simulate patient deterioration by perturbing key clinical features.
    Perturbation values defined in config.yaml under stress_test.
    """
    X_perturbed = X.copy()
    changes     = config["stress_test"][level]

    for feature, delta in changes.items():
        if feature in feature_cols:
            idx = feature_cols.index(feature)
            X_perturbed[:, idx] += delta

    return X_perturbed


# --------------------------------------------------------------------------- #
#  Compute ERI                                                                  #
# --------------------------------------------------------------------------- #

def compute_eri(risk_baseline: np.ndarray,
                risk_perturbed: np.ndarray,
                perturbation_magnitude: float) -> dict:
    """
    Escalation Robustness Index (ERI) = mean risk delta / perturbation magnitude.
    Higher ERI = model responds more strongly to worsening conditions.
    pct_risk_increased = % of patients where risk correctly went up.
    """
    delta  = risk_perturbed - risk_baseline
    eri    = float(np.mean(delta) / perturbation_magnitude)
    pct_up = float(np.mean(delta > 0) * 100)

    return {
        "mean_risk_delta"    : round(float(np.mean(delta)), 4),
        "std_risk_delta"     : round(float(np.std(delta)),  4),
        "ERI"                : round(eri,                   4),
        "pct_risk_increased" : round(pct_up,                2)
    }


# --------------------------------------------------------------------------- #
#  SHAP rank correlation                                                        #
# --------------------------------------------------------------------------- #

def shap_rank_correlation(model, X_baseline: np.ndarray,
    X_perturbed: np.ndarray) -> float:
    """
    Spearman rank correlation between baseline and post-stress SHAP rankings.
    1.0 = explanations identical, 0.0 = completely different.
    """
    explainer     = shap.TreeExplainer(model)
    shap_base     = np.abs(explainer.shap_values(X_baseline)).mean(axis=0)
    shap_stressed = np.abs(explainer.shap_values(X_perturbed)).mean(axis=0)
    corr, _       = spearmanr(shap_base, shap_stressed)
    return round(float(corr), 4)


# --------------------------------------------------------------------------- #
#  Plots                                                                        #
# --------------------------------------------------------------------------- #

def plot_risk_shift(baseline: np.ndarray, stressed: dict, out_dir: Path) -> None:
    """
    Box plot showing risk score distribution at baseline vs each stress level.
    """
    labels = ["Baseline"] + list(stressed.keys())
    data   = [baseline]   + list(stressed.values())

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor="lightblue"),
                medianprops=dict(color="red", linewidth=2))
    plt.ylabel("Predicted Sepsis Risk Score")
    plt.title("Risk Score Distribution — Baseline vs Stress Levels")
    plt.tight_layout()
    plt.savefig(out_dir / "risk_shift.png", dpi=150)
    plt.close()
    print("Saved → risk_shift.png")


def plot_eri(eri_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Bar plot of ERI across perturbation levels.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(eri_df["level"], eri_df["ERI"],
            color=["#4CAF50", "#FF9800", "#F44336"])
    plt.ylabel("Escalation Robustness Index (ERI)")
    plt.title("ERI Across Perturbation Levels")
    plt.tight_layout()
    plt.savefig(out_dir / "eri_bar.png", dpi=150)
    plt.close()
    print("Saved → eri_bar.png")


def plot_shap_comparison(model, X_baseline: np.ndarray,
    X_severe: np.ndarray,
    feature_cols: list, out_dir: Path) -> None:
    """
    Side by side bar chart of top 15 SHAP importances before and after severe stress.
    Visually shows whether explanations stay consistent under worst-case deterioration.
    """
    explainer     = shap.TreeExplainer(model)
    shap_base     = np.abs(explainer.shap_values(X_baseline)).mean(axis=0)
    shap_stressed = np.abs(explainer.shap_values(X_severe)).mean(axis=0)

    top_idx  = np.argsort(shap_base)[::-1][:15]
    features = [feature_cols[i] for i in top_idx]
    base_vals    = shap_base[top_idx]
    stress_vals  = shap_stressed[top_idx]

    x     = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, base_vals,  width, label="Baseline", color="#2196F3")
    ax.bar(x + width/2, stress_vals, width, label="Severe Stress", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel("Mean |SHAP Value|")
    ax.set_title("SHAP Feature Importance — Baseline vs Severe Stress")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "shap_comparison.png", dpi=150)
    plt.close()
    print("Saved → shap_comparison.png")


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #

def run(config: dict):
    out_dir     = Path(config["paths"]["figures"]) / "stress_test"
    results_dir = Path(config["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading artifacts...")
    model, X_baseline, feature_cols, _ = load_artifacts(config)

    risk_baseline = model.predict_proba(X_baseline)[:, 1]
    print(f"Baseline mean risk: {risk_baseline.mean():.4f}")

    levels     = config["stress_test"]["perturbation_levels"]
    magnitudes = {"mild": 3.5, "moderate": 7.0, "severe": 10.5}

    all_eri      = []
    stressed_risks = {}
    X_severe     = None

    for level in levels:
        print(f"\nApplying {level} perturbation...")
        X_perturbed    = perturb(X_baseline, feature_cols, level, config)
        risk_perturbed = model.predict_proba(X_perturbed)[:, 1]

        print(f"  Mean risk: {risk_perturbed.mean():.4f}")

        eri_metrics = compute_eri(risk_baseline, risk_perturbed, magnitudes[level])
        print(f"  ERI: {eri_metrics['ERI']} | "
        f"Risk increased in {eri_metrics['pct_risk_increased']}% of patients")

        print(f"  Computing SHAP rank correlation...")
        corr = shap_rank_correlation(model, X_baseline, X_perturbed)
        print(f"  SHAP rank correlation: {corr}")

        all_eri.append({
            "level"              : level,
            "mean_baseline_risk" : round(float(risk_baseline.mean()), 4),
            "mean_stressed_risk" : round(float(risk_perturbed.mean()), 4),
            **eri_metrics,
            "shap_rank_corr"     : corr
        })

        stressed_risks[level] = risk_perturbed

        if level == "severe":
            X_severe = X_perturbed

    # Save ERI table
    eri_df = pd.DataFrame(all_eri)
    eri_df.to_csv(results_dir / "eri_results.csv", index=False)
    print(f"\nERI Results:\n{eri_df.to_string(index=False)}")
    print(f"\nSaved → {results_dir}/eri_results.csv")

    # Plots
    plot_risk_shift(risk_baseline, stressed_risks, out_dir)
    plot_eri(eri_df, out_dir)
    plot_shap_comparison(model, X_baseline, X_severe, feature_cols, out_dir)

    print("\nStress test complete.")


if __name__ == "__main__":
    config = load_config()
    run(config)