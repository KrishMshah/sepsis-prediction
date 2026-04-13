import numpy as np
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data():
    X_test = np.load("models/top30/X_test.npy")
    y_test = np.load("models/top30/y_test.npy")

    models = {}
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        models[name] = joblib.load(f"models/top30/{name}.pkl")

    return X_test, y_test, models


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores   = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores[:-1])]
    y_pred      = (y_prob >= best_thresh).astype(int)

    return {
        "model"     : name,
        "AUROC"     : round(roc_auc_score(y_test, y_prob),           4),
        "AUPRC"     : round(average_precision_score(y_test, y_prob), 4),
        "F1"        : round(f1_score(y_test, y_pred),                4),
        "Precision" : round(precision_score(y_test, y_pred),         4),
        "Recall"    : round(recall_score(y_test, y_pred),            4),
        "Threshold" : round(best_thresh,                             4),
    }, y_prob


def plot_roc(models_probs: dict, y_test: np.ndarray, out_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, y_prob in models_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Top 30 Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()
    print("Saved → roc_curve.png")


def plot_prc(models_probs: dict, y_test: np.ndarray, out_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, y_prob in models_probs.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, label=f"{name} (AUPRC={auprc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Top 30 Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "prc_curve.png", dpi=150)
    plt.close()
    print("Saved → prc_curve.png")


def plot_calibration(models_probs: dict, y_test: np.ndarray, out_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, y_prob in models_probs.items():
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve — Top 30 Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve.png", dpi=150)
    plt.close()
    print("Saved → calibration_curve.png")


def plot_comparison(metrics_104: pd.DataFrame,
                    metrics_top30: pd.DataFrame, out_dir: Path) -> None:
    """
    Side by side bar chart comparing XGBoost 104 vs top-30 on key metrics.
    This is the ablation study figure for the paper.
    """
    metrics = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]

    xgb_104   = metrics_104[metrics_104["model"] == "xgboost"][metrics].values[0]
    xgb_top30 = metrics_top30[metrics_top30["model"] == "xgboost"][metrics].values[0]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, xgb_104,   width, label="XGBoost (104 features)", color="#2196F3")
    ax.bar(x + width/2, xgb_top30, width, label="XGBoost (Top 30 features)", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("XGBoost — 104 Features vs Top 30 Features (Ablation Study)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_comparison.png", dpi=150)
    plt.close()
    print("Saved → ablation_comparison.png")


def run(config: dict):
    out_dir     = Path(config["paths"]["figures"]) / "evaluation_top30"
    results_dir = Path(config["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test, models = load_data()

    all_metrics  = []
    models_probs = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics, y_prob = evaluate(model, X_test, y_test, name)
        all_metrics.append(metrics)
        models_probs[name] = y_prob

    results_df = pd.DataFrame(all_metrics).set_index("model")
    print("\n--- Top 30 Evaluation Results ---")
    print(results_df.to_string())

    results_df.to_csv(results_dir / "metrics_top30.csv")
    print(f"\nSaved → {results_dir}/metrics_top30.csv")

    plot_roc(models_probs, y_test, out_dir)
    plot_prc(models_probs, y_test, out_dir)
    plot_calibration(models_probs, y_test, out_dir)

    # Ablation comparison plot
    metrics_104 = pd.read_csv(results_dir / "metrics.csv")
    plot_comparison(metrics_104, results_df.reset_index(), out_dir)

    print("\nTop-30 evaluation complete.")


if __name__ == "__main__":
    config = load_config()
    run(config)