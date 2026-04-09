import numpy as np
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, accuracy_score
)
from sklearn.calibration import calibration_curve

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --------------------------------------------------------------------------- #
#  Load test data and models                                                    
# --------------------------------------------------------------------------- #

def load_data(config: dict):
    X_test = np.load("models/X_test.npy")
    y_test = np.load("models/y_test.npy")
    models = {}
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        models[name] = joblib.load(f"models/{name}.pkl")
    return X_test, y_test, models

# --------------------------------------------------------------------------- #
#  Core metrics                                                                 
# --------------------------------------------------------------------------- #

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """
    Compute all metrics on real held-out test data.
    Threshold optimised for F1 — not default 0.5 — since class is imbalanced.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    # Find threshold that maximises F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores  = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores[:-1])]
    y_pred = (y_prob >= best_thresh).astype(int)
    metrics = {
        "model"      : name,
        "Accuracy"   : round(accuracy_score(y_test, y_pred),          4),
        "AUROC"      : round(roc_auc_score(y_test, y_prob),            4),
        "AUPRC"      : round(average_precision_score(y_test, y_prob),  4),
        "F1"         : round(f1_score(y_test, y_pred),                 4),
        "Precision"  : round(precision_score(y_test, y_pred),          4),
        "Recall"     : round(recall_score(y_test, y_pred),             4),
        "Threshold"  : round(best_thresh,                              4),
    }
    return metrics, y_prob

# --------------------------------------------------------------------------- #
#  Plots                                                                        
# --------------------------------------------------------------------------- #

def plot_roc(models_probs: dict, y_test: np.ndarray, out_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, y_prob in models_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — All Models")
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
    plt.title("Precision-Recall Curve — All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "prc_curve.png", dpi=150)
    plt.close()
    print("Saved → prc_curve.png")

def plot_calibration(models_probs: dict, y_test: np.ndarray, out_dir: Path) -> None:
    """
    Calibration curve — checks if predicted probabilities match real outcomes.
    A well-calibrated model at 0.8 predicted risk means ~80% of those patients
    actually had sepsis. Critical for clinical trust and paper quality.
    """
    plt.figure(figsize=(8, 6))
    for name, y_prob in models_probs.items():
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=name)

    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve — All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve.png", dpi=150)
    plt.close()
    print("Saved → calibration_curve.png")

# --------------------------------------------------------------------------- #
#  Main                                                                         
# --------------------------------------------------------------------------- #

def run(config: dict):
    out_dir = Path(config["paths"]["figures"]) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test, models = load_data(config)

    all_metrics  = []
    models_probs = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics, y_prob = evaluate(model, X_test, y_test, name)
        all_metrics.append(metrics)
        models_probs[name] = y_prob

    # Print results table
    results_df = pd.DataFrame(all_metrics).set_index("model")
    print("\n--- Evaluation Results ---")
    print(results_df.to_string())

    # Save results table
    results_df.to_csv(results_dir / "metrics.csv")
    print(f"\nSaved → {results_dir}/metrics.csv")

    # Plots
    plot_roc(models_probs, y_test, out_dir)
    plot_prc(models_probs, y_test, out_dir)
    plot_calibration(models_probs, y_test, out_dir)

    print("\nEvaluation complete")

if __name__ == "__main__":
    config = load_config()
    run(config)