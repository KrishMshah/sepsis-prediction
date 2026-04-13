import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --------------------------------------------------------------------------- #
#  Load both ERI result files                                                   
# --------------------------------------------------------------------------- #

def load_eri_results(results_dir: Path) -> tuple:
    eri_104   = pd.read_csv(results_dir / "eri_results.csv")
    return eri_104

# --------------------------------------------------------------------------- #
#  Print clean summary report                                                   
# --------------------------------------------------------------------------- #

def print_report(eri_df: pd.DataFrame) -> None:
    print("=" * 65)
    print("  ESCALATION ROBUSTNESS INDEX (ERI) — FINAL REPORT")
    print("=" * 65)

    for _, row in eri_df.iterrows():
        print(f"\n  Level       : {row['level'].upper()}")
        print(f"  Baseline Risk    : {row['mean_baseline_risk']}")
        print(f"  Stressed Risk    : {row['mean_stressed_risk']}")
        print(f"  Mean Risk Delta  : {row['mean_risk_delta']} ± {row['std_risk_delta']}")
        print(f"  ERI              : {row['ERI']}")
        print(f"  Risk Increased   : {row['pct_risk_increased']}% of patients")
        print(f"  SHAP Rank Corr   : {row['shap_rank_corr']}")
        print(f"  {'-' * 50}")

    print("\n  KEY FINDINGS")
    print(f"  Mean ERI across levels     : {eri_df['ERI'].mean():.4f}")
    print(f"  Mean SHAP rank correlation : {eri_df['shap_rank_corr'].mean():.4f}")
    print(f"  Min pct risk increased     : {eri_df['pct_risk_increased'].min():.1f}%")
    print(f"  Max pct risk increased     : {eri_df['pct_risk_increased'].max():.1f}%")
    print("=" * 65)

# --------------------------------------------------------------------------- #
#  Final summary plot — ERI + SHAP correlation together                        
# --------------------------------------------------------------------------- #

def plot_eri_summary(eri_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Dual-axis plot: ERI on left axis, SHAP rank correlation on right axis.
    Both across perturbation levels. Single figure for paper.
    """
    levels = eri_df["level"].tolist()
    eri    = eri_df["ERI"].tolist()
    corr   = eri_df["shap_rank_corr"].tolist()

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color1 = "#2196F3"
    ax1.set_xlabel("Perturbation Level")
    ax1.set_ylabel("ERI", color=color1)
    ax1.bar(levels, eri, color=color1, alpha=0.7, label="ERI")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, max(eri) * 2)

    ax2 = ax1.twinx()
    color2 = "#F44336"
    ax2.set_ylabel("SHAP Rank Correlation", color=color2)
    ax2.plot(levels, corr, color=color2, marker="o",
    linewidth=2, markersize=8, label="SHAP Rank Corr")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0.9, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("ERI and SHAP Rank Correlation Across Perturbation Levels")
    plt.tight_layout()
    plt.savefig(out_dir / "eri_final_summary.png", dpi=150)
    plt.close()
    print("Saved → eri_final_summary.png")

# --------------------------------------------------------------------------- #
#  Save paper-ready summary CSV                                                 
# --------------------------------------------------------------------------- #

def save_paper_table(eri_df: pd.DataFrame, results_dir: Path) -> None:
    """
    Clean formatted table with only the columns needed for the paper.
    """
    paper_cols = [
        "level", "mean_baseline_risk", "mean_stressed_risk",
        "mean_risk_delta", "ERI", "pct_risk_increased", "shap_rank_corr"
    ]
    paper_df = eri_df[paper_cols].copy()
    paper_df.columns = [
        "Level", "Baseline Risk", "Stressed Risk",
        "Mean Risk Delta", "ERI", "Pct Risk Increased", "SHAP Rank Corr"
    ]
    paper_df.to_csv(results_dir / "eri_paper_table.csv", index=False)
    print(f"Saved → {results_dir}/eri_paper_table.csv")
    print("\nPaper-ready ERI table:")
    print(paper_df.to_string(index=False))

# --------------------------------------------------------------------------- #
#  Main                                                                         
# --------------------------------------------------------------------------- #

def run(config: dict):
    results_dir = Path(config["paths"]["results"])
    out_dir     = Path(config["paths"]["figures"]) / "stress_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    eri_df = load_eri_results(results_dir)
    print_report(eri_df)
    plot_eri_summary(eri_df, out_dir)
    save_paper_table(eri_df, results_dir)
    print("\nERI report complete")
if __name__ == "__main__":
    config = load_config()
    run(config)