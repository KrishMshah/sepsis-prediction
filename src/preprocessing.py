import pandas as pd
import os

def preprocess_data(input_path, output_dir):
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    print("Original Shape:", df.shape)

    # -------------------------
    # Sort by patient and ICU time
    # -------------------------
    df = df.sort_values(["patient_id", "ICULOS"])

    # -------------------------
    # Forward fill within each patient
    # -------------------------
    df = df.groupby("patient_id", group_keys=False).apply(lambda x: x.ffill())

    # -------------------------
    # Median imputation (excluding patient_id)
    # -------------------------
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    numeric_cols = numeric_cols.drop("patient_id", errors="ignore")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print("Missing values handled")

    # -------------------------
    # Separate features and labels
    # -------------------------
    y = df["SepsisLabel"]

    # Keep patient_id for grouping during training
    X = df.drop(columns=["SepsisLabel"])
    print("Feature Shape:", X.shape)
    print("Label Shape:", y.shape)

    # -------------------------
    # Save processed data
    # -------------------------

    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, "X_processed.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "y_processed.csv"), index=False)
    print("Processed data saved to:", output_dir)

if __name__ == "__main__":
    preprocess_data(
        input_path="data/interim/merged_dataset.csv",
        output_dir="data/processed"
    )