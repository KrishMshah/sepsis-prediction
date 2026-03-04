import pandas as pd
import os

def preprocess_data(input_path, output_dir):
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    print("Original Shape:", df.shape)

    # Sort patient time series
    df = df.sort_values(["patient_id", "ICULOS"])

    # Forward fill data for each patient
    df = df.groupby("patient_id").ffill().reset_index(drop=True)

    # Median taken where still missing values present
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print("Missing values handled")

    # Separating features & label
    y = df["SepsisLabel"]
    X = df.drop(columns=["SepsisLabel"])
    print("Feature Shape:", X.shape)
    print("Label Shape:", y.shape)

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(f"{output_dir}/X_processed.csv", index=False)
    y.to_csv(f"{output_dir}/y_processed.csv", index=False)
    print("Processed data saved to:", output_dir)

if __name__ == "__main__":
    preprocess_data(
        input_path="data/interim/merged_dataset.csv",
        output_dir="data/processed"
    )