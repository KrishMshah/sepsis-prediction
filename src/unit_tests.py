import pandas as pd

X = pd.read_csv("data/processed/X_processed.csv")
print("patient_id" in X.columns)