import pandas as pd
import os
from tqdm import tqdm

def load_physionet_data(data_path):
    all_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".psv"):
                all_files.append(os.path.join(root, file))
    print(f"Total patient files found: {len(all_files)}")
    df_list = []

    for file in tqdm(all_files):
        patient_id = os.path.basename(file).split(".")[0]
        df = pd.read_csv(file, sep="|")
        df["patient_id"] = patient_id
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    raw_data_path = "data/raw"
    df = load_physionet_data(raw_data_path)
    print("Dataset shape:", df.shape)
    os.makedirs("data/interim", exist_ok=True)
    df.to_csv("data/interim/merged_dataset.csv", index=False)
    print("Merged dataset saved to data/interim/merged_dataset.csv")