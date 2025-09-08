# feature_engineering.py
import pandas as pd
import numpy as np
import os

def create_features(input_path="data/raw/simulated_smartgrid.csv",
                    output_path="data/processed/features.csv"):
    """
    Extracts meaningful features from smart grid consumption data.
    """

    df = pd.read_csv(input_path)

    features = df.groupby("household").apply(lambda x: pd.Series({
        "mean_consumption": x["consumption"].mean(),
        "std_consumption": x["consumption"].std(),
        "max_consumption": x["consumption"].max(),
        "min_consumption": x["consumption"].min(),
        "peak_to_avg": x["consumption"].max() / (x["consumption"].mean() + 1e-5),
        "night_consumption": x.loc[(x["hour"] % 24 >= 0) & (x["hour"] % 24 < 6), "consumption"].mean(),
        "day_consumption": x.loc[(x["hour"] % 24 >= 6) & (x["hour"] % 24 < 18), "consumption"].mean(),
        "label": x["label"].iloc[0] 
    }))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=True)
    print(f" Features saved to {output_path}")
    return features


if __name__ == "__main__":
    features = create_features()
    print(features.head())
