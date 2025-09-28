import os

import pandas as pd
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--normalize", action="store_true", help="Normalize scut dataset")
args = argparser.parse_args()

LABELS_PATH = "datasets/scut/labels"

def normalize_dataset():
    for data in ["train", "test", "all"]:
        if os.path.exists(f"{LABELS_PATH}/processed/{data}.txt"):
            continue
        if not os.path.exists(f"{LABELS_PATH}/raw/{data}.txt"):
            raise FileNotFoundError(f"Raw data file {data}.txt not found in {LABELS_PATH}/raw")
        df = pd.read_csv(f"{LABELS_PATH}/raw/{data}.txt", sep=r"\s+", header=None, names=["name", "score"])
        df["score"] = df["score"].astype(float).apply(lambda x: (x - 1) / 4) # normalize scores [1, 5] -> [0, 1]
        df.to_csv(f"{LABELS_PATH}/processed/{data}.txt", index=False)
        print(f"Normalized {data}.txt")
        
if __name__ == "__main__":
    if args.normalize:
        normalize_dataset()
    else:
        argparser.print_help()