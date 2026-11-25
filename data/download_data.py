"""Download the Pima Indians Diabetes Database and save it with proper column names.

Source mirror (raw CSV, no header): Jason Brownlee's public dataset mirror on GitHub.
Kaggle listing named in the ethics form (CC0): see config.yaml -> data.kaggle_listing.
Usage:  python data/download_data.py
"""
from pathlib import Path
import hashlib
import io
import urllib.request

import pandas as pd

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
OUT = Path(__file__).resolve().parent / "pima_diabetes.csv"


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=30).read()
    print("sha256 of downloaded file:", hashlib.sha256(raw).hexdigest())
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLUMNS)
    assert df.shape == (768, 9), f"unexpected shape {df.shape}"
    df.to_csv(OUT, index=False)
    print(f"saved {OUT} shape={df.shape} class counts={df.Outcome.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
