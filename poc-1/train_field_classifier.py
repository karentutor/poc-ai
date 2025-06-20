
"""train_field_classifier.py
--------------------------------
Train a simple TF‑IDF + Logistic Regression model that maps PDF widget
label context to a canonical field_name.

Usage
-----
$ pip install scikit-learn joblib rich
$ python train_field_classifier.py widgets.json

The script:
  1. loads widgets.json (array of dicts)
  2. creates a text feature from label_before + text_after + page_heading
  3. splits into train / test
  4. trains a TF‑IDF + LogisticRegression pipeline
  5. prints metrics
  6. saves the model pipeline to ``model.joblib``

Author: ChatGPT
Created: {date}
""".format(date=datetime.date.today())

import argparse, json, re, string, joblib
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# ---------- text cleaning helpers ---------------------------------------
_PUNCT_TABLE = str.maketrans('', '', string.punctuation)

def normalize(text: str) -> str:
    """Lower‑case, strip punctuation and excess whitespace."""
    text = text.lower().translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_feature(rec: dict) -> str:
    parts: List[str] = [
        rec.get("label_before", ""),
        rec.get("text_after", ""),
        rec.get("page_heading", ""),
        rec.get("tooltip", ""),
    ]
    return normalize(" ".join(p for p in parts if p))


# ------------------------ main ------------------------------------------
def main(json_path: Path):
    records = json.loads(Path(json_path).read_text())

    X = [build_feature(r) for r in records]
    y = [r["field_name"] for r in records]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nDetailed report:\n", classification_report(y_test, preds))

    joblib.dump(pipe, "model.joblib")
    print("\nModel saved ➜ model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train field‑name classifier.")
    parser.add_argument("json_path", type=Path, help="Path to widgets.json")
    args = parser.parse_args()
    main(args.json_path)
