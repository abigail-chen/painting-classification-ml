"""
pred.py
-------
Prediction script for CSC311 Project.
Imports only: standard Python libraries, numpy, pandas.
No sklearn, PyTorch, or TensorFlow.

Usage:
    from pred import predict_all
    predictions = predict_all("test_data.csv")
"""

import os
import re
import json
import math

import numpy as np
import pandas as pd

# ── Globals (loaded once) ───────────────────────────────────────────────────────
_PARAMS = None

# ── Column name constants ───────────────────────────────────────────────────────
_EMOTION_CANDIDATES = [
    "On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?",
    "On a scale of 1-10, how intense is the emotion conveyed by the artwork?",
]
_COLOUR_COL  = "How many prominent colours do you notice in this painting?"
_OBJECTS_COL = "How many objects caught your eye in the painting?"
_MONEY_COL   = "How much (in Canadian dollars) would you be willing to pay for this painting?"
_LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]
_MULTI_SELECT_COLS = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
]
_TEXT_COLS = [
    "Describe how this painting makes you feel.",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
]
_LOG_COLS = [_COLOUR_COL, _OBJECTS_COL, _MONEY_COL]


# ── Load parameters ─────────────────────────────────────────────────────────────
def _load_params():
    global _PARAMS
    if _PARAMS is not None:
        return _PARAMS
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "model_params.json")
    with open(path, "r") as f:
        _PARAMS = json.load(f)
    return _PARAMS


# ── Preprocessing helpers ───────────────────────────────────────────────────────
def _extract_rating(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    m = re.match(r"^\s*(\d+)", str(x))
    return float(m.group(1)) if m else float("nan")


def _clean_text(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).lower().strip()
    return re.sub(r"\s+", " ", s)


def _clean_multiselect(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "no answer"
    s = str(x).lower().strip()
    for sep in [";", "|", "/"]:
        s = s.replace(sep, ",")
    parts = sorted(p.strip() for p in s.split(",") if p.strip())
    return ",".join(parts) if parts else "no answer"


def _parse_money(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    s = str(x).lower().strip()
    for tok in ["$", ",", "cad", "dollars", "dollar"]:
        s = s.replace(tok, "")
    s = re.sub(r"\s+", " ", s)
    rng = re.search(r"(\d*\.?\d+)\s*(k?)\s*-\s*(\d*\.?\d+)\s*(k?)", s)
    if rng:
        a, b = float(rng.group(1)), float(rng.group(3))
        if rng.group(2) == "k": a *= 1000
        if rng.group(4) == "k": b *= 1000
        return (a + b) / 2
    single = re.search(r"(?<!\w)(-?\d*\.?\d+)\s*(k?)", s)
    if single:
        v = float(single.group(1))
        if single.group(2) == "k": v *= 1000
        return v
    return float("nan")


# ── TF-IDF (pure Python/numpy) ──────────────────────────────────────────────────
def _get_ngrams(tokens, ngram_range):
    """Yield all n-grams for n in ngram_range."""
    lo, hi = ngram_range
    for n in range(lo, hi + 1):
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i:i + n])


def _tfidf_transform(texts, vocab, idf, max_features, ngram_range):
    """
    Transform a list of strings into a TF-IDF matrix.
    Returns numpy array of shape (n, max_features).
    """
    n = len(texts)
    X = np.zeros((n, max_features), dtype=np.float32)
    ngram_range = tuple(ngram_range)

    for i, text in enumerate(texts):
        tokens = text.split()
        ngrams = list(_get_ngrams(tokens, ngram_range))
        if not ngrams:
            continue
        # count
        counts = {}
        for ng in ngrams:
            if ng in vocab:
                idx = vocab[ng]
                counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            continue
        total = len(ngrams)
        # tf-idf
        for idx, cnt in counts.items():
            tf = cnt / total
            X[i, idx] = tf * idf[idx]
        # l2 normalise row
        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] /= norm
    return X


# ── One-hot encoding (pure Python/numpy) ────────────────────────────────────────
def _ohe_transform(df, multi_select_cols, categories):
    """
    Replicate sklearn OneHotEncoder(handle_unknown='ignore').
    categories: list of lists (one per column), as stored in model_params.json.
    Returns numpy array.
    """
    cols = []
    for col, cats in zip(multi_select_cols, categories):
        col_vals = df[col].astype(str).tolist()
        cat_to_idx = {c: j for j, c in enumerate(cats)}
        mat = np.zeros((len(col_vals), len(cats)), dtype=np.float32)
        for i, v in enumerate(col_vals):
            if v in cat_to_idx:
                mat[i, cat_to_idx[v]] = 1.0
        cols.append(mat)
    return np.hstack(cols)


# ── Decision tree traversal ─────────────────────────────────────────────────────
def _predict_tree(nodes, x_row):
    """Traverse one serialised tree and return the predicted class."""
    idx = 0
    while True:
        node = nodes[idx]
        if node["leaf"]:
            return node["class"]
        if x_row[node["feature"]] <= node["threshold"]:
            idx = node["left"]
        else:
            idx = node["right"]


def _predict_forest(trees, X, classes):
    """
    Majority-vote across all trees.
    X: numpy array (n_samples, n_features)
    Returns list of predicted class strings.
    """
    n = X.shape[0]
    # vote matrix: shape (n, n_classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    votes = np.zeros((n, len(classes)), dtype=np.int32)

    for tree_nodes in trees:
        for i in range(n):
            pred = _predict_tree(tree_nodes, X[i])
            votes[i, class_to_idx[pred]] += 1

    return [classes[int(np.argmax(votes[i]))] for i in range(n)]


# ── Full preprocessing pipeline ─────────────────────────────────────────────────
def _preprocess(df, params):
    df = df.copy()

    # Resolve emotion column
    emotion_col = params["emotion_col"]
    if emotion_col not in df.columns:
        for c in _EMOTION_CANDIDATES:
            if c in df.columns:
                emotion_col = c
                break
        else:
            for c in df.columns:
                if "intense" in c.lower():
                    emotion_col = c
                    break

    # Strip whitespace
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Likert
    likert_cols = params["likert_cols"]
    for col in likert_cols:
        if col in df.columns:
            df[col] = df[col].apply(_extract_rating)

    # Text columns
    for col in _TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_clean_text)

    # Multi-select
    multi_cols = params["multi_select_cols"]
    for col in multi_cols:
        if col in df.columns:
            df[col] = df[col].apply(_clean_multiselect)

    # Money
    if _MONEY_COL in df.columns:
        df[_MONEY_COL] = df[_MONEY_COL].apply(_parse_money)

    # Numeric imputation
    numeric_cols = params["numeric_cols"]
    if emotion_col not in numeric_cols and emotion_col in df.columns:
        numeric_cols = [emotion_col] + [c for c in numeric_cols if c != params["emotion_col"]]
    medians = params["medians"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(medians.get(col, 0))

    # Likert imputation
    modes = params["modes"]
    for col in likert_cols:
        if col in df.columns:
            df[col] = df[col].fillna(modes.get(col, 4))

    # Multi-select NaN fill
    for col in multi_cols:
        if col in df.columns:
            df[col] = df[col].fillna("no answer")

    # Log transform
    for col in _LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Combined text
    parts = []
    for col in _TEXT_COLS:
        if col in df.columns:
            parts.append(df[col].astype(str))
    if parts:
        combined = parts[0]
        for p in parts[1:]:
            combined = combined + " " + p
        df["combined_text"] = combined
    else:
        df["combined_text"] = ""

    return df, emotion_col


# ── Public API ──────────────────────────────────────────────────────────────────
def predict_all(csv_filename: str) -> list:
    """
    Takes the name of a CSV file and returns a list of predictions,
    one per row, in the same order as the input file.

    Parameters
    ----------
    csv_filename : str
        Path to the CSV file containing survey responses.

    Returns
    -------
    list of str
        Predicted painting names:
        'The Persistence of Memory', 'The Starry Night', or 'The Water Lily Pond'.
    """
    params = _load_params()

    df = pd.read_csv(csv_filename)
    df_clean, emotion_col = _preprocess(df, params)

    # Build numeric block
    base_cols = params["base_cols"]
    # remap emotion col in case column name differs
    actual_base = []
    for c in base_cols:
        if c == params["emotion_col"] and c not in df_clean.columns:
            actual_base.append(emotion_col)
        else:
            actual_base.append(c)
    actual_base = [c for c in actual_base if c in df_clean.columns]
    X_num = df_clean[actual_base].astype(float).values

    # Build TF-IDF block
    vocab = params["tfidf_vocabulary"]
    idf   = params["tfidf_idf"]
    texts = df_clean["combined_text"].tolist()
    X_text = _tfidf_transform(texts, vocab, idf,
                              params["tfidf_max_features"],
                              params["tfidf_ngram_range"])

    # Build OHE block
    multi_cols = params["multi_select_cols"]
    multi_cols_present = [c for c in multi_cols if c in df_clean.columns]
    cats_present = [params["ohe_categories"][i]
                    for i, c in enumerate(multi_cols) if c in multi_cols_present]
    X_cat = _ohe_transform(df_clean, multi_cols_present, cats_present)

    # Stack: numeric | categorical | tfidf
    X = np.hstack([X_num, X_cat, X_text])

    # Predict
    predictions = _predict_forest(params["trees"], X, params["classes"])
    return predictions


# ── CLI convenience ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 pred.py <csv_file>")
        sys.exit(1)
    preds = predict_all(sys.argv[1])
    for p in preds:
        print(p)
