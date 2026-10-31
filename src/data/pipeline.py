"""
Per-client stratified split + federated scaler (Phase 0, plan section 2).

Order matters, per the plan: split BEFORE any balancing or scaling, or
test data leaks into the training distribution.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def dedupe_rows(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop exact-duplicate feature rows BEFORE any split.

    Flow-based NIDS datasets (TON_IoT included) routinely produce identical
    feature vectors for many flows once identifier columns (IP, port,
    timestamp) are removed -- e.g. many 'normal' flows, or many instances of
    the same scripted attack, look byte-for-byte identical in feature space.
    A random split on the RAW rows then puts copies of the same row in both
    train and test, which silently inflates reported accuracy: the model
    isn't generalizing, it's recognizing a row it memorized under a
    different split draw. Deduplicating first (keep the first occurrence of
    each distinct feature row) is the fix, and it's why this function runs
    before stratified_split rather than after.
    """
    mask = ~X.duplicated(keep="first")
    return X[mask.values].reset_index(drop=True), y[mask.values]


def stratified_split(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2,
                      seed: int = 42, dedupe: bool = True):
    """Per-client stratified train/test split. Falls back to an unstratified
    split for classes with <2 members (can't stratify a singleton).

    dedupe=True (default) removes exact-duplicate feature rows first --
    see dedupe_rows(). This is on by default because leaving it off produces
    a leakage_check() failure on TON_IoT (measured: 55.6% duplicate rows)."""
    if dedupe:
        X, y = dedupe_rows(X, y)
    counts = pd.Series(y).value_counts()
    singleton_classes = counts[counts < 2].index.tolist()
    if singleton_classes:
        mask = ~pd.Series(y).isin(singleton_classes)
        X, y = X[mask.values], y[mask.values]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def federated_scaler_stats(X_train_per_client: list[np.ndarray]):
    """Each client computes (count, sum, sum_of_squares) locally; this
    function plays the server's role of combining them into one global
    mean/std -- the only aggregate ever shared, matching the plan's
    one-extra-round scaler (section Phase 0)."""
    client_stats = []
    for X in X_train_per_client:
        X = np.asarray(X, dtype=np.float64)  # bool/mixed columns break ** otherwise
        n = X.shape[0]
        s = X.sum(axis=0)
        ss = (X ** 2).sum(axis=0)
        client_stats.append((n, s, ss))

    n_total = sum(s[0] for s in client_stats)
    mean = sum(s[1] for s in client_stats) / n_total
    var = sum(s[2] for s in client_stats) / n_total - mean ** 2
    std = np.sqrt(np.clip(var, 1e-12, None))
    std = np.clip(std, 1e-8, None)
    return mean, std, n_total


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Gate check: leakage test (plan: tests/test_leakage.py)
# ---------------------------------------------------------------------------

def leakage_check(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """No row should appear in both train and test. We hash each row's
    feature values (not index, since index resets) to compare content."""
    train_hashes = set(pd.util.hash_pandas_object(X_train, index=False).values)
    test_hashes = set(pd.util.hash_pandas_object(X_test, index=False).values)
    overlap = train_hashes & test_hashes
    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_overlap_hashes": len(overlap),
        "passed": len(overlap) == 0,
    }
