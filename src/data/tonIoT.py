"""
TON_IoT (train_test_network.csv) loader (Phase 0).

Per the stack review (method issue #1): drop identifier/leakage columns
(src/dst IP, ports) before training, or the model memorizes testbed addresses
instead of learning attack patterns. High-cardinality string columns (DNS
query, HTTP URI, SSL subject/issuer) are hashed rather than dropped outright,
so we don't throw away a feature family entirely. Low-cardinality categoricals
(proto, service) are one-hot encoded. The `type` column is the multiclass
target (normal + 9 attack types); the `label` column (0/1) is kept alongside
for the Phase 4 binary-head zero-day experiment.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DROP_COLS = ["src_ip", "dst_ip", "src_port", "dst_port"]
HASH_COLS = ["dns_query", "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent"]
LOW_CARD_CATEGORICAL = ["proto", "service", "conn_state"]
HASH_BUCKETS = 256


def load_tonIoT(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns (X, y_multiclass, y_binary). y_multiclass is a string label
    (the 'type' column); caller encodes to int after seeing the global
    class list, so new unseen-family splits (Phase 4) stay easy to build."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    y_type = df["type"].copy()
    y_label = df["label"].copy()

    X = df.drop(columns=["type", "label"])

    # Drop pure identifiers -- these never belong in the feature set.
    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])

    # Hash high-cardinality strings into a fixed small space instead of
    # dropping them outright -- keeps *some* signal (e.g. repeated DNS
    # queries during a DoS) without memorizing exact testbed values.
    for col in HASH_COLS:
        if col in X.columns:
            X[col] = X[col].astype(str).apply(
                lambda v: hash(v) % HASH_BUCKETS if v != "-" else -1
            )

    # One-hot encode small categoricals.
    for col in LOW_CARD_CATEGORICAL:
        if col in X.columns:
            X = pd.get_dummies(X, columns=[col], prefix=col)

    # Remaining non-numeric columns ('-' placeholders etc.) -> numeric coercion.
    # NB: pandas >= 2.x can give string columns a dedicated "str" dtype
    # instead of generic "object" -- check is_numeric_dtype directly rather
    # than comparing dtype == object, or these silently pass through as text.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    X = X.fillna(0)
    return X, y_type, y_label


def dirichlet_partition(X: pd.DataFrame, y: pd.Series, n_clients: int,
                         alpha: float, seed: int = 42) -> list[np.ndarray]:
    """Dirichlet-skewed partition of TON_IoT across n_clients, per class,
    so every client's class *mixture* differs by alpha (plan section 0:
    alpha in {0.1, 0.5, 1.0})."""
    rng = np.random.RandomState(seed)
    y_arr = y.reset_index(drop=True).values
    classes = np.unique(y_arr)
    client_indices = [[] for _ in range(n_clients)]
    for c in classes:
        idx_c = np.where(y_arr == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=[alpha] * n_clients)
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        parts = np.split(idx_c, split_points)
        for i, part in enumerate(parts):
            client_indices[i].extend(part.tolist())
    return [np.array(sorted(idx)) for idx in client_indices]
