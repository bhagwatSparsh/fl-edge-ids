"""
N-BaIoT loader (Phase 0).

Layout on disk: Dataset/<device_id>.<class>.csv, 115 numeric features, no header
issues (header row present, all numeric). 9 devices = 9 natural FL clients.
Device 3 (Ennio_Doorbell) and device 7 (Samsung_SNH_1011_N_Webcam) have NO
Mirai files at all -- this is a real property of the raw dataset (confirmed
against the user's own data_summary.csv), not something we construct. It is
exactly what makes the Phase 4 "locally unseen attack" experiment meaningful.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DEVICE_NAMES = {
    1: "Danmini_Doorbell",
    2: "Ecobee_Thermostat",
    3: "Ennio_Doorbell",
    4: "Philips_B120N10_Baby_Monitor",
    5: "Provision_PT_737E_Security_Camera",
    6: "Provision_PT_838_Security_Camera",
    7: "Samsung_SNH_1011_N_Webcam",
    8: "SimpleHome_XCS7_1002_WHT_Security_Camera",
    9: "SimpleHome_XCS7_1003_WHT_Security_Camera",
}

# 11 classes: benign + 5 gafgyt + 5 mirai (plan section 2, Phase 0)
CLASS_FILES = [
    "benign",
    "gafgyt.combo", "gafgyt.junk", "gafgyt.scan", "gafgyt.tcp", "gafgyt.udp",
    "mirai.ack", "mirai.scan", "mirai.syn", "mirai.udp", "mirai.udpplain",
]

CLASS_TO_LABEL = {name: i for i, name in enumerate(CLASS_FILES)}
LABEL_TO_CLASS = {i: name for name, i in CLASS_TO_LABEL.items()}


def load_device(data_dir: Path, device_id: int, cap_per_class: int = 20_000,
                 seed: int = 42, fast: bool = True) -> pd.DataFrame:
    """Load one device's CSVs, capped per (device, class), tagged with
    device_id and integer label. Missing class files (e.g. no mirai for
    devices 3 and 7) are silently skipped -- that absence IS the data.

    fast=True (default): read only the first `cap_per_class` rows with
    pandas' own `nrows`, which stops parsing once it has enough rows instead
    of reading the whole file and discarding the rest. For files well over
    100MB this is the difference between ~1s and ~100s per file, which
    matters when the environment running this has to pull the CSVs over a
    slow connection first. The tradeoff: rows are the first N in the file
    rather than a uniform random N -- acceptable here because each file is
    already single-class traffic from one continuous capture, so an
    early-rows slice is not biased toward a different *class* mix, only
    toward an earlier *time window* within that one class.

    fast=False: original behavior -- read the whole file, then draw a true
    uniform-random sample of cap_per_class rows. Use this once data transfer
    isn't the bottleneck (e.g. running Phase 3+ on a fast machine) if a
    referee might ask whether the sampling was random."""
    rng = np.random.RandomState(seed + device_id)
    frames = []
    for cls in CLASS_FILES:
        fpath = data_dir / f"{device_id}.{cls}.csv"
        if not fpath.exists():
            continue
        if fast:
            df = pd.read_csv(fpath, nrows=cap_per_class)
        else:
            df = pd.read_csv(fpath)
            if len(df) > cap_per_class:
                idx = rng.choice(len(df), size=cap_per_class, replace=False)
                df = df.iloc[idx]
        df = df.copy()
        df["label"] = CLASS_TO_LABEL[cls]
        df["device_id"] = device_id
        frames.append(df)
    if not frames:
        raise ValueError(f"No files found for device {device_id} in {data_dir}")
    out = pd.concat(frames, ignore_index=True)
    return out


def load_all_devices(data_dir: Path, cap_per_class: int = 20_000,
                      seed: int = 42, fast: bool = True) -> dict[int, pd.DataFrame]:
    """Returns {device_id: DataFrame}, one entry per of the 9 natural FL clients."""
    return {d: load_device(data_dir, d, cap_per_class, seed, fast) for d in range(1, 10)}


def class_count_table(clients: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Gate check #1: per-client class-count table."""
    rows = []
    for device_id, df in clients.items():
        counts = df["label"].value_counts().reindex(range(len(CLASS_FILES)), fill_value=0)
        row = {"device_id": device_id, "device_name": DEVICE_NAMES[device_id]}
        row.update({LABEL_TO_CLASS[i]: counts[i] for i in range(len(CLASS_FILES))})
        rows.append(row)
    return pd.DataFrame(rows)
