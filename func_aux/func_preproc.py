# func_preproc.py
# !/usr/bin/env python

"""
Preprocessing utilities for EVSE power + host kernel-event datasets.

Includes:
- Reproducibility: global seeding (Python/NumPy/TensorFlow).
- Label prep for 3 tasks: binary (benign vs attack), scenario (attack groups), multiattack (raw attack names),
  with a mapping function that normalizes many attack-name variants into broader groups.
- Data loading/cleaning:
  - Power CSV: parse/sort timestamps, add a binary charging-state feature.
  - Host CSV: coerce event columns to numeric, drop constant features and 'time', add state_bin and an index-based timestamp.
- Encoding and splitting:
  - Label-encode targets, drop rows with missing features/labels.
  - Chronological train/val/test split per (Attack, State) to reduce leakage across different conditions.
- Windowing for sequence models:
  - Fit StandardScaler on train numeric features, build sliding windows (with optional episode IDs).
- JSON sanitization:
  - Convert numpy/pandas/Path and NaN/inf values into JSON-safe primitives for logging/serialization.

Main entry points: prepare_labels_for_task(_host), load_power_data, load_and_clean_host_data,
split_bytime, scale_windows / scale_and_window, json_sanitize.
"""


import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import random, math


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def map_attack_to_group(attack: str) -> str:
    if attack is None:
        return "none"

    a = str(attack).strip().lower()

    if a in {"0", "none", "benign", "none (ie. benign)"}:
        return "none"

    if a in {"backdoor", "cryptojacking"}:
        return "host-attack"

    if a in {
        "aggressive-scan",
        "os-fingerpriting",
        "os-fingerprinting",
        "os-scan",
        "service-version-detection",
        "service-detection",
        "serice-detection",
        "service-detection-scan",
        "tcp-port-scan",
        "port-scan",
        "vuln-scan",
        "vulnerability-scan",
    }:
        return "recon"

    if a in {
        "icmp-flood",
        "icmp-fragmentation",
        "icmp-fragmentation_old",
        "pshack-flood",
        "push-ack-flood",
        "syn-flood",
        "syn-stealth",
        "syn-stealth-scan",
        "tcp-flood",
        "upd-flood",
        "udp-flood",
        "synonymousip-flood",
        "synonymous-ip-flood",
        "slowloris-scan",
    }:
        return "DoS"

    return attack


def prepare_labels_for_task(df: pd.DataFrame, task: str):
    df = df.copy()
    task = task.lower()

    if "Attack" not in df.columns:
        raise ValueError("Column 'Attack' is required in the dataframe.")

    if task == "binary":
        if "Label" not in df.columns:
            atk = df["Attack"].astype(str).str.lower()
            df["Label"] = np.where(
                atk.isin(["none", "benign", "none (ie. benign)"]),
                "benign",
                "attack",
            )
        target_col = "Label"

    elif task == "scenario":
        if "Attack-Group" not in df.columns:
            df["Attack-Group"] = df["Attack"].apply(map_attack_to_group)
        target_col = "Attack-Group"

    elif task == "multiattack":
        target_col = "Attack"

    else:
        raise ValueError("Unknown task. Must be 'binary', 'scenario' or 'multiattack'.")

    return df, target_col


def load_power_data(csv_path, col_time):
    df = pd.read_csv(csv_path)

    df["timestamp"] = pd.to_datetime(df[col_time])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["state_bin"] = (df["State"] == "charging").astype(int)

    return df


def encode_y_data(df, feature_cols, target_col="Attack"):
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = df.dropna(subset=feature_cols + [target_col])

    le = LabelEncoder()
    df["y"] = le.fit_transform(df[target_col])

    print("Cleaned dataframe shape:", df.shape)
    print("Features:", feature_cols)
    print(f"Target ({target_col}) classes:", list(le.classes_))

    return df, feature_cols, le


def split_df_per_attack_and_state_chronologically(
    df,
    attack_col="Attack",
    state_col="State",
    time_col="timestamp",
    train_frac=0.7,
    val_frac=0.15,
):
    dfs_train, dfs_val, dfs_test = [], [], []

    df = df.dropna(subset=[attack_col, state_col])

    for (attack, state), df_cs in df.groupby([attack_col, state_col]):
        df_cs = df_cs.sort_values(time_col)
        n = len(df_cs)
        if n == 0:
            continue

        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        if n_train + n_val >= n:
            n_train = max(1, n_train)
            n_val = max(0, min(n - n_train - 1, n_val))

        df_train_cs = df_cs.iloc[:n_train]
        df_val_cs = df_cs.iloc[n_train : n_train + n_val]
        df_test_cs = df_cs.iloc[n_train + n_val :]

        dfs_train.append(df_train_cs)
        dfs_val.append(df_val_cs)
        dfs_test.append(df_test_cs)

        print(
            f"(Attack={attack}, State={state}) -> "
            f"total={n}, train={len(df_train_cs)}, "
            f"val={len(df_val_cs)}, test={len(df_test_cs)}"
        )

    df_train = pd.concat(dfs_train).sort_values(time_col).reset_index(drop=True)
    df_val = pd.concat(dfs_val).sort_values(time_col).reset_index(drop=True)
    df_test = pd.concat(dfs_test).sort_values(time_col).reset_index(drop=True)

    print(
        "Per-Attack & State split shapes:",
        df_train.shape,
        df_val.shape,
        df_test.shape,
    )
    return df_train, df_val, df_test


def make_sequences_from_df_single_class(
    df,
    num_features,
    cat_features,
    scaler,
    seq_len: int,
    step: int = 1,
    label_col: str = "label",
    episode_col: str | None = None,
    return_episode_ids: bool = False,
):
    num_vals = scaler.transform(df[num_features])
    cat_vals = df[cat_features].to_numpy(dtype=float)
    X_all = np.hstack([num_vals, cat_vals])

    y_all = df[label_col].to_numpy()

    if episode_col is not None and episode_col in df.columns:
        ep_all = df[episode_col].to_numpy()
    else:
        ep_all = None

    X_seq, y_seq = [], []
    ep_seq = []
    N = len(df)

    for start in range(0, N - seq_len + 1, step):
        end = start + seq_len
        X_seq.append(X_all[start:end])
        y_seq.append(y_all[end - 1])

        if ep_all is not None and return_episode_ids:
            ep_seq.append(ep_all[end - 1])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    if return_episode_ids and ep_all is not None:
        ep_seq = np.array(ep_seq)
        return X_seq, y_seq, ep_seq
    else:
        return X_seq, y_seq


def scale_windows(
    df_train,
    df_val,
    df_test,
    num_features,
    cat_features,
    seq_len=15,
    step=1,
    label_col: str = "y",
    episode_col: str | None = None,
    return_episode_ids: bool = False,
):
    scaler = StandardScaler()
    scaler.fit(df_train[num_features])

    if return_episode_ids and episode_col is not None:
        X_train, y_train, ep_train = make_sequences_from_df_single_class(
            df_train,
            num_features,
            cat_features,
            scaler,
            seq_len=seq_len,
            step=step,
            label_col=label_col,
            episode_col=episode_col,
            return_episode_ids=True,
        )

        X_val, y_val, ep_val = make_sequences_from_df_single_class(
            df_val,
            num_features,
            cat_features,
            scaler,
            seq_len=seq_len,
            step=step,
            label_col=label_col,
            episode_col=episode_col,
            return_episode_ids=True,
        )

        X_test, y_test, ep_test = make_sequences_from_df_single_class(
            df_test,
            num_features,
            cat_features,
            scaler,
            seq_len=seq_len,
            step=step,
            label_col=label_col,
            episode_col=episode_col,
            return_episode_ids=True,
        )

        print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
        print("Unique labels (train):", np.unique(y_train))

        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            scaler,
            ep_train,
            ep_val,
            ep_test,
        )

    else:
        X_train, y_train = make_sequences_from_df_single_class(
            df_train,
            num_features,
            cat_features,
            scaler,
            seq_len=seq_len,
            step=step,
            label_col=label_col,
        )

        X_val, y_val = make_sequences_from_df_single_class(
            df_val,
            num_features,
            cat_features,
            scaler,
            seq_len=seq_len,
            step=step,
            label_col=label_col,
        )

        X_test, y_test = make_sequences_from_df_single_class(
            df_test,
            num_features,
            cat_features,
            scaler,
            seq_len=seq_len,
            step=step,
            label_col=label_col,
        )

        print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
        print("Unique labels (train):", np.unique(y_train))

        return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def split_bytime(df_power, feature_cols, target_col, train_frac=0.7, val_frac=0.15):
    df_clean, feature_cols, label_encoder = encode_y_data(
        df_power, feature_cols, target_col
    )

    df_train, df_val, df_test = split_df_per_attack_and_state_chronologically(
        df_clean,
        attack_col="Attack",
        state_col="State",
        time_col="timestamp",
        train_frac=train_frac,
        val_frac=val_frac,
    )

    df_train_ = df_train.copy()
    df_val_ = df_val.copy()
    df_test_ = df_test.copy()

    df_train_["split"] = "train"
    df_val_["split"] = "val"
    df_test_["split"] = "test"

    df_for_plot = pd.concat([df_train_, df_val_, df_test_]).sort_values("timestamp")
    return df_train, df_val, df_test, df_for_plot, feature_cols, label_encoder


def scale_and_window(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    seq_len: int,
    step: int,
    label_col: str = "y",
):
    scaler = StandardScaler()
    scaler.fit(df_train[num_features])

    X_train, y_train, id_train = make_sequences_from_df_single_class(
        df=df_train,
        num_features=num_features,
        cat_features=cat_features,
        scaler=scaler,
        seq_len=seq_len,
        step=step,
        label_col=label_col,
        episode_col="Attack",
        return_episode_ids=True,
    )

    X_val, y_val, id_val = make_sequences_from_df_single_class(
        df=df_val,
        num_features=num_features,
        cat_features=cat_features,
        scaler=scaler,
        seq_len=seq_len,
        step=step,
        label_col=label_col,
        episode_col="Attack",
        return_episode_ids=True,
    )

    X_test, y_test, id_test = make_sequences_from_df_single_class(
        df=df_test,
        num_features=num_features,
        cat_features=cat_features,
        scaler=scaler,
        seq_len=seq_len,
        step=step,
        label_col=label_col,
        episode_col="Attack",
        return_episode_ids=True,
    )

    print("[windows] Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler,
        id_train,
        id_val,
        id_test,
    )


def _get_attack_candidates_from_df(df: pd.DataFrame) -> list[str]:
    if "Attack" not in df.columns:
        raise ValueError("Dataframe must contain column 'Attack'.")

    benign_like = {"none", "benign", "none (ie. benign)"}
    attacks = []
    for a in sorted(df["Attack"].dropna().unique()):
        a_str = str(a).strip().lower()
        if a_str not in benign_like:
            attacks.append(str(a))
    return attacks


def _derive_attack_state_per_window(
    df: pd.DataFrame,
    seq_len: int,
    step: int,
):
    attacks = []
    states = []

    attacks_col = df["Attack"].to_numpy()
    states_col = df["State"].to_numpy()
    n = len(df)

    for start in range(0, n - seq_len + 1, step):
        idx_label = start + seq_len - 1
        attacks.append(attacks_col[idx_label])
        states.append(states_col[idx_label])

    return np.array(attacks), np.array(states)


# ----------------------------------------------------------------------
# JSON sanitize helpers
# ----------------------------------------------------------------------
def json_sanitize(obj):
    """
    Convert objects to JSON-serializable structures.
    - np.nan, inf, -inf -> None
    - numpy arrays -> list
    - numpy scalars -> python scalars
    - Path -> str
    - dict/list/tuple -> recursively sanitized
    """
    # None, bool, int, str are fine
    if obj is None or isinstance(obj, (bool, str)):
        return obj

    # Path -> str
    if isinstance(obj, Path):
        return str(obj)

    # Python numbers: handle nan/inf for floats
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, int):
        return obj

    # numpy scalars
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())

    # pandas types (per si de cas)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()

    # dict-like
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [json_sanitize(v) for v in obj]

    # fallback: try cast to str (últim recurs)
    return str(obj)


def load_and_clean_host_data(csv_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load EVSE-B-HPC-Kernel-Events-Combined.csv and prepare a clean dataframe.

    - Converts all event columns to numeric (coerce errors to NaN)
    - Drops 'time' as a feature (keeps ordering only)
    - Drops constant columns
    - Adds 'state_bin' = 1 if State == 'Charging', else 0
    - Adds 'timestamp' = row order (float)

    Returns
    -------
    df : cleaned dataframe
    event_cols : list of event column names (excluding state_bin)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    cols = list(df.columns)
    if "State" not in cols:
        raise ValueError("Column 'State' not found in HOST CSV.")
    idx_state = cols.index("State")

    # Events = all columns before 'State'
    event_cols = cols[:idx_state]

    # Drop 'time' as feature if present
    if "time" in event_cols:
        event_cols.remove("time")

    # Convert events to numeric
    if event_cols:
        df[event_cols] = df[event_cols].apply(pd.to_numeric, errors="coerce")

        # Drop constant columns
        nunique = df[event_cols].nunique(dropna=False)
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            print(
                f"[clean] Dropping {len(const_cols)} constant event cols "
                f"(showing up to 10): {const_cols[:10]}"
            )
            df = df.drop(columns=const_cols)
            event_cols = [c for c in event_cols if c not in const_cols]
    # Add state_bin + timestamp in a de-fragmented way (avoids PerformanceWarning on wide frames)
    df = df.reset_index(drop=True).copy()
    df = df.assign(
        state_bin=df["State"].astype(str).str.lower().eq("charging").astype(np.int8),
        timestamp=df.index.astype(float),
    )
    print(f"Cleaned HOST dataframe shape: {df.shape}")
    print(f"#event_cols after cleaning: {len(event_cols)}")

    return df, event_cols



def prepare_labels_for_task_host(df: pd.DataFrame, task: str):
    df = df.copy()
    task = task.lower()

    if "Attack" not in df.columns:
        raise ValueError("Column 'Attack' is required in the dataframe.")

    if task == "binary":
        atk = df["Attack"].astype(str).str.strip().str.lower()
        benign_like = {"none", "benign", "none (ie. benign)"}

        # Always overwrite Label to avoid mixed types like 0/1 + strings
        df["Label"] = np.where(atk.isin(benign_like), "benign", "attack")

        # Safety: enforce only the two classes
        df["Label"] = df["Label"].astype(str).str.strip().str.lower()
        df = df[df["Label"].isin(["benign", "attack"])].copy()

        target_col = "Label"

    elif task == "scenario":
        if "Attack-Group" not in df.columns:
            df["Attack-Group"] = df["Attack"].apply(map_attack_to_group)
        target_col = "Attack-Group"

    elif task == "multiattack":
        target_col = "Attack"

    else:
        raise ValueError("Unknown task. Must be 'binary', 'scenario' or 'multiattack'.")

    return df, target_col
