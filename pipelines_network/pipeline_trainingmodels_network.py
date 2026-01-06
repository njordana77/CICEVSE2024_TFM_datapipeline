#!/usr/bin/env python

"""
Network-modality training pipeline (pipeline1).

What it does:
- Defines label/ID mappings to normalize raw file-derived labels into:
  - Attack (multiattack classes), ScenarioStr (scenario classes), and Label (binary).
- Optionally preprocesses raw EVSE network CSV folders into a single cleaned CSV:
  - loads per-file CSVs (EVSE-B by default), infers labels from filenames, drops text/fingerprint columns,
    drops NaNs, maps attacks/scenarios/IDs, builds a time column, and writes a preprocessed CSV.
- Loads the (preprocessed or provided) network CSV, keeps EVSE-B only, constructs the task target:
  - task=binary   -> target Label (0/1)
  - task=scenario -> target ScenarioStr (Benign/Recon/DoS/…)
  - task=multiattack -> target Attack (None + attack types)
  and applies task-specific filtering (removes Benign/None where required).
- Cleans features for modeling:
  - ensures state_bin exists, drops leaky ID/time/IP/MAC/file columns and label-derived columns,
    encodes object features to category codes, removes near-empty and constant features,
    and drops rows with NaNs in features/target.
- Splits the dataframe chronologically (train/val/test = 0.7/0.15/0.15), scales numeric features,
  and windows the time series into (seq_len, step) sequences; saves windows.npz.
- Trains the chosen model:
  - tcn/lstm via Keras (saves history.json + training plots)
  - xgb via XGBoost (flattened windows)
  and measures training time + inference latency per window.
- Evaluates and saves metrics:
  - global multiclass “research” metrics + global confusion matrix for all tasks
  - operational binary metrics (FPR-targeted) for task=binary
  - per-state (charging/idle via state_bin) metrics:
    - binary: research + operational per state
    - scenario/multiattack: per-state confusion matrices + optional research metrics
- Writes artifacts to output_root/<task>_<model_type>_seq<seq_len>_step<step>/:
  model file, scaler.pkl, label_encoder.pkl, metrics.json, and config.json.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from func_aux.func_preproc import (
    json_sanitize,
    scale_and_window,
    set_global_seed,
    split_bytime,
)
from func_aux.func_models import (
    build_lstm_model,
    build_tcn_model,
    train_unimodal_model,
    train_xgb_model,
)
from func_aux.func_plot import plot_training_history
from func_aux.func_test import compute_multiclass_metrics, evaluate_binary_operational


# Notebook-derived mappings
LABEL_MAPPING: Dict[str, str] = {
    "Charging Benign": "None",
    "MaliciousEV aggressive scan": "aggressive-scan",
    "MaliciousEV os fingerprinting": "os-fingerpriting",
    "MaliciousEV port scan": "TCP-port-scan",
    "MaliciousEV service detection": "service-version-detection",
    "MaliciousEV syn stealth scan": "syn-stealth-scan",
    "MaliciousEV vulnerability scan": "vulnerability-scan",
    "charging Aggressive scan": "aggressive-scan",
    "charging aggressive scan": "aggressive-scan",
    "charging icmp flood": "icmp-flood",
    "charging icmp fragmentation": "icmp-fragmentation",
    "charging os fingerprinting": "os-fingerpriting",
    "charging port scan": "TCP-port-scan",
    "charging portscan": "TCP-port-scan",
    "charging push ack flood": "pshack-flood",
    "charging service detection": "service-version-detection",
    "charging service detection scan": "service-version-detection",
    "charging slowLoris scan": "slowloris-scan",
    "charging syn flood": "syn-flood",
    "charging syn stealth": "syn-stealth-scan",
    "charging synonymous ip": "synonymousIP-flood",
    "charging synonymous ip flood": "synonymousIP-flood",
    "charging tcp flood": "TCP-flood",
    "charging udp flood": "upd-flood",
    "charging vulnerability scan": "vulnerability-scan",
    "idle aggressive scan": "aggressive-scan",
    "idle benign": "None",
    "idle icmp flood": "icmp-flood",
    "idle icmp fragmentation": "icmp-fragmentation",
    "idle os fingerprinting": "os-fingerpriting",
    "idle port scan": "TCP-port-scan",
    "idle portscan": "TCP-port-scan",
    "idle push ack flood": "pshack-flood",
    "idle service detection": "service-version-detection",
    "idle slowloris scan": "slowloris-scan",
    "idle syn flood": "syn-flood",
    "idle syn stealth scan": "syn-stealth-scan",
    "idle synonymous ip": "synonymousIP-flood",
    "idle synonymous ip flood": "synonymousIP-flood",
    "idle tcp flood": "TCP-flood",
    "idle udp flood": "upd-flood",
    "idle vulnerability scan": "vulnerability-scan",
}

ATTACK_TO_SCENARIO: Dict[str, str] = {
    "None": "Benign",
    "TCP-flood": "DoS",
    "TCP-port-scan": "Recon",
    "aggressive-scan": "Recon",
    "icmp-flood": "DoS",
    "icmp-fragmentation": "DoS",
    "os-fingerpriting": "Recon",
    "pshack-flood": "DoS",
    "service-version-detection": "Recon",
    "slowloris-scan": "DoS",
    "syn-flood": "DoS",
    "syn-stealth-scan": "Recon",
    "synonymousIP-flood": "DoS",
    "upd-flood": "DoS",
    "vulnerability-scan": "Recon",
}

ATTACK_TO_ID: Dict[str, int] = {
    "None": 0,
    "TCP-flood": 12,
    "TCP-port-scan": 5,
    "aggressive-scan": 1,
    "icmp-flood": 2,
    "icmp-fragmentation": 3,
    "os-fingerpriting": 4,
    "pshack-flood": 6,
    "service-version-detection": 7,
    "slowloris-scan": 8,
    "syn-flood": 9,
    "syn-stealth-scan": 10,
    "synonymousIP-flood": 11,
    "upd-flood": 13,
    "vulnerability-scan": 14,
}


ID_TO_ATTACK: Dict[int, str] = {v: k for k, v in ATTACK_TO_ID.items()}

SCENARIO_TO_ID: Dict[str, int] = {
    "Backdoor": 4,
    "Benign": 0,
    "Cryptojacking": 3,
    "DoS": 2,
    "Recon": 1,
}


def _infer_label_from_filename(fname: str) -> str:
    """Notebook logic: label = " ".join(file_name.split(".")[0].split("-")[2:])"""
    stem = Path(fname).stem
    parts = stem.split("-")
    if len(parts) >= 3:
        return " ".join(parts[2:]).strip()
    return stem.strip()


def preprocess_network_csvs(
    evse_a_dir: Path,
    evse_b_dir: Path,
    output_csv: Path,
    col_time: str = "timestamp",
    force: bool = False,
) -> Path:
    """Merge + clean + label Network CSVs like the notebook."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists() and not force:
        print(f"[network-preproc] Found existing preprocessed CSV → {output_csv}")
        return output_csv

    evse_dirs = {"B": Path(evse_b_dir)}

    dfs: List[pd.DataFrame] = []
    n_files = 0

    for evse_label, folder in evse_dirs.items():
        if not folder.exists():
            raise FileNotFoundError(f"Network folder not found: {folder}")

        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".csv"):
                continue

            full_path = folder / fname
            df = pd.read_csv(full_path, low_memory=False)
            n_files += 1

            df["label_mul_raw"] = _infer_label_from_filename(fname)
            df["evse"] = evse_label
            df["source_file"] = fname

            dfs.append(df)

    if not dfs:
        raise RuntimeError("No network CSV files found. Check your EVSE-A/EVSE-B csv directories.")

    ev = pd.concat(dfs, ignore_index=True)
    print(f"[network-preproc] Loaded {n_files} CSV files → merged shape: {ev.shape}")

    drop_cols = ["requested_server_name", "user_agent", "content_type"]
    fingerprint_cols = [c for c in ev.columns if "fingerprint" in str(c).lower()]
    cols_to_drop = list(dict.fromkeys(drop_cols + fingerprint_cols))
    if cols_to_drop:
        ev = ev.drop(columns=cols_to_drop, errors="ignore")
        print(f"[network-preproc] Dropped {len(cols_to_drop)} cols (text/fingerprint).")

    before = len(ev)
    ev = ev.dropna().reset_index(drop=True)
    print(f"[network-preproc] dropna: {before} → {len(ev)} rows")

    mapped_attack = ev["label_mul_raw"].map(LABEL_MAPPING)
    unknown_counts = ev.loc[mapped_attack.isna(), "label_mul_raw"].value_counts().head(20)
    if len(unknown_counts) > 0:
        print("[network-preproc][warn] Unmapped label_mul_raw values (top 20):")
        for k, v in unknown_counts.items():
            print(f"  - {k!r}: {int(v)}")

    ev["Attack"] = mapped_attack.fillna(ev["label_mul_raw"])
    ev["ScenarioStr"] = ev["Attack"].map(ATTACK_TO_SCENARIO).fillna("Benign")

    ev["label_mul_id"] = ev["Attack"].map(ATTACK_TO_ID)
    ev["scenario_id"] = ev["ScenarioStr"].map(SCENARIO_TO_ID)

    ev["label_mul"] = ev["label_mul_id"]
    ev["Scenario"] = ev["scenario_id"]

    ev["Label"] = (ev["Attack"].astype(str).str.lower() != "none").astype(int)

    if col_time not in ev.columns:
        if "time" in ev.columns:
            ev[col_time] = pd.to_numeric(ev["time"], errors="coerce")
            if ev[col_time].isna().any():
                ev[col_time] = np.arange(len(ev), dtype=float)
        else:
            ev[col_time] = np.arange(len(ev), dtype=float)

    ev.to_csv(output_csv, index=False)
    print(f"[network-preproc] Saved preprocessed CSV → {output_csv}")
    return output_csv


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _ensure_state_bin(df: pd.DataFrame) -> pd.DataFrame:
    if "state_bin" in df.columns:
        return df

    if "State" in df.columns:
        state = df["State"].fillna("unknown").astype(str).str.lower().str.strip()
    elif "label_mul_raw" in df.columns:
        state = (
            df["label_mul_raw"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.split()
            .str[0]
        )
    else:
        state = pd.Series(["unknown"] * len(df), index=df.index)

    df["state_bin"] = (state == "charging").astype(int)
    return df


def _load_network_dataframe(csv_path: Path, col_time: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    if col_time not in df.columns:
        if "bidirectional_first_seen_ms" in df.columns:
            df[col_time] = pd.to_numeric(df["bidirectional_first_seen_ms"], errors="coerce")
        elif "timestamp" in df.columns:
            df[col_time] = pd.to_numeric(df["timestamp"], errors="coerce")
        else:
            df[col_time] = np.arange(len(df), dtype=float)

    df[col_time] = pd.to_numeric(df[col_time], errors="coerce")
    df = df.sort_values(col_time).reset_index(drop=True)
    return df


def _prepare_network_target(df: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, str]:
    task = task.lower().strip()

    if "Attack" not in df.columns:
        if "label_mul_raw" in df.columns:
            mapped_attack = df["label_mul_raw"].map(LABEL_MAPPING)
            df["Attack"] = mapped_attack.fillna(df["label_mul_raw"])
        elif "label_mul" in df.columns:
            src = df["label_mul"]
            if pd.api.types.is_numeric_dtype(src):
                df["Attack"] = src.map(ID_TO_ATTACK).fillna(src.astype(str))
            else:
                raw = src.astype(str).str.strip()
                mapped_exact = raw.map(LABEL_MAPPING)
                if mapped_exact.notna().any():
                    df["Attack"] = mapped_exact.fillna(raw)
                else:
                    norm_map = {str(k).strip().lower(): v for k, v in LABEL_MAPPING.items()}
                    df["Attack"] = raw.str.lower().map(norm_map).fillna(raw)
        else:
            raise ValueError("Network CSV must include one of: 'Attack', 'label_mul', or 'label_mul_raw'")

    df["Attack"] = df["Attack"].astype(str).str.strip()
    df.loc[df["Attack"].str.contains("benign", case=False, na=False), "Attack"] = "None"

    if "ScenarioStr" not in df.columns:
        df["ScenarioStr"] = df["Attack"].map(ATTACK_TO_SCENARIO).fillna("Benign")

    if task == "binary":
        if "Label" not in df.columns:
            df["Label"] = (df["Attack"].astype(str).str.lower() != "none").astype(int)
        else:
            if not pd.api.types.is_numeric_dtype(df["Label"]):
                df["Label"] = (df["Label"].astype(str).str.lower() != "none").astype(int)
        target_col = "Label"

    elif task == "scenario":
        target_col = "ScenarioStr"

    elif task == "multiattack":
        target_col = "Attack"

    else:
        raise ValueError(f"Unknown task: {task!r}. Use: binary, scenario, multiattack")

    return df, target_col


def _clean_for_modeling(
    df: pd.DataFrame,
    feature_blocklist: Optional[List[str]],
    target_col: str,
    col_time: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = _ensure_state_bin(df)

    # Drop leaky or ID-like columns if present
    drop_cols = [
        "id",
        "expiration_id",
        "src_ip",
        "dst_ip",
        "src_mac",
        "dst_mac",
        "src_oui",
        "dst_oui",
        "tunnel_id",
        "bidirectional_first_seen_ms",
        "bidirectional_last_seen_ms",
        "src2dst_first_seen_ms",
        "src2dst_last_seen_ms",
        "dst2src_first_seen_ms",
        "dst2src_last_seen_ms",
        "source_file",  # label leaks via filename
    ]

    # Also avoid feeding derived label columns as features
    drop_cols += [
        "label_mul_raw",
        "label_mul_id",
        "scenario_id",
        "label_mul",
        "Scenario",
    ]

    if feature_blocklist:
        drop_cols += feature_blocklist

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Candidate feature columns
    excluded = {target_col, col_time, "y", "Attack", "ScenarioStr", "Label"}
    feature_cols = [c for c in df.columns if c not in excluded]

    # Drop columns with too many NaNs (features only)
    if feature_cols:
        null_ratio = df[feature_cols].isna().mean()
        cols_to_drop_nan = null_ratio[null_ratio > 0.95].index.tolist()
        if cols_to_drop_nan:
            df = df.drop(columns=cols_to_drop_nan)
            feature_cols = [c for c in feature_cols if c not in cols_to_drop_nan]

    # Encode object features
    for c in [c for c in feature_cols if df[c].dtype == "object"]:
        df[c] = df[c].fillna("missing").astype(str)
        df[c] = df[c].astype("category").cat.codes

    # Ensure numeric features
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Remove constant features
    const_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    if const_cols:
        df = df.drop(columns=const_cols)
        feature_cols = [c for c in feature_cols if c not in const_cols]

    if not feature_cols:
        raise ValueError("No features left after network cleaning.")

    # Drop rows with NaNs in features or target
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Network dataframe is empty after cleaning. Check NaNs / filters.")

    cat_features = ["state_bin"] if "state_bin" in feature_cols else []
    num_features = [c for c in feature_cols if c not in cat_features]

    return df, num_features, cat_features



# Main pipeline

def run_pipeline1(
    task: str,
    csv_path: Path,
    col_time: str,
    output_root: Path,
    model_type: str = "tcn",
    seq_len: int = 15,
    step: int = 1,
    fpr_target: float = 1e-3,
    sample_period_seconds: float = 1.0,
    seed: int = 42,
    evse_a_dir: Optional[Path] = None,
    evse_b_dir: Optional[Path] = None,
    force_preproc: bool = False,
) -> None:
    set_global_seed(seed)

    task = task.lower()
    model_type = model_type.lower()

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = f"{task}_{model_type}_seq{seq_len}_step{step}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline1-network] Output dir : {output_dir}")


    # Load + (optional) preprocess + label data

    csv_path = Path(csv_path)
    if (not csv_path.exists()) and (evse_b_dir is not None):
        preproc_path = output_root / "CICEVSE2024_Dataset" / "CICEVSE2024_network_preprocessed.csv"
        csv_path = preprocess_network_csvs(
            evse_a_dir=Path(evse_a_dir) if evse_a_dir is not None else Path("."),
            evse_b_dir=Path(evse_b_dir),
            output_csv=preproc_path,
            col_time=col_time,
            force=force_preproc,
        )

    df_net = _load_network_dataframe(csv_path, col_time=col_time)
    # Keep EVSE-B only
    for _col in ("EVSE", "evse"):
        if _col in df_net.columns:
            df_net[_col] = df_net[_col].astype(str).str.strip().str.upper()
            df_net = df_net[df_net[_col] == "B"].reset_index(drop=True)
            break
    df_net, target_col = _prepare_network_target(df_net, task)

    # Task-specific filtering
    if task == "scenario":
        df_net = df_net[df_net[target_col].astype(str).str.lower() != "benign"].reset_index(drop=True)
    elif task == "multiattack":
        df_net = df_net[df_net[target_col].astype(str).str.lower() != "none"].reset_index(drop=True)

    df_net, num_features, cat_features = _clean_for_modeling(
        df_net,
        feature_blocklist=None,
        target_col=target_col,
        col_time=col_time,
    )

    feature_cols_all = num_features + cat_features

    df_train, df_val, df_test, _, feature_cols_used, label_encoder = split_bytime(
        df_net,
        feature_cols=feature_cols_all,
        target_col=target_col,
        train_frac=0.7,
        val_frac=0.15,
    )

    (
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
    ) = scale_and_window(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        num_features=num_features,
        cat_features=cat_features,
        seq_len=seq_len,
        step=step,
        label_col="y",
    )

    np.savez(
        output_dir / "windows.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


    # Train model

    train_start = time.perf_counter()

    if model_type in {"tcn", "lstm"}:
        build_fn = build_tcn_model if model_type == "tcn" else build_lstm_model
        history, model, class_weights = train_unimodal_model(
            X_train,
            y_train,
            X_val,
            y_val,
            label_encoder=label_encoder,
            build_model=build_fn,
            initial_lr=1e-3,
        )

        with open(output_dir / "history.json", "w") as f:
            json.dump(json_sanitize(history.history), f, indent=2, allow_nan=False)

        plot_training_history(history, output_dir=output_dir, prefix="training")

        train_time = time.perf_counter() - train_start
        pred_start = time.perf_counter()
        y_test_proba = model.predict(X_test, verbose=0)
        pred_time = time.perf_counter() - pred_start

    else:
        X_train_f = X_train.reshape(len(X_train), -1)
        X_val_f = X_val.reshape(len(X_val), -1)
        X_test_f = X_test.reshape(len(X_test), -1)

        model, y_test_proba, train_time, pred_time, class_weights = train_xgb_model(
            X_train_f,
            y_train,
            X_val_f,
            y_val,
            X_test_f,
            label_encoder=label_encoder,
            use_class_weights=True,
        )

    avg_latency = pred_time / len(X_test)

    # GLOBAL research metrics
    test_global_metrics = compute_multiclass_metrics(y_test, y_test_proba)


    # GLOBAL confusion matrix
    y_test_pred = np.argmax(y_test_proba, axis=1)
    label_indices = np.arange(len(label_encoder.classes_))

    cm_global = confusion_matrix(y_test, y_test_pred, labels=label_indices)
    test_confusion_matrix = {
        "labels": label_encoder.classes_.tolist(),
        "matrix": cm_global.tolist(),
    }


    # GLOBAL operational metrics (binary)
    binary_operational_metrics = None
    window_hop_seconds = step * sample_period_seconds

    if task == "binary":
        binary_operational_metrics = evaluate_binary_operational(
            model=model,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            attack_ids_test=id_test,
            label_encoder=label_encoder,
            fpr_target=fpr_target,
            output_dir=output_dir,
            window_hop_seconds=window_hop_seconds,
        )

    # --------------------------------------------------
    # PER-STATE metrics
    # --------------------------------------------------
    binary_state_metrics: Dict[str, Dict[str, Any]] = {}
    state_multiclass_metrics: Dict[str, Dict[str, Any]] = {}

    if "state_bin" in cat_features:
        idx_state = feature_cols_all.index("state_bin")
        state_bin_last = X_test[:, -1, idx_state]
        state_labels = np.where(state_bin_last >= 0.5, "charging", "idle")

        if task == "binary":
            for state in ["charging", "idle"]:
                mask = state_labels == state
                if not mask.any():
                    continue

                state_dir = output_dir / f"operational_{state}"
                state_dir.mkdir(parents=True, exist_ok=True)

                research_metrics = compute_multiclass_metrics(y_test[mask], y_test_proba[mask])
                op_metrics = evaluate_binary_operational(
                    model=model,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test[mask],
                    y_test=y_test[mask],
                    attack_ids_test=None,
                    label_encoder=label_encoder,
                    fpr_target=fpr_target,
                    output_dir=state_dir,
                    window_hop_seconds=window_hop_seconds,
                )

                binary_state_metrics[state] = {"research": research_metrics, "operational": op_metrics}

        if task in {"scenario", "multiattack"}:
            for state in ["charging", "idle"]:
                mask = state_labels == state
                if not mask.any():
                    continue

                y_true_state = y_test[mask]
                y_proba_state = y_test_proba[mask]
                y_pred_state = np.argmax(y_proba_state, axis=1)

                cm_state = confusion_matrix(y_true_state, y_pred_state, labels=label_indices)

                unique_labels_state = np.unique(y_true_state)
                research_metrics_state = None
                if unique_labels_state.size > 1:
                    research_metrics_state = compute_multiclass_metrics(y_true_state, y_proba_state)

                state_multiclass_metrics[state] = {
                    "research": research_metrics_state,
                    "confusion_matrix": {
                        "labels": label_encoder.classes_.tolist(),
                        "matrix": cm_state.tolist(),
                    },
                }

    # --------------------------------------------------
    # Save artifacts
    # --------------------------------------------------
    if model_type in {"tcn", "lstm"}:
        model.save(output_dir / "model.keras")
    else:
        joblib.dump(model, output_dir / "xgb_model.pkl")

    joblib.dump(label_encoder, output_dir / "label_encoder.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")

    metrics = {
        "task": task,
        "model_type": model_type,
        "target_col": str(target_col),
        "col_time": str(col_time),
        "training_time_seconds": float(train_time),
        "avg_inference_latency_per_window_sec": float(avg_latency),
        "test_global_metrics": test_global_metrics,
        "test_confusion_matrix": test_confusion_matrix,
        "binary_operational_metrics": binary_operational_metrics,
        "binary_state_metrics": binary_state_metrics,
        "state_multiclass_metrics": state_multiclass_metrics,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(json_sanitize(metrics), f, indent=2, allow_nan=False)

    config = {
        "task": task,
        "model_type": model_type,
        "csv_path": str(csv_path),
        "col_time": str(col_time),
        "target_col": str(target_col),
        "seq_len": seq_len,
        "step": step,
        "fpr_target": fpr_target,
        "sample_period_seconds": sample_period_seconds,
        "num_features": num_features,
        "cat_features": cat_features,
        "seed": seed,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(json_sanitize(config), f, indent=2, allow_nan=False)

    print(f"[done] Pipeline1-network finished → {output_dir}")
