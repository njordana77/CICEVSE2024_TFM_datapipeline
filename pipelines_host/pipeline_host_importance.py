#!/usr/bin/env python3
"""
pipeline_host_importance.py

Train a model on HOST events (EVSE-B-HPC-Kernel-Events-Combined.csv) and compute
permutation feature importance on the validation windows using *weighted* log-loss.

Tasks:
  - binary      -> target = Label (benign vs attack)
  - scenario    -> target = Attack-Group (attack scenario)
  - multiattack -> target = Attack (multi-class)

Models:
  - tcn, lstm, xgb

Permutation importance (THIS MATCHES YOUR CURRENT METHOD):
  For each feature j, we *globally shuffle* X[:, :, j] across all validation windows
  and all timesteps by flattening N×T values, shuffling, and reshaping back.
  This breaks:
    (1) feature↔label alignment (values no longer belong to the right window/label)
    (2) the feature’s within-window temporal structure (autocorrelation/trends/bursts)
  while keeping all other features unchanged.

Outputs in output_dir:
  - windows_host.npz
  - feature_importances.json
  - feature_importances.pkl
  - (optional but useful) model + label_encoder + scaler
"""

from __future__ import annotations

import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from func_aux.func_preproc import (
    set_global_seed,
    prepare_labels_for_task,
    split_bytime,
    scale_windows,
)
from func_aux.func_models import (
    build_lstm_model,
    build_tcn_model,
    train_unimodal_model,
    train_xgb_model,
)
from func_aux.func_test import compute_multiclass_metrics


# Small helper to make JSON dumping robust
def _json_sanitize(x: Any) -> Any:
    """Convert numpy/scalars/arrays to JSON-friendly Python types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]
    return x



# Load + basic cleaning
def load_and_clean_host_data(csv_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load EVSE-B-HPC-Kernel-Events-Combined.csv and return (df, event_cols).

    We assume:
      - "State" exists and separates event columns (left) from metadata (right).
      - Event columns are numeric-like; invalid parses become NaN.
      - We add:
          * state_bin: 1 if State == "charging" else 0
          * timestamp: row index as float for chronological splitting
    """
    df = pd.read_csv(csv_path, low_memory=False)
    cols = list(df.columns)

    if "State" not in cols:
        raise ValueError("Column 'State' not found in HOST CSV.")

    idx_state = cols.index("State")
    event_cols = cols[:idx_state]

    # Drop time if present among event columns
    if "time" in event_cols:
        event_cols.remove("time")

    # Convert events to numeric
    df[event_cols] = df[event_cols].apply(pd.to_numeric, errors="coerce")

    # Drop constant event columns
    nunique = df[event_cols].nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"[clean] Dropping {len(const_cols)} constant event cols (up to 10): {const_cols[:10]}")
        df = df.drop(columns=const_cols)
        event_cols = [c for c in event_cols if c not in const_cols]

    # Add state_bin
    df["state_bin"] = df["State"].astype(str).str.lower().eq("charging").astype(int)

    # Use file order as timestamp
    df = df.reset_index(drop=True)
    df["timestamp"] = df.index.astype(float)

    return df, event_cols


# 2) Weighted log-loss + permutation importance (global shuffle)
def compute_weighted_logloss(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_weights: Dict[int, float],
    eps: float = 1e-12,
) -> float:
    """
    Weighted sparse multiclass log-loss.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_proba.ndim != 2:
        raise ValueError(f"y_proba must be 2D (N, C). Got {y_proba.shape}.")
    if y_proba.shape[0] != y_true.shape[0]:
        raise ValueError("Mismatch: y_true length != y_proba rows.")

    y_proba = np.clip(y_proba, eps, 1.0 - eps)
    idx = np.arange(len(y_true))
    p_true = y_proba[idx, y_true]
    losses = -np.log(p_true)

    w = np.array([class_weights[int(c)] for c in y_true], dtype=float)
    return float(np.sum(losses * w) / np.sum(w))


def permutation_importance_windows(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    class_weights: Dict[int, float],
    feature_indices: Optional[np.ndarray] = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Tuple[float, List[Dict]]:
    """
    Permutation importance on windows X with shape (N, T, F).
      - Take all N×T values of feature j, shuffle them, reshape back to (N, T).
      - This breaks label alignment and destroys temporal structure for that feature.
    """
    rng = np.random.RandomState(random_state)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)

    if X.ndim != 3:
        raise ValueError(f"X must be (N, T, F). Got {X.shape}.")

    n_samples, seq_len, n_features = X.shape

    if feature_indices is None:
        feature_indices = np.arange(n_features, dtype=int)
    else:
        feature_indices = np.asarray(feature_indices, dtype=int)

    # Baseline
    y_proba = predict_proba_fn(X)
    baseline_loss = compute_weighted_logloss(y, y_proba, class_weights)
    print(f"[perm] Baseline weighted log-loss: {baseline_loss:.6f}")

    importances: List[Dict] = []
    for j in feature_indices:
        drops: List[float] = []

        for _ in range(n_repeats):
            X_perm = X.copy()

            # Global shuffle across windows AND timesteps for feature j
            col = X_perm[:, :, j].reshape(-1)  # (N*T,)
            rng.shuffle(col)
            X_perm[:, :, j] = col.reshape(n_samples, seq_len)

            y_proba_perm = predict_proba_fn(X_perm)
            loss_perm = compute_weighted_logloss(y, y_proba_perm, class_weights)
            drops.append(loss_perm - baseline_loss)

        mean_drop = float(np.mean(drops))
        std_drop = float(np.std(drops))

        feat_name = feature_names[j] if j < len(feature_names) else f"f{j}"
        print(f"[perm] {feat_name:40s} -> Δloss = {mean_drop:.6f} ± {std_drop:.6f}")

        importances.append(
            {
                "feature": feat_name,
                "index": int(j),
                "mean_drop": mean_drop,
                "std_drop": std_drop,
            }
        )

    importances_sorted = sorted(importances, key=lambda d: d["mean_drop"], reverse=True)
    return baseline_loss, importances_sorted



# Main pipeline
def run_host_importance_pipeline(
    csv_path: Path,
    task: str,
    model_type: str,
    output_dir: Path,
    seq_len: int = 15,
    step: int = 1,
    n_repeats: int = 5,
    seed: int = 42,
) -> Tuple[object, List[Dict], float]:
    set_global_seed(seed)

    task = task.lower().strip()
    model_type = model_type.lower().strip()

    if task not in {"binary", "scenario", "multiattack"}:
        raise ValueError("task must be one of: binary, scenario, multiattack")
    if model_type not in {"tcn", "lstm", "xgb"}:
        raise ValueError("model_type must be one of: tcn, lstm, xgb")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] Output dir: {output_dir}")

    # Load + labels
    df_host, event_cols = load_and_clean_host_data(csv_path)
    df_host, target_col = prepare_labels_for_task(df_host, task)

    # Match your power filtering logic
    if task == "scenario":
        df_host = df_host[df_host[target_col] != "none"].reset_index(drop=True)
    elif task == "multiattack":
        benign_like = {"none", "benign", "none (ie. benign)"}
        atk_lower = df_host["Attack"].astype(str).str.lower()
        df_host = df_host[~atk_lower.isin(benign_like)].reset_index(drop=True)

    # Features
    num_features = list(event_cols)
    cat_features = ["state_bin"]
    feature_cols_all = num_features + cat_features

    print(f"[pipeline] num_features: {len(num_features)} | cat_features: {len(cat_features)}")
    print(f"[pipeline] target_col  : {target_col}")

    # Split chronologically by attack & state
    df_train, df_val, df_test, _, feature_cols_used, label_encoder = split_bytime(
        df_host,
        feature_cols=feature_cols_all,
        target_col=target_col,
        train_frac=0.7,
        val_frac=0.15,
    )
    print(f"[pipeline] Split rows: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test")

    # Windowing
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = scale_windows(
        df_train,
        df_val,
        df_test,
        num_features=[c for c in feature_cols_used if c != "state_bin"],
        cat_features=["state_bin"],
        seq_len=seq_len,
        step=step,
        label_col="y",
    )
    print(f"[pipeline] Windows: train {X_train.shape} | val {X_val.shape} | test {X_test.shape}")

    # Save windows
    np.savez(
        output_dir / "windows_host.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    # Feature names must match the last dimension of X
    feature_names_ordered = list(feature_cols_used)

    # Train + evaluate
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

        y_test_proba = model.predict(X_test, verbose=0)
        test_metrics = compute_multiclass_metrics(y_test, y_test_proba)

        baseline_loss, importances = permutation_importance_windows(
            predict_proba_fn=lambda X_: model.predict(X_, verbose=0),
            X=X_val,
            y=y_val,
            feature_names=feature_names_ordered,
            class_weights=class_weights,
            n_repeats=n_repeats,
            random_state=seed,
        )

        # Save model + extras (handy for later)
        model.save(output_dir / "model.keras")
        with open(output_dir / "history.pkl", "wb") as f:
            pickle.dump(history.history, f)

    else:
        # XGB expects flat vectors
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

        test_metrics = compute_multiclass_metrics(y_test, y_test_proba)

        def xgb_predict_proba(X_win: np.ndarray) -> np.ndarray:
            return model.predict_proba(X_win.reshape(len(X_win), -1))

        baseline_loss, importances = permutation_importance_windows(
            predict_proba_fn=xgb_predict_proba,
            X=X_val,
            y=y_val,
            feature_names=feature_names_ordered,
            class_weights=class_weights,
            n_repeats=n_repeats,
            random_state=seed,
        )

        joblib.dump(model, output_dir / "xgb_model.pkl")

    # Save label encoder + scaler
    joblib.dump(label_encoder, output_dir / "label_encoder.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")

    # Save importances + metadata
    metadata = {
        "task": task,
        "model_type": model_type,
        "csv_path": str(csv_path),
        "target_col": str(target_col),
        "seq_len": int(seq_len),
        "step": int(step),
        "n_repeats": int(n_repeats),
        "classes": list(map(str, label_encoder.classes_)),
        "baseline_weighted_logloss": float(baseline_loss),
        "test_metrics": test_metrics,
        "feature_order": feature_names_ordered,
        "perm_scheme": "global_shuffle_across_windows_and_timesteps",
    }

    with open(output_dir / "feature_importances.json", "w") as f:
        json.dump(
            _json_sanitize({"metadata": metadata, "importances": importances}),
            f,
            indent=2,
            allow_nan=False,
        )

    with open(output_dir / "feature_importances.pkl", "wb") as f:
        pickle.dump(importances, f)

    with open(output_dir / "config.json", "w") as f:
        json.dump(_json_sanitize(metadata), f, indent=2, allow_nan=False)

    print(f"[done] Saved outputs → {output_dir}")
    return model, importances, baseline_loss



# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HOST pipeline: train model + permutation feature importance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=str, required=True, help="Path to EVSE-B-HPC-Kernel-Events-Combined.csv")
    p.add_argument("--task", type=str, choices=["binary", "scenario", "multiattack"], required=True)
    p.add_argument("--model-type", type=str, choices=["tcn", "lstm", "xgb"], required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seq-len", type=int, default=15)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--n-repeats", type=int, default=5, help="Permutation repeats per feature.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_host_importance_pipeline(
        csv_path=Path(args.csv),
        task=args.task,
        model_type=args.model_type,
        output_dir=Path(args.output_dir),
        seq_len=args.seq_len,
        step=args.step,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
