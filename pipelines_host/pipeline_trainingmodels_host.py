#!/usr/bin/env python
"""
Training pipeline for HOST events with the same artifacts/schema as
pipeline_trainingmodels_power.py.

Artifacts:
  - windows.npz
  - model.keras (tcn/lstm) OR xgb_model.pkl (xgb)
  - label_encoder.pkl
  - scaler.pkl
  - metrics.json
  - config.json
  - history.json + training_*.png (tcn/lstm only)
  - operational outputs from evaluate_binary_operational (binary only)

Optional: TOP-K feature selection from a feature_importances.json file.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from func_aux.func_preproc import (
    json_sanitize,
    load_and_clean_host_data,
    prepare_labels_for_task_host,
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
from func_aux.func_test import (
    compute_multiclass_metrics,
    evaluate_binary_operational,
)
from func_aux.func_plot import plot_training_history


def load_topk_features_from_json(
    fi_json_path: Path,
    K: int,
    cat_features_all: Optional[List[str]] = None,
) -> Dict:
    """Load feature_importances.json and return the TOP-K selected features."""
    fi_json_path = Path(fi_json_path)
    with open(fi_json_path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "importances" in obj:
        importances = obj["importances"]
    elif isinstance(obj, list):
        importances = obj
    else:
        raise ValueError(f"Unexpected format in {fi_json_path}")

    importances = sorted(importances, key=lambda d: d.get("mean_drop", 0.0), reverse=True)

    if cat_features_all is None:
        cat_features_all = ["state_bin"]

    K = min(int(K), len(importances))
    topk = importances[:K]
    selected = [imp["feature"] for imp in topk]

    new_cat = [f for f in selected if f in cat_features_all]
    new_num = [f for f in selected if f not in new_cat]

    print("\n=== TOP-K FEATURES ===")
    print(f"K = {K}")
    print("Selected:", selected)
    print("Num:", len(new_num), "| Cat:", new_cat)

    return {
        "K": K,
        "selected_features_raw": selected,
        "new_num_features": new_num,
        "new_cat_features": new_cat,
        "importances": topk,
    }


def run_pipeline1_host(
    task: str,
    csv_path: Path,
    output_root: Path,
    model_type: str = "tcn",
    seq_len: int = 15,
    step: int = 1,
    fpr_target: float = 1e-3,
    sample_period_seconds: float = 1.0,
    seed: int = 42,
    feature_importances_path: Optional[Path] = None,
    K: Optional[int] = None,
) -> None:
    """Train HOST models and save the same artifacts as the POWER pipeline."""
    set_global_seed(seed)

    task = task.lower()
    model_type = model_type.lower()

    if task not in {"binary", "scenario", "multiattack"}:
        raise ValueError("task must be one of: 'binary', 'scenario', 'multiattack'")
    if model_type not in {"tcn", "lstm", "xgb"}:
        raise ValueError("model_type must be one of: 'tcn', 'lstm', 'xgb'")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = f"{task}_{model_type}_seq{seq_len}_step{step}"
    if feature_importances_path is not None and K is not None:
        run_name += f"_top{int(K)}"

    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline1_host] Output dir: {output_dir}")

    df_host, event_cols = load_and_clean_host_data(csv_path)
    df_host, target_col = prepare_labels_for_task_host(df_host, task)

    # Drop known problematic attacks
    DROP_ATTACKS = {"serice-detection", "icmp-fragmentation_old"}
    atk = df_host["Attack"].astype(str).str.strip().str.lower()
    df_host = df_host.loc[~atk.isin(DROP_ATTACKS)].reset_index(drop=True)

    if task == "scenario":
        df_host = df_host.loc[df_host[target_col] != "none"].reset_index(drop=True)
    elif task == "multiattack":
        benign_like = {"none", "benign", "none (ie. benign)"}
        atk_lower = df_host["Attack"].astype(str).str.lower()
        df_host = df_host.loc[~atk_lower.isin(benign_like)].reset_index(drop=True)

    col_time = "timestamp"

    cat_features = ["state_bin"]
    if feature_importances_path is not None and K is not None:
        print(f"[pipeline1_host] Importances: {feature_importances_path}")
        topk_info = load_topk_features_from_json(
            feature_importances_path, K=int(K), cat_features_all=cat_features
        )
        num_features = [f for f in topk_info["new_num_features"] if f in df_host.columns]
        cat_features = [f for f in topk_info["new_cat_features"] if f in df_host.columns]

        if "state_bin" in df_host.columns and "state_bin" not in cat_features:
            cat_features = cat_features + ["state_bin"]

        if not num_features and not cat_features:
            raise ValueError("No TOP-K selected features exist in the dataframe.")
    else:
        num_features = [c for c in event_cols if c in df_host.columns]
        if "state_bin" not in df_host.columns:
            raise ValueError("Expected 'state_bin' to exist after cleaning.")

    feature_cols_all = num_features + cat_features
    print(f"[pipeline1_host] Features: {len(feature_cols_all)} (num={len(num_features)}, cat={cat_features})")

    df_train, df_val, df_test, _, feature_cols_used, label_encoder = split_bytime(
        df_host,
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

    test_global_metrics = compute_multiclass_metrics(y_test, y_test_proba)

    y_test_pred = np.argmax(y_test_proba, axis=1)
    label_indices = np.arange(len(label_encoder.classes_))
    cm_global = confusion_matrix(y_test, y_test_pred, labels=label_indices)
    test_confusion_matrix = {
        "labels": label_encoder.classes_.tolist(),
        "matrix": cm_global.tolist(),
    }

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

    binary_state_metrics: Dict[str, Dict] = {}
    if task == "binary" and "state_bin" in cat_features:
        idx_state = (num_features + cat_features).index("state_bin")
        state_bin_last = X_test[:, -1, idx_state]
        state_labels = np.where(state_bin_last >= 0.5, "charging", "idle")

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

    state_multiclass_metrics: Dict[str, Dict] = {}
    if task in {"scenario", "multiattack"} and "state_bin" in cat_features:
        idx_state = (num_features + cat_features).index("state_bin")
        state_bin_last = X_test[:, -1, idx_state]
        state_labels = np.where(state_bin_last >= 0.5, "charging", "idle")

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
        "seq_len": int(seq_len),
        "step": int(step),
        "fpr_target": float(fpr_target),
        "sample_period_seconds": float(sample_period_seconds),
        "num_features": list(num_features),
        "cat_features": list(cat_features),
        "seed": int(seed),
    }
    if feature_importances_path is not None and K is not None:
        config["feature_importances_path"] = str(feature_importances_path)
        config["K_top_features"] = int(K)

    with open(output_dir / "config.json", "w") as f:
        json.dump(json_sanitize(config), f, indent=2, allow_nan=False)

    print(f"[done] Pipeline1 HOST finished -> {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HOST models (same outputs as power pipeline).")
    p.add_argument("--task", required=True, choices=["binary", "scenario", "multiattack"])
    p.add_argument("--csv_path", required=True, type=Path)
    p.add_argument("--output_root", required=True, type=Path)

    p.add_argument("--model_type", default="tcn", choices=["tcn", "lstm", "xgb"])
    p.add_argument("--seq_len", default=15, type=int)
    p.add_argument("--step", default=1, type=int)

    p.add_argument("--fpr_target", default=1e-3, type=float)
    p.add_argument("--sample_period_seconds", default=5.0, type=float)
    p.add_argument("--seed", default=42, type=int)

    p.add_argument("--feature_importances_path", default=None, type=Path)
    p.add_argument("--K", default=None, type=int)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if (args.feature_importances_path is None) != (args.K is None):
        raise ValueError("Use both --feature_importances_path and --K, or neither.")

    run_pipeline1_host(
        task=args.task,
        csv_path=args.csv_path,
        output_root=args.output_root,
        model_type=args.model_type,
        seq_len=args.seq_len,
        step=args.step,
        fpr_target=args.fpr_target,
        sample_period_seconds=args.sample_period_seconds,
        seed=args.seed,
        feature_importances_path=args.feature_importances_path,
        K=args.K,
    )
