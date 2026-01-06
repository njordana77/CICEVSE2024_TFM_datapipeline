#!/usr/bin/env python

"""
End-to-end training + evaluation pipeline for EVSE power-based attack detection.

Workflow:
- Set deterministic seeds and create a run-specific output directory.
- Load the power CSV, build labels for the selected task:
  - binary (benign vs attack), scenario (attack-group), or multiattack (raw attack names),
  with optional filtering to remove benign/none classes for non-binary tasks.
- Define features (numeric power signals + binary charging-state), then:
  - split chronologically per (Attack, State) into train/val/test,
  - standardize numeric features using train-only statistics,
  - create sliding-window sequences (seq_len, step) and save them to windows.npz.
- Train a model:
  - deep sequence model (TCN/LSTM) with Keras (saves model.keras + history.json + loss/accuracy PNGs), or
  - XGBoost on flattened windows (saves xgb_model.pkl).
- Evaluate and record artifacts:
  - global multiclass “research” metrics + global confusion matrix (all tasks),
  - binary-only operational evaluation targeting a desired FPR (and per-state operational metrics),
  - per-state metrics (charging vs idle), including per-state confusion matrices for scenario/multiattack.
- Persist everything needed to reproduce runs:
  label_encoder.pkl, scaler.pkl, metrics.json (JSON-sanitized), and config.json.

Main entry point: run_pipeline1(...).
"""

import argparse
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
)

import joblib
from xgboost import XGBClassifier

from func_aux.func_preproc import (
    set_global_seed,
    load_power_data,
    split_bytime,
    prepare_labels_for_task,
    scale_and_window,
    json_sanitize,
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
from func_aux.func_plot import (
    plot_training_history,
)



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
) -> None:
    set_global_seed(seed)

    task = task.lower()
    model_type = model_type.lower()

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = f"{task}_{model_type}_seq{seq_len}_step{step}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline1] Output dir : {output_dir}")

    # Load + label data
    df_power = load_power_data(str(csv_path), col_time=col_time)
    df_power, target_col = prepare_labels_for_task(df_power, task)

    if task == "scenario":
        df_power = df_power[df_power[target_col] != "none"].reset_index(drop=True)
    elif task == "multiattack":
        benign_like = {"none", "benign", "none (ie. benign)"}
        atk_lower = df_power["Attack"].astype(str).str.lower()
        df_power = df_power[~atk_lower.isin(benign_like)].reset_index(drop=True)

    num_features = ["current_mA", "bus_voltage_V", "power_mW"]
    cat_features = ["state_bin"]
    feature_cols_all = num_features + cat_features

    df_train, df_val, df_test, _, feature_cols_used, label_encoder = split_bytime(
        df_power,
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
            X_train, y_train, X_val, y_val,
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
            X_train_f, y_train, X_val_f, y_val, X_test_f,
            label_encoder=label_encoder,
            use_class_weights=True,
        )

    avg_latency = pred_time / len(X_test)

    # GLOBAL research metrics

    test_global_metrics = compute_multiclass_metrics(y_test, y_test_proba)


    # GLOBAL confusion matrix (all tasks)
    y_test_pred = np.argmax(y_test_proba, axis=1)
    label_indices = np.arange(len(label_encoder.classes_))

    cm_global = confusion_matrix(
        y_test,
        y_test_pred,
        labels=label_indices,
    )
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

    # persatte metrics (BINARY)
    binary_state_metrics: Dict[str, Dict] = {}

    if task == "binary" and "state_bin" in cat_features:
        idx_state = feature_cols_all.index("state_bin")
        state_bin_last = X_test[:, -1, idx_state]
        state_labels = np.where(state_bin_last >= 0.5, "charging", "idle")

        for state in ["charging", "idle"]:
            mask = state_labels == state
            if not mask.any():
                continue

            state_dir = output_dir / f"operational_{state}"
            state_dir.mkdir(parents=True, exist_ok=True)

            # Research
            research_metrics = compute_multiclass_metrics(
                y_test[mask], y_test_proba[mask]
            )

            # Operational
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

            binary_state_metrics[state] = {
                "research": research_metrics,
                "operational": op_metrics,
            }

    # perstate metrics for scenario & multiattack
    # (idle / charging, with confusion matrices)
    state_multiclass_metrics: Dict[str, Dict] = {}

    if task in {"scenario", "multiattack"} and "state_bin" in cat_features:
        idx_state = feature_cols_all.index("state_bin")
        state_bin_last = X_test[:, -1, idx_state]
        state_labels = np.where(state_bin_last >= 0.5, "charging", "idle")

        for state in ["charging", "idle"]:
            mask = state_labels == state
            if not mask.any():
                continue

            y_true_state = y_test[mask]
            y_proba_state = y_test_proba[mask]
            y_pred_state = np.argmax(y_proba_state, axis=1)

            # Confusion matrix over all known labels
            cm_state = confusion_matrix(
                y_true_state,
                y_pred_state,
                labels=label_indices,
            )

            # Only compute "research" metrics if at least 2 classes present
            unique_labels_state = np.unique(y_true_state)
            research_metrics_state = None
            if unique_labels_state.size > 1:
                research_metrics_state = compute_multiclass_metrics(
                    y_true_state, y_proba_state
                )

            state_multiclass_metrics[state] = {
                "research": research_metrics_state,
                "confusion_matrix": {
                    "labels": label_encoder.classes_.tolist(),
                    "matrix": cm_state.tolist(),
                },
            }


    # Save artifacts
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

    # sanitize before dumping, and (optionally) enforce strict JSON
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

    print(f"[done] Pipeline1 finished → {output_dir}")
