#!/usr/bin/env python
"""
pipeline_tests_host.py
Pipeline 3: robustness & generalization for EVSE-B HOST models.

Assumes you already have runs from pipeline1 (HOST training) stored in a directory like:

    outputs_host/binary_tcn_seq15_step1/
    outputs_host/scenario_xgb_seq10_step1_top70/
    outputs_host/multiattack_lstm_seq15_step1/

with at least:
    - model.keras (for tcn/lstm) OR xgb_model.pkl (for xgb)
    - label_encoder.pkl
    - config.json
and for robustness:
    - windows.npz (X_train, X_val, X_test, y_train, y_val, y_test)

Provides:

  1) run_pipeline3_host_robustness(...)
     - loads the saved model + X_test/y_test (and X_val/y_val for binary operational)
     - applies degradations (packet_loss, missing_variables, etc.) with repeats per severity
     - saves robustness_curves.json and plots
     - if task == 'binary': also computes operational metrics (FPR-target threshold, FPH, etc.)
     - also computes per-state_bin metrics (charging/idle) if state_bin is present in features.

  2) run_pipeline3_host_generalization(...)
     - reloads raw HOST CSV (EVSE-B-HPC-Kernel-Events-Combined.csv)
     - applies HOST data cleaning (numeric conversion, drop constants, add state_bin, add timestamp)
     - applies the same task filtering as pipeline1 (scenario removes 'none'; multiattack removes benign-like)
     - optionally re-applies TOP-K feature selection (using feature_importances.json + K from config)
     - leave-one-attack-out: trains models WITHOUT a given Attack in train/val, evaluates on test
     - aggregates mean/std over several seeds
     - if task == 'binary': also stores operational metrics per run (nested dict)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import sys

# Add project rootto sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from func_aux.func_preproc import (
    set_global_seed,
    split_bytime,
    scale_windows,
    prepare_labels_for_task_host,
    _get_attack_candidates_from_df,
    load_and_clean_host_data,
)

from func_aux.func_models import (
    build_lstm_model,
    build_tcn_model,
    train_unimodal_model,
    train_xgb_model,
)

from func_aux.func_test import (
    compute_multiclass_metrics,
    aggregate_metric_list,
    degrade_sequences,
    evaluate_binary_operational,
)



# Helpers
def _ensure_proba_2d(y_proba: np.ndarray) -> np.ndarray:
    """Ensure predict/proba output is (n, n_classes)."""
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        y_proba = np.vstack([1.0 - y_proba, y_proba]).T
    if y_proba.ndim == 2 and y_proba.shape[1] == 1:
        y_proba = np.hstack([1.0 - y_proba, y_proba])
    return y_proba


class _XGBPredictWrapper:
    """So evaluate_binary_operational can call .predict(X, verbose=0) on an XGB model."""
    def __init__(self, xgb_model):
        self.model = xgb_model

    def predict(self, X, verbose=0):
        Xf = X.reshape(len(X), -1)
        return _ensure_proba_2d(self.model.predict_proba(Xf))


def empty_multiclass_metrics_nan() -> Dict[str, float]:
    """Return a dict with the same keys as compute_multiclass_metrics, filled with NaN."""
    keys = [
        "accuracy",
        "balanced_accuracy",
        "pr_auc_micro",
        "pr_auc_macro",
        "roc_auc_macro",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "f2_macro",
        "f0_5_macro",
    ]
    return {k: float("nan") for k in keys}



def _derive_attack_state_per_window(
    df_test: pd.DataFrame,
    seq_len: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive per-window Attack and State labels by taking the value at the LAST timestep
    of each window.
    """
    n = len(df_test)
    if n < seq_len:
        return np.array([], dtype=object), np.array([], dtype=object)

    n_windows = 1 + (n - seq_len) // step
    end_idxs = (np.arange(n_windows) * step) + (seq_len - 1)

    if "Attack" in df_test.columns:
        attack_test = df_test["Attack"].iloc[end_idxs].fillna("none").astype(str).values
    else:
        attack_test = np.array(["none"] * n_windows, dtype=object)

    if "State" in df_test.columns:
        state_test = df_test["State"].iloc[end_idxs].fillna("unknown").astype(str).values
    else:
        state_test = np.array(["unknown"] * n_windows, dtype=object)

    return attack_test, state_test



# TOP-K selection
def load_topk_features_from_json(
    fi_json_path: Path,
    K: int,
    cat_features_all: Optional[List[str]] = None,
) -> Dict:
    """
    Load feature_importances.json and select top-K features by mean_drop (desc).
    Expected:
      { "metadata": {...}, "importances": [ {"feature":..., "mean_drop":...}, ... ] }
    OR a raw list at root.
    """
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
    raw_selected = [imp["feature"] for imp in topk]

    new_cat_features = [f for f in raw_selected if f in cat_features_all]
    new_num_features = [f for f in raw_selected if f not in new_cat_features]

    return {
        "K": K,
        "selected_features_raw": raw_selected,
        "new_num_features": new_num_features,
        "new_cat_features": new_cat_features,
        "importances": topk,
        "source_importances_json": str(fi_json_path),
    }


def _resolve_features_from_config_and_df(
    config: Dict,
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Determine num_features / cat_features for HOST generalization:
      - If config has feature_importances_path + K_top_features, reselect topK
      - Else use config['num_features'] + config['cat_features']
    Always intersects with df.columns.
    """
    cat_all = ["state_bin"]

    fi_path = config.get("feature_importances_path", None)
    K = config.get("K_top_features", None)

    if fi_path and K:
        topk = load_topk_features_from_json(Path(fi_path), int(K), cat_features_all=cat_all)
        num_features = [c for c in topk["new_num_features"] if c in df.columns]
        cat_features = [c for c in topk["new_cat_features"] if c in df.columns]

        if not num_features and not cat_features:
            raise ValueError("None of the TOP-K features exist in the cleaned HOST dataframe.")
        return num_features, cat_features

    num_features = [c for c in config.get("num_features", []) if c in df.columns]
    cat_features = [c for c in config.get("cat_features", []) if c in df.columns]
    if not num_features and not cat_features:
        raise ValueError("No usable features found from config in the cleaned HOST dataframe.")
    return num_features, cat_features



# Pipeline 3A: Robustness

def run_pipeline3_host_robustness(
    run_dir: Path,
    severities: Optional[np.ndarray] = None,
    kinds: Tuple[str, ...] = ("packet_loss", "missing_variables"),
    n_repeats: int = 5,
    base_seed: int = 123,
    seed: int = 42,
) -> Dict:
    """
    Robustness evaluation for an existing HOST pipeline1 run.

    - Uses windows.npz saved by pipeline1.
    - Adds per-state_bin metrics if 'state_bin' exists in feature set.
    - If task == 'binary', also computes operational metrics.
    """
    set_global_seed(seed)
    run_dir = Path(run_dir)

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    task = str(config.get("task", "")).lower()

    # Load label encoder
    label_encoder = joblib.load(run_dir / "label_encoder.pkl")

    # Load correct model
    model_type = str(config.get("model_type", "")).lower()
    keras_path = run_dir / "model.keras"
    xgb_path = run_dir / "xgb_model.pkl"

    if model_type in {"tcn", "lstm"}:
        if not keras_path.exists():
            raise FileNotFoundError(f"Expected Keras model at: {keras_path}")
        model = tf.keras.models.load_model(keras_path)
        predict_fn = lambda X: _ensure_proba_2d(model.predict(X, verbose=0))
        op_model = model

    elif model_type == "xgb":
        if not xgb_path.exists():
            candidates = list(run_dir.glob("*.pkl")) + list(run_dir.glob("*.joblib"))
            raise FileNotFoundError(f"Expected XGB model at: {xgb_path}. Found: {[c.name for c in candidates]}")
        model = joblib.load(xgb_path)

        def predict_fn(X):
            Xf = X.reshape(len(X), -1)
            return _ensure_proba_2d(model.predict_proba(Xf))

        op_model = _XGBPredictWrapper(model)

    else:
        # infer by files
        if keras_path.exists():
            model = tf.keras.models.load_model(keras_path)
            predict_fn = lambda X: _ensure_proba_2d(model.predict(X, verbose=0))
            op_model = model
        elif xgb_path.exists():
            model = joblib.load(xgb_path)

            def predict_fn(X):
                Xf = X.reshape(len(X), -1)
                return _ensure_proba_2d(model.predict_proba(Xf))

            op_model = _XGBPredictWrapper(model)
        else:
            raise ValueError(f"Unknown model_type='{model_type}' and no model file found in {run_dir}")

    # Load windows
    npz_path = run_dir / "windows.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found. Re-run HOST pipeline1 so that it saves windows.npz.")
    data = np.load(npz_path)
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    if severities is None:
        severities = np.linspace(0.0, 0.6, 7)

    y_true = np.asarray(y_test)

    # Derive per-window state from state_bin feature if present
    num_features = list(config.get("num_features", []))
    cat_features = list(config.get("cat_features", []))
    feature_cols_all = num_features + cat_features

    has_state_bin = "state_bin" in feature_cols_all
    state_test = None
    states_in_data: List[str] = []
    if has_state_bin and X_test.ndim == 3:
        idx_state_bin = feature_cols_all.index("state_bin")
        state_bin_last = X_test[:, -1, idx_state_bin]
        state_test = np.where(state_bin_last >= 0.5, "charging", "idle")
        states_in_data = ["charging", "idle"]
        print("[robustness_host] Per-state metrics enabled via state_bin (charging/idle).")
    else:
        print("[robustness_host] state_bin not available -> per-state robustness metrics disabled.")

    results: Dict[str, Dict] = {
        "task": task,
        "n_repeats": int(n_repeats),
        "severities": [float(s) for s in severities],
        "kinds": list(kinds),
        "curves": {},
    }

    for kind in kinds:
        print(f"\n=== Robustness (HOST): {kind} ===")
        curves_for_kind: Dict[str, List[float]] = {"severity": []}

        for i_sev, sev in enumerate(severities):
            metric_runs: List[Dict[str, float]] = []

            for r in range(n_repeats):
                rep_seed = base_seed + 1000 * i_sev + r
                rng = np.random.default_rng(rep_seed)

                X_deg = degrade_sequences(
                    X_test,
                    severity=float(sev),
                    kind=kind,
                    rng=rng,
                )

                y_proba = predict_fn(X_deg)
                metrics = compute_multiclass_metrics(y_true, y_proba)

                # per-state metrics
                if state_test is not None:
                    for st in states_in_data:
                        mask_st = (state_test == st)
                        if mask_st.any():
                            m_st = compute_multiclass_metrics(y_true[mask_st], y_proba[mask_st])
                        else:
                            m_st = empty_multiclass_metrics_nan()

                        metrics[f"state_{st}_accuracy"] = float(m_st["accuracy"])
                        metrics[f"state_{st}_balanced_accuracy"] = float(m_st["balanced_accuracy"])
                        metrics[f"state_{st}_pr_auc_micro"] = float(m_st["pr_auc_micro"])
                        metrics[f"state_{st}_pr_auc_macro"] = float(m_st["pr_auc_macro"])

                # binary operational
                if task == "binary":
                    fpr_target = float(config.get("fpr_target", 1e-3))
                    sample_period_seconds = float(config.get("sample_period_seconds", 1.0))
                    step = int(config.get("step", 1))
                    window_hop_seconds = step * sample_period_seconds

                    op_dir = run_dir / "robustness_operational" / kind / f"sev{sev:.2f}" / f"rep{r}"
                    op_dir.mkdir(parents=True, exist_ok=True)

                    op_metrics = evaluate_binary_operational(
                        model=op_model,
                        X_val=X_val,
                        y_val=y_val,
                        X_test=X_deg,
                        y_test=y_test,
                        attack_ids_test=None,
                        label_encoder=label_encoder,
                        fpr_target=fpr_target,
                        output_dir=op_dir,
                        window_hop_seconds=window_hop_seconds,
                    )
                    metrics["op_FPR"] = float(op_metrics["test"]["FPR"])
                    metrics["op_FPH"] = float(op_metrics["test"]["false_positives_per_hour"])
                    metrics["op_TPR"] = float(op_metrics["test"]["TPR"])
                    metrics["op_balanced_accuracy"] = float(op_metrics["test"]["balanced_accuracy"])

                metric_runs.append(metrics)

            agg = aggregate_metric_list(metric_runs)

            print(
                f"  severity={sev:.2f} → "
                f"acc={agg.get('accuracy_mean', float('nan')):.3f}±{agg.get('accuracy_std', float('nan')):.3f}, "
                f"PR-AUC_micro={agg.get('pr_auc_micro_mean', float('nan')):.3f}"
                f"±{agg.get('pr_auc_micro_std', float('nan')):.3f}"
            )

            curves_for_kind["severity"].append(float(sev))
            for k, v in agg.items():
                curves_for_kind.setdefault(k, []).append(v)

        results["curves"][kind] = curves_for_kind

        # Plots (global)
        sev_list = curves_for_kind["severity"]

        if "accuracy_mean" in curves_for_kind:
            plt.figure(figsize=(6, 4))
            plt.errorbar(
                sev_list,
                curves_for_kind["accuracy_mean"],
                yerr=curves_for_kind.get("accuracy_std", None),
                marker="o",
                capsize=4,
            )
            plt.xlabel("Degradation severity")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy vs severity ({kind}) [HOST]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / f"robustness_acc_{kind}.png", dpi=150)
            plt.close()

        if "pr_auc_micro_mean" in curves_for_kind:
            plt.figure(figsize=(6, 4))
            plt.errorbar(
                sev_list,
                curves_for_kind["pr_auc_micro_mean"],
                yerr=curves_for_kind.get("pr_auc_micro_std", None),
                marker="o",
                capsize=4,
            )
            plt.xlabel("Degradation severity")
            plt.ylabel("PR-AUC (micro)")
            plt.title(f"PR-AUC_micro vs severity ({kind}) [HOST]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / f"robustness_prauc_{kind}.png", dpi=150)
            plt.close()

        # Operational plot
        if task == "binary" and "op_FPH_mean" in curves_for_kind:
            plt.figure(figsize=(6, 4))
            plt.errorbar(
                sev_list,
                curves_for_kind["op_FPH_mean"],
                yerr=curves_for_kind.get("op_FPH_std", None),
                marker="o",
                capsize=4,
            )
            plt.xlabel("Degradation severity")
            plt.ylabel("False positives per hour (FPH)")
            plt.title(f"Operational FPH vs severity ({kind}) [HOST]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / f"robustness_fph_{kind}.png", dpi=150)
            plt.close()

        # Per-state plots
        if state_test is not None and "state_charging_accuracy_mean" in curves_for_kind:
            plt.figure(figsize=(6, 4))
            plt.errorbar(
                sev_list,
                curves_for_kind["state_charging_accuracy_mean"],
                yerr=curves_for_kind.get("state_charging_accuracy_std", None),
                marker="o",
                capsize=4,
            )
            plt.xlabel("Degradation severity")
            plt.ylabel("Accuracy (charging)")
            plt.title(f"Accuracy(charging) vs severity ({kind}) [HOST]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / f"robustness_acc_charging_{kind}.png", dpi=150)
            plt.close()

        if state_test is not None and "state_idle_accuracy_mean" in curves_for_kind:
            plt.figure(figsize=(6, 4))
            plt.errorbar(
                sev_list,
                curves_for_kind["state_idle_accuracy_mean"],
                yerr=curves_for_kind.get("state_idle_accuracy_std", None),
                marker="o",
                capsize=4,
            )
            plt.xlabel("Degradation severity")
            plt.ylabel("Accuracy (idle)")
            plt.title(f"Accuracy(idle) vs severity ({kind}) [HOST]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / f"robustness_acc_idle_{kind}.png", dpi=150)
            plt.close()

    out_path = run_dir / "robustness_curves.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[robustness_host] Saved curves (with error bars) to {out_path}")
    return results



# Pipeline 3B: Generalization (leave-one-attack-out)

def run_pipeline3_host_generalization(
    run_dir: Path,
    n_repeats: int = 3,
    holdout_attacks: Optional[List[str]] = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Leave-one-attack-out generalization per ANY HOST pipeline1 run.

    Includes:
      - HOST data cleaning
      - task-specific filtering
      - optional TOP-K re-selection (from config: feature_importances_path + K_top_features)
      - per-state metrics using raw 'State' column (value at last timestep of each window)
      - if task == 'binary': stores operational metrics per run under
            overall_results["metrics"][attack_name]["binary_operational_per_run"].
    """
    set_global_seed(seed)
    run_dir = Path(run_dir)

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    task = str(config.get("task", "")).lower()
    if task not in {"binary", "scenario", "multiattack"}:
        raise ValueError(f"Unexpected task='{task}' in config.json.")

    csv_path = Path(config["csv_path"])
    col_time = str(config.get("col_time", "timestamp"))
    model_type = str(config.get("model_type", "")).lower()
    seq_len = int(config["seq_len"])
    step = int(config["step"])

    print(f"[generalization_host] Loading HOST data from {csv_path}")
    df_host, _event_cols = load_and_clean_host_data(csv_path)

    df_host, target_col = prepare_labels_for_task_host(df_host, task)
    print(f"[generalization_host] Task='{task}', using target_col='{target_col}'")

    # Match pipeline1 task filtering
    if task == "scenario":
        # remove 'none' scenarios
        df_host = df_host[df_host[target_col] != "none"].reset_index(drop=True)
    elif task == "multiattack":
        benign_like = {"none", "benign", "none (ie. benign)"}
        if "Attack" in df_host.columns:
            atk_lower = df_host["Attack"].astype(str).str.lower()
            df_host = df_host[~atk_lower.isin(benign_like)].reset_index(drop=True)

    # Resolve features (topK if specified in config, else config lists)
    num_features, cat_features = _resolve_features_from_config_and_df(config, df_host)
    feature_cols_all = list(num_features) + list(cat_features)

    print(f"[generalization_host] Features used: {len(feature_cols_all)} (num={len(num_features)}, cat={len(cat_features)})")
    if len(feature_cols_all) <= 30:
        print("[generalization_host] feature_cols_all:", feature_cols_all)

    # Base chronological split
    df_train_base, df_val_base, df_test_base, _, feature_cols_used, label_encoder = split_bytime(
        df_host,
        feature_cols=feature_cols_all,
        target_col=target_col,
        train_frac=float(train_frac),
        val_frac=float(val_frac),
        col_time=col_time if "col_time" in split_bytime.__code__.co_varnames else None,  # backward-compat
    ) if "col_time" in split_bytime.__code__.co_varnames else split_bytime(
        df_host,
        feature_cols=feature_cols_all,
        target_col=target_col,
        train_frac=float(train_frac),
        val_frac=float(val_frac),
    )

    print("[generalization_host] Base split shapes:", df_train_base.shape, df_val_base.shape, df_test_base.shape)

    if "State" in df_test_base.columns:
        states_in_data = sorted(df_test_base["State"].fillna("unknown").astype(str).unique().tolist())
    else:
        states_in_data = ["unknown"]

    # Candidate attacks
    attack_candidates = _get_attack_candidates_from_df(df_host)

    if holdout_attacks is None:
        holdout_attacks = attack_candidates
    else:
        holdout_attacks = [a for a in holdout_attacks if a in attack_candidates]
        if not holdout_attacks:
            raise ValueError("holdout_attacks did not match any non-benign 'Attack' values in HOST data.")

    print("[generalization_host] Attacks considered for hold-out:", holdout_attacks)

    seeds = [seed + i for i in range(int(n_repeats))]

    out_dir = run_dir / "generalization"
    out_dir.mkdir(exist_ok=True)

    overall_results: Dict[str, Dict] = {
        "task": task,
        "target_col": target_col,
        "n_repeats": int(n_repeats),
        "holdout_attacks": holdout_attacks,
        "metrics": {},
        "features": {
            "num_features": num_features,
            "cat_features": cat_features,
            "feature_importances_path": config.get("feature_importances_path"),
            "K_top_features": config.get("K_top_features"),
        },
    }

    for attack_name in holdout_attacks:
        print(f"\n=== Generalization (HOST): training WITHOUT Attack='{attack_name}' ===")

        metrics_runs_all: List[Dict[str, float]] = []
        metrics_runs_seen: List[Dict[str, float]] = []
        metrics_runs_heldout: List[Dict[str, float]] = []

        metrics_runs_all_state = {st: [] for st in states_in_data}
        metrics_runs_seen_state = {st: [] for st in states_in_data}
        metrics_runs_heldout_state = {st: [] for st in states_in_data}

        train_times_sec: List[float] = []
        avg_latencies_sec: List[float] = []

        binary_operational_runs: List[Dict] = []

        for r, seed_i in enumerate(seeds):
            print(f"  [run {r + 1}/{n_repeats}] seed={seed_i}")
            set_global_seed(seed_i)

            if "Attack" not in df_train_base.columns:
                raise ValueError("HOST generalization requires an 'Attack' column in the dataframe.")

            df_train = df_train_base[df_train_base["Attack"] != attack_name].reset_index(drop=True)
            df_val = df_val_base[df_val_base["Attack"] != attack_name].reset_index(drop=True)
            df_test = df_test_base.copy()

            attack_test, state_test = _derive_attack_state_per_window(df_test, seq_len=seq_len, step=step)

            X_train, y_train, X_val, y_val, X_test, y_test, scaler = scale_windows(
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                num_features=num_features,
                cat_features=cat_features,
                seq_len=seq_len,
                step=step,
                label_col="y" if "label_col" in scale_windows.__code__.co_varnames else None,
            ) if "label_col" in scale_windows.__code__.co_varnames else scale_windows(
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                num_features=num_features,
                cat_features=cat_features,
                seq_len=seq_len,
                step=step,
            )

            # Train
            if model_type in {"tcn", "lstm"}:
                build_model_fn = build_tcn_model if model_type == "tcn" else build_lstm_model

                train_start = time.perf_counter()
                history, model, class_weights = train_unimodal_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    label_encoder=label_encoder,
                    build_model=build_model_fn,
                    initial_lr=1e-3,
                )
                train_time_sec = time.perf_counter() - train_start
                train_times_sec.append(float(train_time_sec))

                infer_start = time.perf_counter()
                y_proba_test = model.predict(X_test, verbose=0)
                pred_time_sec = time.perf_counter() - infer_start

                avg_latency_sec = float(pred_time_sec) / len(X_test) if len(X_test) > 0 else float("nan")
                avg_latencies_sec.append(avg_latency_sec)

                y_proba_test = _ensure_proba_2d(y_proba_test)

            elif model_type == "xgb":
                X_train_f = X_train.reshape(len(X_train), -1)
                X_val_f = X_val.reshape(len(X_val), -1)
                X_test_f = X_test.reshape(len(X_test), -1)

                model, y_proba_test, train_time_sec, pred_time_sec, class_weights = train_xgb_model(
                    X_train_f, y_train,
                    X_val_f, y_val,
                    X_test_f,
                    label_encoder=label_encoder,
                    use_class_weights=True,
                )
                train_times_sec.append(float(train_time_sec))
                avg_latency_sec = float(pred_time_sec) / len(X_test) if len(X_test) > 0 else float("nan")
                avg_latencies_sec.append(avg_latency_sec)

                y_proba_test = _ensure_proba_2d(y_proba_test)

            else:
                raise ValueError(f"Unsupported model_type='{model_type}'. Use tcn, lstm, or xgb.")

            # Metrics: all
            metrics_all = compute_multiclass_metrics(y_test, y_proba_test)
            metrics_all["training_time_sec"] = float(train_time_sec)
            metrics_all["avg_inference_latency_per_window_sec"] = float(avg_latency_sec)

            # Operational metrics for binary
            if task == "binary":
                fpr_target = float(config.get("fpr_target", 1e-3))
                sample_period_seconds = float(config.get("sample_period_seconds", 1.0))
                window_hop_seconds = step * sample_period_seconds

                op_out_dir = out_dir / "binary_operational" / attack_name / f"seed{seed_i}"
                op_out_dir.mkdir(parents=True, exist_ok=True)

                op_model = model if model_type != "xgb" else _XGBPredictWrapper(model)

                binary_metrics = evaluate_binary_operational(
                    model=op_model,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    attack_ids_test=attack_test,
                    label_encoder=label_encoder,
                    fpr_target=fpr_target,
                    output_dir=op_out_dir,
                    window_hop_seconds=window_hop_seconds,
                )
                binary_operational_runs.append(binary_metrics)

            # seen vs heldout masks
            y_test_attack = attack_test
            mask_heldout = (y_test_attack == attack_name)
            mask_seen = ~mask_heldout

            metrics_seen = compute_multiclass_metrics(y_test[mask_seen], y_proba_test[mask_seen]) if mask_seen.any() else empty_multiclass_metrics_nan()
            metrics_heldout = compute_multiclass_metrics(y_test[mask_heldout], y_proba_test[mask_heldout]) if mask_heldout.any() else empty_multiclass_metrics_nan()

            metrics_runs_all.append(metrics_all)
            metrics_runs_seen.append(metrics_seen)
            metrics_runs_heldout.append(metrics_heldout)

            # per-state splits (from raw State column)
            for st in states_in_data:
                mask_state = (state_test == st)

                metrics_all_st = compute_multiclass_metrics(y_test[mask_state], y_proba_test[mask_state]) if mask_state.any() else empty_multiclass_metrics_nan()
                metrics_runs_all_state[st].append(metrics_all_st)

                mask_seen_st = mask_seen & mask_state
                metrics_seen_st = compute_multiclass_metrics(y_test[mask_seen_st], y_proba_test[mask_seen_st]) if mask_seen_st.any() else empty_multiclass_metrics_nan()
                metrics_runs_seen_state[st].append(metrics_seen_st)

                mask_heldout_st = mask_heldout & mask_state
                metrics_heldout_st = compute_multiclass_metrics(y_test[mask_heldout_st], y_proba_test[mask_heldout_st]) if mask_heldout_st.any() else empty_multiclass_metrics_nan()
                metrics_runs_heldout_state[st].append(metrics_heldout_st)

        # Aggregate
        metrics_all_agg = aggregate_metric_list(metrics_runs_all)
        metrics_seen_agg = aggregate_metric_list(metrics_runs_seen)
        metrics_heldout_agg = aggregate_metric_list(metrics_runs_heldout)

        metrics_by_state = {}
        for st in states_in_data:
            metrics_by_state[st] = {
                "all": aggregate_metric_list(metrics_runs_all_state[st]),
                "seen_attacks": aggregate_metric_list(metrics_runs_seen_state[st]),
                "heldout_attack": aggregate_metric_list(metrics_runs_heldout_state[st]),
            }

        train_times_arr = np.array(train_times_sec, dtype=float)
        lat_arr = np.array(avg_latencies_sec, dtype=float)

        train_time_mean = float(np.nanmean(train_times_arr)) if len(train_times_arr) else float("nan")
        train_time_std = float(np.nanstd(train_times_arr, ddof=1)) if len(train_times_arr) > 1 else 0.0

        lat_mean = float(np.nanmean(lat_arr)) if len(lat_arr) else float("nan")
        lat_std = float(np.nanstd(lat_arr, ddof=1)) if len(lat_arr) > 1 else 0.0

        overall_results["metrics"][attack_name] = {
            "all": metrics_all_agg,
            "seen_attacks": metrics_seen_agg,
            "heldout_attack": metrics_heldout_agg,
            "by_state": metrics_by_state,
            "training_time_sec": {"mean": train_time_mean, "std": train_time_std},
            "avg_inference_latency_per_window_sec": {"mean": lat_mean, "std": lat_std},
        }

        if task == "binary":
            overall_results["metrics"][attack_name]["binary_operational_per_run"] = binary_operational_runs

    out_path = out_dir / "generalization_leave_one_attack_out.json"
    with open(out_path, "w") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\n[generalization_host] Saved results (with error bars) to {out_path}")
    return overall_results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HOST Pipeline 3: robustness & generalization tests.")
    sub = p.add_subparsers(dest="mode", required=True)

    p_r = sub.add_parser("robustness", help="Run robustness curves from an existing run_dir (needs windows.npz).")
    p_r.add_argument("--run_dir", type=str, required=True)
    p_r.add_argument("--kinds", type=str, nargs="+", default=["packet_loss", "missing_variables"])
    p_r.add_argument("--n_repeats", type=int, default=5)
    p_r.add_argument("--seed", type=int, default=42)
    p_r.add_argument("--base_seed", type=int, default=123)
    p_r.add_argument("--sev_min", type=float, default=0.0)
    p_r.add_argument("--sev_max", type=float, default=0.6)
    p_r.add_argument("--sev_n", type=int, default=7)

    p_g = sub.add_parser("generalization", help="Leave-one-attack-out generalization from an existing run_dir.")
    p_g.add_argument("--run_dir", type=str, required=True)
    p_g.add_argument("--n_repeats", type=int, default=3)
    p_g.add_argument("--seed", type=int, default=42)
    p_g.add_argument("--train_frac", type=float, default=0.7)
    p_g.add_argument("--val_frac", type=float, default=0.15)
    p_g.add_argument("--holdout_attacks", type=str, nargs="*", default=None)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "robustness":
        severities = np.linspace(float(args.sev_min), float(args.sev_max), int(args.sev_n))
        run_pipeline3_host_robustness(
            run_dir=Path(args.run_dir),
            severities=severities,
            kinds=tuple(args.kinds),
            n_repeats=int(args.n_repeats),
            base_seed=int(args.base_seed),
            seed=int(args.seed),
        )
    elif args.mode == "generalization":
        run_pipeline3_host_generalization(
            run_dir=Path(args.run_dir),
            n_repeats=int(args.n_repeats),
            holdout_attacks=args.holdout_attacks,
            train_frac=float(args.train_frac),
            val_frac=float(args.val_frac),
            seed=int(args.seed),
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
