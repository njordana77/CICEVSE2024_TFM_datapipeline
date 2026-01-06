# pipeline_tests_power.py
# !/usr/bin/env python

"""
Pipeline 3: robustness & generalization for EVSE-B power models.

Assumes you already have runs from pipeline1 stored in a directory like:

    outputs_pipeline1/binary_tcn_seq15_step1/
    outputs_pipeline1/scenario_tcn_seq15_step1/
    outputs_pipeline1/multiattack_lstm_seq15_step1/

with at least:
    - model.keras
    - label_encoder.pkl
    - config.json

For robustness, you also need:
    - windows.npz   (X_train, X_val, X_test, y_train, y_val, y_test)
      saved by pipeline1.

High-level functions:

  1) run_pipeline3_robustness(...)
       - uses the saved model + X_test, y_test
       - applies degradations (packet_loss, missing_variables, etc.)
       - runs several repetitions per severity (for error bars)
       - saves JSON with mean/std metrics per severity
       - if task == 'binary': also computes operational metrics (FPR-target threshold, FPH, etc.)
       - NEW: also computes metrics per state_bin (charging/idle), based on last timestep of each window.

  2) run_pipeline3_generalization(...)
       - works for task = 'binary', 'scenario', or 'multiattack'
       - reloads the raw CSV and split logic
       - for each Attack name, trains models WITHOUT that Attack in train/val
       - evaluates performance on all / seen attacks / held-out attack
       - aggregates mean/std over several seeds (for error bars)
       - if task == 'binary': also computes operational metrics per run (saved nested)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from func_aux.func_preproc import (
    set_global_seed,
    load_power_data,
    split_bytime,
    scale_windows,
    prepare_labels_for_task,
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


def _ensure_proba_2d(y_proba: np.ndarray) -> np.ndarray:
    """Ensure predict_proba output is (n, n_classes)."""
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        # binary probability for class 1
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


def _get_attack_candidates_from_df(df_power) -> List[str]:
    """
    Return candidate non-benign attacks from raw df_power['Attack'].
    Adjust benign-like set if needed.
    """
    if "Attack" not in df_power.columns:
        return []
    benign_like = {"none", "benign", "none (ie. benign)"}
    attacks = (
        df_power["Attack"]
        .fillna("none")
        .astype(str)
        .str.strip()
    )
    attacks_lower = attacks.str.lower()
    out = sorted(attacks[~attacks_lower.isin(benign_like)].unique().tolist())
    return out


def _derive_attack_state_per_window(
    df_test,
    seq_len: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive per-window Attack and State labels by taking the value at the LAST timestep
    of each window. This matches common window labeling logic.

    Returns:
        attack_test: (n_windows,) array of Attack strings
        state_test : (n_windows,) array of State strings
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



# Pipeline 3A: Robustness under packet_loss / missing_variables

def run_pipeline3_robustness(
    run_dir: Path,
    severities: Optional[np.ndarray] = None,
    kinds: Tuple[str, ...] = ("packet_loss", "missing_variables"),
    n_repeats: int = 5,
    base_seed: int = 123,
    seed: int = 42,
) -> Dict:
    """
    Robustness evaluation for an existing run from pipeline1.

    NEW:
      - If task == 'binary': adds operational metrics (op_FPR/op_FPH/op_TPR/op_balanced_accuracy)
      - Adds metrics per state_bin (charging/idle), derived from X_test (last timestep state_bin)
    """
    set_global_seed(seed)
    run_dir = Path(run_dir)

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    task = config.get("task", "")

    # Load model and artifacts
    print(f"[robustness] Loading model from {run_dir}")
    model_type = str(config.get("model_type", "")).lower()

    label_encoder = joblib.load(run_dir / "label_encoder.pkl")

    # Load correct model depending on type / files
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
            # fallback: in case saved with another name
            candidates = list(run_dir.glob("*.pkl")) + list(run_dir.glob("*.joblib"))
            raise FileNotFoundError(f"Expected XGB model at: {xgb_path}. Found: {[c.name for c in candidates]}")
        model = joblib.load(xgb_path)

        # predict_proba with flattened windows
        def predict_fn(X):
            Xf = X.reshape(len(X), -1)
            return _ensure_proba_2d(model.predict_proba(Xf))

        # wrapper so evaluate_binary_operational can call .predict(X, verbose=0)
        op_model = _XGBPredictWrapper(model)

    else:
        # infer by file existence
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
        raise FileNotFoundError(
            f"{npz_path} not found. Re-run pipeline1 so that it saves windows.npz."
        )
    data = np.load(npz_path)
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    if severities is None:
        severities = np.linspace(0.0, 0.6, 7)

    y_true = np.asarray(y_test)

    # Derive per-window state from state_bin feature (if present)
    num_features = config.get("num_features", ["current_mA", "bus_voltage_V", "power_mW"])
    cat_features = config.get("cat_features", ["state_bin"])
    feature_cols_all = list(num_features) + list(cat_features)

    has_state_bin = "state_bin" in feature_cols_all
    state_test = None
    states_in_data: List[str] = []
    idx_state_bin = None

    if has_state_bin and X_test.ndim == 3:
        idx_state_bin = feature_cols_all.index("state_bin")
        state_bin_last = X_test[:, -1, idx_state_bin]
        # convention: >= 0.5 -> charging, else idle
        state_test = np.where(state_bin_last >= 0.5, "charging", "idle")
        states_in_data = ["charging", "idle"]
        print("[robustness] Per-state metrics enabled via state_bin (charging/idle).")
    else:
        print("[robustness] state_bin not available -> per-state robustness metrics disabled.")

    results: Dict[str, Dict] = {
        "task": task,
        "n_repeats": int(n_repeats),
        "severities": [float(s) for s in severities],
        "kinds": list(kinds),
        "curves": {},
    }

    for kind in kinds:
        print(f"\n=== Robustness: {kind} ===")
        curves_for_kind: Dict[str, List[float]] = {"severity": []}

        for i_sev, sev in enumerate(severities):
            metric_runs: List[Dict[str, float]] = []

            for r in range(n_repeats):
                rep_seed = base_seed + 1000 * i_sev + r
                rng = np.random.default_rng(rep_seed)

                # degrade X_test
                X_deg = degrade_sequences(
                    X_test,
                    severity=float(sev),
                    kind=kind,
                    rng=rng,
                )

                # predict on degraded windows
                y_proba = predict_fn(X_deg)


                metrics = compute_multiclass_metrics(y_true, y_proba)


                # Per-state metrics
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


                # Binary operational metrics
                if task == "binary":
                    fpr_target = float(config.get("fpr_target", 1e-3))
                    sample_period_seconds = float(config.get("sample_period_seconds", 1.0))
                    step = int(config.get("step", 1))
                    window_hop_seconds = step * sample_period_seconds

                    op_dir = (
                        run_dir
                        / "robustness_operational"
                        / kind
                        / f"sev{sev:.2f}"
                        / f"rep{r}"
                    )
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


        sev_list = curves_for_kind["severity"]

        # Accuracy vs severity
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
            plt.title(f"Accuracy vs severity ({kind})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / f"robustness_acc_{kind}.png", dpi=150)
            plt.close()

        # PR-AUC_micro vs severity
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
            plt.title(f"PR-AUC_micro vs severity ({kind})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / f"robustness_prauc_{kind}.png", dpi=150)
            plt.close()


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
            plt.title(f"Operational FPH vs severity ({kind})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / f"robustness_fph_{kind}.png", dpi=150)
            plt.close()

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
            plt.title(f"Accuracy(charging) vs severity ({kind})")
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
            plt.title(f"Accuracy(idle) vs severity ({kind})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(run_dir / f"robustness_acc_idle_{kind}.png", dpi=150)
            plt.close()

    out_path = run_dir / "robustness_curves.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[robustness] Saved curves (with error bars) to {out_path}")
    return results



# Pipeline 3B: Generalization (leave-one-attack-out) for any task


def run_pipeline3_generalization(
    run_dir: Path,
    n_repeats: int = 3,
    holdout_attacks: Optional[List[str]] = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Leave-one-attack-out generalization per ANY pipeline1 run.

    If task == 'binary', also stores per-run operational metrics (nested dict)
    under:
        overall_results["metrics"][attack_name]["binary_operational_per_run"].
    """
    set_global_seed(seed)
    run_dir = Path(run_dir)

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    task = config.get("task")
    if task not in {"binary", "scenario", "multiattack"}:
        raise ValueError(
            f"Unexpected task='{task}' in config.json. "
            "Expected 'binary', 'scenario' or 'multiattack'."
        )

    csv_path = Path(config["csv_path"])
    col_time = config["col_time"]
    model_type = config["model_type"]
    seq_len = int(config["seq_len"])
    step = int(config["step"])

    # Features used in pipeline1
    num_features = config.get(
        "num_features",
        ["current_mA", "bus_voltage_V", "power_mW"],
    )
    cat_features = config.get("cat_features", ["state_bin"])
    feature_cols_all = num_features + cat_features

    # Reload raw data and re-create labels for this task
    print(f"[generalization] Loading power data from {csv_path}")
    df_power = load_power_data(str(csv_path), col_time=col_time)

    df_power, target_col = prepare_labels_for_task(df_power, task)
    print(f"[generalization] Task='{task}', using target_col='{target_col}'")

    # Base chronological split by (Attack, State)
    df_train_base, df_val_base, df_test_base, _, feature_cols_used, label_encoder = (
        split_bytime(
            df_power,
            feature_cols=feature_cols_all,
            target_col=target_col,
            train_frac=train_frac,
            val_frac=val_frac,
        )
    )

    print(
        "[generalization] Base split shapes:",
        df_train_base.shape,
        df_val_base.shape,
        df_test_base.shape,
    )

    # States existing in test (from raw df State column)
    states_in_data = sorted(df_test_base["State"].dropna().unique())

    # Which Attack values we can hold out
    attack_candidates = _get_attack_candidates_from_df(df_power)

    if holdout_attacks is None:
        holdout_attacks = attack_candidates
    else:
        holdout_attacks = [a for a in holdout_attacks if a in attack_candidates]
        if not holdout_attacks:
            raise ValueError(
                "holdout_attacks did not match any non-benign 'Attack' values."
            )

    print("[generalization] Attacks considered for hold-out:", holdout_attacks)

    # Seeds to repeat training
    seeds = [seed + i for i in range(n_repeats)]

    out_dir = run_dir / "generalization"
    out_dir.mkdir(exist_ok=True)

    overall_results: Dict[str, Dict] = {
        "task": task,
        "target_col": target_col,
        "n_repeats": int(n_repeats),
        "holdout_attacks": holdout_attacks,
        "metrics": {},
    }

    for attack_name in holdout_attacks:
        print(f"\n=== Generalization: training WITHOUT Attack='{attack_name}' ===")

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

            df_train = df_train_base[df_train_base["Attack"] != attack_name].reset_index(drop=True)
            df_val = df_val_base[df_val_base["Attack"] != attack_name].reset_index(drop=True)
            df_test = df_test_base.copy()

            attack_test, state_test = _derive_attack_state_per_window(
                df_test,
                seq_len=seq_len,
                step=step,
            )

            X_train, y_train, X_val, y_val, X_test, y_test, scaler = scale_windows(
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                num_features=num_features,
                cat_features=cat_features,
                seq_len=seq_len,
                step=step,
            )

            model_type = str(model_type).lower()

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
                train_end = time.perf_counter()
                train_time_sec = train_end - train_start
                train_times_sec.append(train_time_sec)

                infer_start = time.perf_counter()
                y_proba_test = model.predict(X_test, verbose=0)
                infer_end = time.perf_counter()
                total_infer_time = infer_end - infer_start
                avg_latency_sec = total_infer_time / len(X_test) if len(X_test) > 0 else np.nan
                avg_latencies_sec.append(avg_latency_sec)

                y_proba_test = _ensure_proba_2d(y_proba_test)

            elif model_type == "xgb":
                # flatten windows for XGB
                X_train_f = X_train.reshape(len(X_train), -1)
                X_val_f   = X_val.reshape(len(X_val), -1)
                X_test_f  = X_test.reshape(len(X_test), -1)

                model, y_proba_test, train_time_sec, pred_time_sec, class_weights = train_xgb_model(
                    X_train_f, y_train,
                    X_val_f, y_val,
                    X_test_f,
                    label_encoder=label_encoder,
                    use_class_weights=True,
                )
                train_times_sec.append(float(train_time_sec))
                avg_latency_sec = float(pred_time_sec) / len(X_test) if len(X_test) > 0 else np.nan
                avg_latencies_sec.append(avg_latency_sec)

                y_proba_test = _ensure_proba_2d(y_proba_test)

            else:
                raise ValueError(f"Unsupported model_type='{model_type}'. Use tcn, lstm, or xgb.")


            metrics_all = compute_multiclass_metrics(y_test, y_proba_test)
            metrics_all["training_time_sec"] = float(train_time_sec)
            metrics_all["avg_inference_latency_per_window_sec"] = float(avg_latency_sec)

            # Operational metric only for binary
            if task == "binary":
                fpr_target = float(config.get("fpr_target", 1e-3))
                sample_period_seconds = float(config.get("sample_period_seconds", 1.0))
                window_hop_seconds = step * sample_period_seconds

                op_out_dir = out_dir / "binary_operational" / attack_name / f"seed{seed_i}"
                op_out_dir.mkdir(parents=True, exist_ok=True)

                op_model = model
                if model_type == "xgb":
                    op_model = _XGBPredictWrapper(model)

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

            # heldout vs seen masks (per window)
            y_test_attack = attack_test
            mask_heldout = (y_test_attack == attack_name)
            mask_seen = ~mask_heldout

            if mask_seen.any():
                metrics_seen = compute_multiclass_metrics(y_test[mask_seen], y_proba_test[mask_seen])
            else:
                metrics_seen = empty_multiclass_metrics_nan()

            if mask_heldout.any():
                metrics_heldout = compute_multiclass_metrics(y_test[mask_heldout], y_proba_test[mask_heldout])
            else:
                metrics_heldout = empty_multiclass_metrics_nan()

            metrics_runs_all.append(metrics_all)
            metrics_runs_seen.append(metrics_seen)
            metrics_runs_heldout.append(metrics_heldout)

            # per-state splits (based on raw State column)
            for st in states_in_data:
                mask_state = (state_test == st)

                if mask_state.any():
                    metrics_all_st = compute_multiclass_metrics(y_test[mask_state], y_proba_test[mask_state])
                else:
                    metrics_all_st = empty_multiclass_metrics_nan()
                metrics_runs_all_state[st].append(metrics_all_st)

                mask_seen_st = mask_seen & mask_state
                if mask_seen_st.any():
                    metrics_seen_st = compute_multiclass_metrics(y_test[mask_seen_st], y_proba_test[mask_seen_st])
                else:
                    metrics_seen_st = empty_multiclass_metrics_nan()
                metrics_runs_seen_state[st].append(metrics_seen_st)

                mask_heldout_st = mask_heldout & mask_state
                if mask_heldout_st.any():
                    metrics_heldout_st = compute_multiclass_metrics(y_test[mask_heldout_st], y_proba_test[mask_heldout_st])
                else:
                    metrics_heldout_st = empty_multiclass_metrics_nan()
                metrics_runs_heldout_state[st].append(metrics_heldout_st)

        # aggregate over repeats
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

        train_times_sec_arr = np.array(train_times_sec, dtype=float)
        avg_latencies_sec_arr = np.array(avg_latencies_sec, dtype=float)
        train_time_mean = float(np.nanmean(train_times_sec_arr))
        train_time_std = float(np.nanstd(train_times_sec_arr, ddof=1)) if len(train_times_sec_arr) > 1 else 0.0
        latency_mean = float(np.nanmean(avg_latencies_sec_arr))
        latency_std = float(np.nanstd(avg_latencies_sec_arr, ddof=1)) if len(avg_latencies_sec_arr) > 1 else 0.0

        overall_results["metrics"][attack_name] = {
            "all": metrics_all_agg,
            "seen_attacks": metrics_seen_agg,
            "heldout_attack": metrics_heldout_agg,
            "by_state": metrics_by_state,
            "training_time_sec": {"mean": train_time_mean, "std": train_time_std},
            "avg_inference_latency_per_window_sec": {"mean": latency_mean, "std": latency_std},
        }

        if task == "binary":
            overall_results["metrics"][attack_name]["binary_operational_per_run"] = binary_operational_runs

    out_path = out_dir / "generalization_leave_one_attack_out.json"
    with open(out_path, "w") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\n[generalization] Saved results (with error bars) to {out_path}")
    return overall_results
