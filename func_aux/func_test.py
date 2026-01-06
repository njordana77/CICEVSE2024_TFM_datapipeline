# func_test.py

# !/usr/bin/env python

"""
Evaluation and testing utilities for classification models.

This module provides:
- Helpers to evaluate Keras models on test data and print reports.
- Computation of multiclass metrics and PR/ROC AUC from probabilities.
- Binary-operational evaluation at a target FPR, including TTD statistics.
- Utilities to aggregate metrics across runs and degrade sequences synthetically.
- Sample-size calculators for FPR/FPH guarantees under zero-FP assumptions.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf, sqrt, log

from pathlib import Path
from typing import List, Dict, Tuple, Optional

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from scipy.stats import beta


def test_model(model, X_test, y_test, label_encoder):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("TEST:", test_loss, test_acc)

    y_proba = model.predict(X_test)
    y_pred = y_proba.argmax(axis=1)

    print(
        classification_report(
            y_test, y_pred, target_names=label_encoder.classes_
        )
    )

    plot_confusion_matrix(y_test, y_pred, label_encoder)


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n_classes = y_proba.shape[1]

    # discrete predictions
    y_pred = np.argmax(y_proba, axis=1)

    # basic accuracies
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # one-hot labels for PR/ROC AUC
    y_true_onehot = np.eye(n_classes)[y_true]

    # PR–AUC
    pr_micro = average_precision_score(
        y_true_onehot,
        y_proba,
        average="micro",
    )
    pr_macro = average_precision_score(
        y_true_onehot,
        y_proba,
        average="macro",
    )

    # ROC–AUC (macro)
    try:
        roc_macro = roc_auc_score(
            y_true_onehot,
            y_proba,
            average="macro",
        )
    except ValueError:
        # happens if only one class present, etc.
        roc_macro = np.nan

    # Precision / recall / F1
    prec_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    rec_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    prec_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    rec_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # F2 and F0.5
    f2_macro = fbeta_score(
        y_true,
        y_pred,
        beta=2.0,
        average="macro",
        zero_division=0,
    )
    f0_5_macro = fbeta_score(
        y_true,
        y_pred,
        beta=0.5,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "pr_auc_micro": float(pr_micro),
        "pr_auc_macro": float(pr_macro),
        "roc_auc_macro": float(roc_macro),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "f2_macro": float(f2_macro),
        "f0_5_macro": float(f0_5_macro),
    }


# ----------------------------------------------------------------------
# Binary operational metrics
# ----------------------------------------------------------------------


def clopper_pearson_ci(fp: int, n_neg: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n_neg == 0:
        return np.nan, np.nan

    if fp == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2.0, fp, n_neg - fp + 1)

    if fp == n_neg:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2.0, fp + 1, n_neg - fp)

    return float(lower), float(upper)


def compute_ttd_stats(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    window_hop_seconds: float,
    episode_ids: Optional[np.ndarray] = None,
) -> Dict:
    n = len(y_true_binary)
    episodes_total = 0
    episodes_detected = 0
    delays_sec = []

    i = 0
    while i < n:
        if y_true_binary[i] == 1:
            episodes_total += 1
            current_ep = episode_ids[i] if episode_ids is not None else None
            start = i
            j = i + 1

            while j < n and y_true_binary[j] == 1:
                if episode_ids is not None and episode_ids[j] != current_ep:
                    break
                j += 1
            end = j

            det_idx = None
            for t in range(start, end):
                if y_pred_binary[t] == 1:
                    det_idx = t
                    break

            if det_idx is not None:
                episodes_detected += 1
                delay_windows = det_idx - start
                delay_seconds = delay_windows * window_hop_seconds
                delays_sec.append(delay_seconds)

            i = end
        else:
            i += 1

    if delays_sec:
        avg_ttd_seconds = float(np.mean(delays_sec))
        avg_ttd_hours = avg_ttd_seconds / 3600.0
    else:
        avg_ttd_seconds = np.nan
        avg_ttd_hours = np.nan

    return {
        "episodes_total": int(episodes_total),
        "episodes_detected": int(episodes_detected),
        "avg_ttd_seconds": avg_ttd_seconds,
        "avg_ttd_hours": avg_ttd_hours,
    }


def select_threshold_at_fpr(
    y_true_binary: np.ndarray,
    y_score_attack: np.ndarray,
    fpr_target: float,
) -> Tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(y_true_binary, y_score_attack)

    idx_candidates = np.where(fpr <= fpr_target)[0]
    if len(idx_candidates) > 0:
        idx = idx_candidates[-1]
    else:
        idx = int(np.argmin(np.abs(fpr - fpr_target)))

    threshold = thr[idx]
    fpr_at_thr = fpr[idx]
    tpr_at_thr = tpr[idx]

    return float(threshold), float(fpr_at_thr), float(tpr_at_thr)


def evaluate_binary_operational(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    attack_ids_test: Optional[np.ndarray],  # IDs d'episodi per finestra (Attack)
    label_encoder,
    fpr_target: float,
    output_dir: Path,
    window_hop_seconds: float,
) -> Dict:
    class_names = list(label_encoder.classes_)
    if "attack" not in class_names or "benign" not in class_names:
        raise ValueError(
            f"Binary task expects classes ['attack', 'benign'], got: {class_names}"
        )

    idx_attack = class_names.index("attack")

    is_xgb = hasattr(model, "predict_proba")

    # -------- Validation probabilities --------
    if is_xgb:
        X_val_in = X_val.reshape(X_val.shape[0], -1)
        y_val_proba = model.predict_proba(X_val_in)
    else:
        y_val_proba = model.predict(X_val, verbose=0)

    y_val_labels = label_encoder.inverse_transform(y_val)
    y_val_binary = (y_val_labels == "attack").astype(int)
    p_attack_val = y_val_proba[:, idx_attack]

    # ROC–AUC on validation
    roc_auc_val = roc_auc_score(y_val_binary, p_attack_val)

    thr, fpr_val, tpr_val = select_threshold_at_fpr(
        y_val_binary, p_attack_val, fpr_target=fpr_target
    )
    #print(
    #    f"[binary] Threshold on VAL for FPR_target={fpr_target:.4f}: "
    #    f"thr={thr:.4f}, FPR_val={fpr_val:.4g}, TPR_val={tpr_val:.4g}"
    #)

    # -------- Test probabilities & latency --------
    start_time = time.perf_counter()
    if is_xgb:
        X_test_in = X_test.reshape(X_test.shape[0], -1)
        y_test_proba = model.predict_proba(X_test_in)
    else:
        y_test_proba = model.predict(X_test, verbose=0)
    end_time = time.perf_counter()
    total_inference_time = end_time - start_time
    avg_latency_per_window = (
        total_inference_time / len(X_test) if len(X_test) > 0 else np.nan
    )

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_test_binary = (y_test_labels == "attack").astype(int)
    p_attack_test = y_test_proba[:, idx_attack]
    y_test_pred_bin = (p_attack_test >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test_binary, y_test_pred_bin, labels=[0, 1]
    ).ravel()

    # ROC–AUC on test
    roc_auc_test = roc_auc_score(y_test_binary, p_attack_test)

    # F1 / F2 / F0.5 on test (binary preds at chosen threshold)
    f1_test = f1_score(y_test_binary, y_test_pred_bin, zero_division=0)
    f2_test = fbeta_score(
        y_test_binary,
        y_test_pred_bin,
        beta=2.0,
        zero_division=0,
    )
    f0_5_test = fbeta_score(
        y_test_binary,
        y_test_pred_bin,
        beta=0.5,
        zero_division=0,
    )

    n_neg = tn + fp
    fpr_test = fp / n_neg if n_neg > 0 else np.nan
    tpr_test = tp / (tp + fn + 1e-12)
    precision_test = tp / (tp + fp + 1e-12)
    recall_test = tpr_test
    tnr_test = tn / (tn + fp + 1e-12) if (tn + fp) > 0 else np.nan
    balanced_acc_test = (
        0.5 * (tpr_test + tnr_test) if not np.isnan(tnr_test) else np.nan
    )

    alpha = 0.05  # two sided dist.: 95% CL (nearly 2sigma)
    z = 1.96
    if n_neg > 0:
        se = np.sqrt(fpr_test * (1 - fpr_test) / n_neg)
        ci_low_norm = max(0.0, fpr_test - z * se)
        ci_high_norm = min(1.0, fpr_test + z * se)
    else:
        ci_low_norm, ci_high_norm = np.nan, np.nan

    ci_low_cp, ci_high_cp = clopper_pearson_ci(fp, n_neg, alpha=alpha)

    # FP per hour of benign operation
    total_hours_benign = n_neg * window_hop_seconds / 3600.0
    if total_hours_benign > 0:
        fp_per_hour = fp / total_hours_benign
    else:
        fp_per_hour = np.nan

    avg_fp_per_hour = fp_per_hour


    if window_hop_seconds > 0:
        fpr_to_fph = 3600.0 / window_hop_seconds
    else:
        fpr_to_fph = np.nan


    if not np.isnan(fpr_to_fph):
        fph_low_norm = ci_low_norm * fpr_to_fph
        fph_high_norm = ci_high_norm * fpr_to_fph
        fph_low_cp = ci_low_cp * fpr_to_fph
        fph_high_cp = ci_high_cp * fpr_to_fph
    else:
        fph_low_norm = fph_high_norm = np.nan
        fph_low_cp = fph_high_cp = np.nan

    # Also keep total hours of the full test set
    total_hours_test = len(y_test_binary) * window_hop_seconds / 3600.0

    if attack_ids_test is not None:
        atk_series = pd.Series(attack_ids_test).fillna("none").astype(str).str.lower()
        is_benign = atk_series.isin(["none", "benign", "none (ie. benign)"])


        y_true_binary_ttd = (~is_benign).astype(int).values

        episode_ids = atk_series.where(~is_benign, other="benign").values

        ttd_stats = compute_ttd_stats(
            y_true_binary=y_true_binary_ttd,
            y_pred_binary=y_test_pred_bin,
            window_hop_seconds=window_hop_seconds,
            episode_ids=episode_ids,
        )
    else:

        ttd_stats = compute_ttd_stats(
            y_true_binary=y_test_binary,
            y_pred_binary=y_test_pred_bin,
            window_hop_seconds=window_hop_seconds,
            episode_ids=None,
        )

    # ROC curves
    fpr_val_curve, tpr_val_curve, _ = roc_curve(y_val_binary, p_attack_val)
    fpr_test_curve, tpr_test_curve, _ = roc_curve(y_test_binary, p_attack_test)

    plt.figure()
    plt.plot(fpr_val_curve, tpr_val_curve, label="Validation ROC", color="red")
    plt.plot(fpr_test_curve, tpr_test_curve, label="Test ROC", color="grey")
    plt.scatter([fpr_val], [tpr_val], label="Selected thr (val)", marker="o")
    plt.xscale("log")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Binary ROC: attack vs benign")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_binary_val_test.png", dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test_binary, p_attack_test)
    ap = average_precision_score(y_test_binary, p_attack_test)

    plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Binary PR curve (test, AP={ap:.3f})")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_binary_test.png", dpi=150)
    plt.close()

    curves = {
        "roc_val": {
            "fpr": fpr_val_curve.tolist(),
            "tpr": tpr_val_curve.tolist(),
        },
        "roc_test": {
            "fpr": fpr_test_curve.tolist(),
            "tpr": tpr_test_curve.tolist(),
        },
        "pr_test": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        },
    }

    return {
        "threshold": thr,
        "fpr_target": fpr_target,
        "validation": {
            "FPR": fpr_val,
            "TPR": tpr_val,
            "ROC_AUC": float(roc_auc_val),
        },
        "test": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "FPR": float(fpr_test),
            "TPR": float(tpr_test),
            "TNR": float(tnr_test),
            "precision": float(precision_test),
            "recall": float(recall_test),
            "balanced_accuracy": float(balanced_acc_test),
            "ROC_AUC": float(roc_auc_test),
            "PR_AUC": float(ap),  # average precision
            "F1": float(f1_test),
            "F2": float(f2_test),
            "F0_5": float(f0_5_test),
            "FPR_CI_normal": [float(ci_low_norm), float(ci_high_norm)],
            "FPR_CI_clopper_pearson": [float(ci_low_cp), float(ci_high_cp)],
            "FPH_CI_normal": [float(fph_low_norm), float(fph_high_norm)],
            "FPH_CI_clopper_pearson": [float(fph_low_cp), float(fph_high_cp)],
            "false_positives_per_hour": float(fp_per_hour),
            "average_FP_per_hour": float(avg_fp_per_hour),
            "avg_inference_latency_per_window_sec": float(avg_latency_per_window),
            "TTD_stats": ttd_stats,
            "total_hours_test": float(total_hours_test),
        },
        "curves": curves,
    }


def aggregate_metric_list(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}

    keys = metric_list[0].keys()
    out: Dict[str, float] = {}

    for k in keys:
        values = np.array([m[k] for m in metric_list], dtype=float)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        out[f"{k}_mean"] = mean
        out[f"{k}_std"] = std

    return out


# ---------------------------------------------------------------------
# Degradation operators (packet_loss, missing_variables, etc.)
# ---------------------------------------------------------------------


def degrade_sequences(
    X: np.ndarray,
    severity: float,
    kind: str = "packet_loss",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    X_deg = X.copy()
    rng = np.random.default_rng() if rng is None else rng
    n_samples, T, F = X_deg.shape

    if severity <= 0:
        return X_deg

    if kind == "packet_loss":
        # Proportion of time steps set to 0 (all features)
        mask = rng.random((n_samples, T)) < severity
        mask = np.repeat(mask[:, :, None], F, axis=2)
        X_deg[mask] = 0.0

    elif kind == "missing_variables":
        # Proportion of features set to 0 (across all time)
        mask = rng.random((n_samples, F)) < severity
        mask = np.repeat(mask[:, None, :], T, axis=1)
        X_deg[mask] = 0.0

    elif kind == "clock_offset":
        # Circular time shift per sample
        max_shift = int(severity * T)
        if max_shift > 0:
            shifts = rng.integers(-max_shift, max_shift + 1, size=n_samples)
            for i in range(n_samples):
                X_deg[i] = np.roll(X_deg[i], shift=shifts[i], axis=0)

    elif kind == "drift":
        # Linear drift over time per feature
        t = np.linspace(0.0, 1.0, T)
        for i in range(n_samples):
            coef = rng.normal(0.0, severity, size=(F,))
            drift = t[:, None] * coef[None, :]
            X_deg[i] += drift

    else:
        raise ValueError(f"Unknown degradation kind: '{kind}'")

    return X_deg


def required_negatives_for_sigma_fpr(
    fpr_max: float,
    sigma: float = 3.0,
    use_approx: bool = True,
) -> float:
    """
    Nombre mínim de finestres benignes (n_neg) necessàries per poder afirmar,
    amb un nivell de confiança corresponent a 'sigma', que FPR <= fpr_max,
    assumint FP = 0 (cas de disseny més desfavorable).

    Si use_approx=True, fa servir n ≈ ln(1/alpha)/fpr_max.
    Si use_approx=False, resol la fórmula exacta amb Clopper–Pearson invertit.
    """
    if fpr_max <= 0:
        return np.inf

    cl = erf(sigma / sqrt(2.0)) 
    alpha = 1.0 - cl

    if use_approx:
        return log(1.0 / alpha) / fpr_max

    return log(alpha) / log(1.0 - fpr_max)

def required_hours_for_sigma_fph(
    fph_max: float,
    sigma: float = 3.0,
    use_approx: bool = True,
) -> float:
    """
    Hores de funcionament benigne necessàries per poder afirmar
    (a un nivell 'sigma') que FPH <= fph_max, assumint FP = 0.

    Torna T_benign_hours.
    """
    if fph_max <= 0:
        return np.inf

    cl = erf(sigma / sqrt(2.0))
    alpha = 1.0 - cl

    if use_approx:
        return log(1.0 / alpha) / fph_max

    return log(1.0 / alpha) / fph_max


def required_negatives_for_sigma_fph(
    fph_max: float,
    window_hop_seconds: float,
    sigma: float = 3.0,
    use_approx: bool = True,
) -> float:
    """
    Nombre de finestres benignes necessàries per aconseguir un límit
    FPH <= fph_max a 'sigma' (FP=0).
    """
    T_hours = required_hours_for_sigma_fph(
        fph_max=fph_max,
        sigma=sigma,
        use_approx=use_approx,
    )
    if window_hop_seconds <= 0:
        return np.inf

    return T_hours * 3600.0 / window_hop_seconds
