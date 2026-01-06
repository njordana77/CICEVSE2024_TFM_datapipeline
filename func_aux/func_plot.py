# func_plot.py

# !/usr/bin/env python


"""
Plotting and results-aggregation utilities for the EVSE attack-detection experiments.

Includes:
- ACF tools: pure-NumPy autocorrelation (compute_acf) + plotting to file (plot_acf).
- Exploratory power plots by state:
  - 2x2 figure (timeseries + boxplots) split into charging vs idle, colored by Attack and labeled by Attack-Group.
  - Split-distribution bar charts across train/val/test per (State, Attack).
- Model evaluation visuals:
  - Confusion matrix plotting (interactive) and saving (PNG + JSON), including helper to plot from a precomputed matrix.
  - Training history plotting/saving (loss + accuracy) as PNGs, plus JSON export of history.
- Metrics aggregation across runs:
  - Recursively load every metrics.json under an output root, merge with config.json, and flatten nested metrics into columns.
  - Build “metric triplets” (ALL / charging / idle) and generate comparable plots across model types, seq_len, and step.
  - Robustness/generalization plotting helpers using mean/std curves vs severity or leave-one-attack-out subsets.

Main entry points: plot_power_two_states_with_legend, plot_split_distribution,
plot_training_history / save_training_history, save_confusion_matrix,
load_all_metrics_flat, plot_all_triplets_for_tasks, plot_robustness_all_charge_idle,
plot_all_confusion_matrices_for_tasks.
"""


import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any

import json
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


def compute_acf(x, max_lag):
    """
    Compute autocorrelation function up to max_lag using only NumPy.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()

    corr = np.correlate(x, x, mode="full")
    mid = corr.size // 2
    acf = corr[mid:] / corr[mid]
    return acf[: max_lag + 1]


def plot_acf(power_df, col, savepath, max_lag=200):
    acf_vals1 = compute_acf(power_df[col].to_numpy(), max_lag)
    lags = np.arange(max_lag + 1)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(lags, acf_vals1, color="black")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("ACF of " + col)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")


def plot_power_two_states_with_legend(
    df: pd.DataFrame,
    column_y: str,
    attack_col: str = "Attack",
    state_col: str = "State",
    attack_group_col: str = "Attack-Group",
    avg_window_seconds: int | None = None,
    save_path: str | None = None,
    none_attack_label: str = "None",
):
    """
    Visualize timeseries (left) and boxplots (right) for two states: charging & idle.

    - Left:
        Row 0: charging – one line per Attack
        Row 1: idle     – one line per Attack
      x-axis = time [min], assuming 1 second per sample, starting at 0
      y-axis = df[column_y]

    - Right:
        Row 0: charging – boxplot of df[column_y] per Attack
        Row 1: idle     – boxplot of df[column_y] per Attack

    Colors:
        - One viridis color per Attack (consistent across both states)
        - 'none_attack_label' (default 'None') is forced to red.

    Legend: "Attack-Group: Attack" for each line.
    """

    df = df.copy()

    # basic checks
    needed = [attack_col, state_col, column_y]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    if attack_group_col not in df.columns:
        print(f"Warning: '{attack_group_col}' not in df.columns. Using '{attack_col}' as group.")
        attack_group_col = attack_col

    # list unique attacks and states
    attacks = sorted(df[attack_col].dropna().unique())
    states = sorted(df[state_col].dropna().unique())

    print(f"Unique {attack_col} values:", attacks)
    print(f"Unique {state_col} values:", states)

    print("\nExisting Attack × State combinations (with counts):")
    for a in attacks:
        for s in states:
            mask = (df[attack_col] == a) & (df[state_col] == s)
            n = mask.sum()
            if n > 0:
                print(f"  {attack_col}={a}, {state_col}={s} -> {n} samples")

    # detect idle/charging labels (case-insensitive)
    state_values = df[state_col].dropna().unique()
    state_map = {str(s).lower(): s for s in state_values}

    idle_key = "idle" if "idle" in state_map else ("iddle" if "iddle" in state_map else None)
    charging_key = "charging" if "charging" in state_map else None

    if idle_key is None or charging_key is None:
        raise ValueError(
            f"Could not detect 'idle/iddle' and 'charging' states from {state_values}. "
            "Please adapt the mapping."
        )

    idle_state_val = state_map[idle_key]
    charging_state_val = state_map[charging_key]

    # optional: sort by timestamp if it exists (helps keep traces ordered)
    order_col = "timestamp" if "timestamp" in df.columns else None

    # build mapping Attack -> Attack-Group (for legend text)
    attack_to_group = {}
    for attack_val, g_attack in df.groupby(attack_col):
        group_vals = g_attack[attack_group_col].dropna()
        if len(group_vals) == 0:
            group_val = "unknown"
        else:
            group_val = group_vals.mode().iloc[0]
        attack_to_group[attack_val] = group_val

    # color mapping: one color per Attack (viridis)
    cmap = plt.get_cmap("viridis")
    n_attacks = max(len(attacks), 1)
    attack_to_color = {
        a: cmap(i / (n_attacks - 1 if n_attacks > 1 else 1))
        for i, a in enumerate(attacks)
    }

    # helper: override color for "None"
    def get_attack_color(attack_val):
        if (
            none_attack_label is not None
            and str(attack_val).lower() == str(none_attack_label).lower()
        ):
            return "red"
        return attack_to_color.get(attack_val, "k")

    # helper: build time and aggregated y for one Attack+State group
    def build_time_and_y(g_state: pd.DataFrame):
        if order_col is not None:
            g_state = g_state.sort_values(order_col)

        n = len(g_state)
        if n == 0:
            return np.array([]), np.array([])

        if avg_window_seconds is None or avg_window_seconds <= 1:
            t_sec = np.arange(n)
            t_min = t_sec / 60.0
            y = g_state[column_y].to_numpy()
        else:
            t_sec = np.arange(n)
            bin_index = t_sec // avg_window_seconds
            tmp = g_state.copy()
            tmp["__bin"] = bin_index
            agg = tmp.groupby("__bin")[column_y].mean()

            t_sec_bins = agg.index.to_numpy() * avg_window_seconds
            t_min = t_sec_bins / 60.0
            y = agg.to_numpy()

        return t_min, y

    # collect data for boxplots
    box_data_charging = {}
    box_data_idle = {}

    # create 2x2 figure: left=timeseries, right=boxplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False)

    ax_ts_charging = axes[0, 0]
    ax_box_charging = axes[0, 1]
    ax_ts_idle = axes[1, 0]
    ax_box_idle = axes[1, 1]

    # charging: timeseries & box data
    for attack_val, g_attack in df.groupby(attack_col):
        g_state = g_attack[g_attack[state_col] == charging_state_val].copy()
        if g_state.empty:
            continue

        t_min, y = build_time_and_y(g_state)
        if len(y) == 0:
            continue

        color = get_attack_color(attack_val)
        group_val = attack_to_group.get(attack_val, "unknown")
        label = f"{group_val}: {attack_val}"

        ax_ts_charging.plot(t_min, y, label=label, color=color)
        box_data_charging[attack_val] = y

    ax_ts_charging.set_title(f"{state_col} = {charging_state_val} – timeseries")
    ax_ts_charging.set_ylabel(column_y)
    ax_ts_charging.set_xlabel("time [min]")
    ax_ts_charging.legend(title=f"{attack_group_col}: {attack_col}", fontsize=8)
    for label in ax_ts_charging.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    if box_data_charging:
        attacks_ch = list(box_data_charging.keys())
        data_ch = [box_data_charging[a] for a in attacks_ch]
        bp = ax_box_charging.boxplot(
            data_ch,
            labels=attacks_ch,
            patch_artist=True,
            showfliers=False,
        )
        for patch, a in zip(bp['boxes'], attacks_ch):
            patch.set_facecolor(get_attack_color(a))

        ax_box_charging.set_title(f"{state_col} = {charging_state_val} – boxplot")
        ax_box_charging.set_ylabel(column_y)
        ax_box_charging.set_xlabel(attack_col)
        for label in ax_box_charging.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    # idle: timeseries & box data
    for attack_val, g_attack in df.groupby(attack_col):
        g_state = g_attack[g_attack[state_col] == idle_state_val].copy()
        if g_state.empty:
            continue

        t_min, y = build_time_and_y(g_state)
        if len(y) == 0:
            continue

        color = get_attack_color(attack_val)
        group_val = attack_to_group.get(attack_val, "unknown")
        label = f"{group_val}: {attack_val}"

        ax_ts_idle.plot(t_min, y, label=label, color=color)
        box_data_idle[attack_val] = y

    ax_ts_idle.set_title(f"{state_col} = {idle_state_val} – timeseries")
    ax_ts_idle.set_ylabel(column_y)
    ax_ts_idle.set_xlabel("time [min]")
    ax_ts_idle.legend(title=f"{attack_group_col}: {attack_col}", fontsize=8)
    for label in ax_ts_idle.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    if box_data_idle:
        attacks_id = list(box_data_idle.keys())
        data_id = [box_data_idle[a] for a in attacks_id]
        bp = ax_box_idle.boxplot(
            data_id,
            labels=attacks_id,
            patch_artist=True,
            showfliers=False,
        )
        for patch, a in zip(bp['boxes'], attacks_id):
            patch.set_facecolor(get_attack_color(a))

        ax_box_idle.set_title(f"{state_col} = {idle_state_val} – boxplot")
        ax_box_idle.set_ylabel(column_y)
        ax_box_idle.set_xlabel(attack_col)
        for label in ax_box_idle.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    plt.tight_layout()

    # save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



def plot_confusion_matrix(y_test, y_pred, label_encoder):
    cm = confusion_matrix(y_test, y_pred)
    classes = label_encoder.classes_

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (counts)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2.0 else "black",
            )

    plt.tight_layout()
    plt.show()



def plot_split_distribution(
    df,
    attack_col="Attack",
    state_col="State",
    split_col="split",
    save_path=None,
):

    counts = (
        df.groupby([state_col, attack_col, split_col])
          .size()
          .reset_index(name="count")
    )

    states  = sorted(counts[state_col].dropna().unique())
    attacks = sorted(counts[attack_col].dropna().unique())
    splits  = ["train", "val", "test"]  # ordre fix
    splits_present = [s for s in splits if s in counts[split_col].unique()]

    n_states = len(states)
    fig, axes = plt.subplots(
        n_states, 1,
        figsize=(5, 4 * n_states),
        sharex=True
    )
    if n_states == 1:
        axes = [axes]

    width = 0.2
    x = np.arange(len(attacks))


    cmap = plt.get_cmap("magma")
    n_splits = len(splits_present)
    split_colors = {
        split: cmap(i / max(1, 3))
        for i, split in enumerate(splits_present)
    }

    for ax, state in zip(axes, states):
        df_state = counts[counts[state_col] == state]

        for i, split in enumerate(splits_present):
            df_s = df_state[df_state[split_col] == split]

            counts_for_attacks = []
            for a in attacks:
                row = df_s[df_s[attack_col] == a]
                counts_for_attacks.append(
                    int(row["count"].iloc[0]) if not row.empty else 0
                )

            offset = (i - (len(splits_present)-1)/2) * width
            ax.bar(
                x + offset,
                counts_for_attacks,
                width=width,
                label=split,
                color=split_colors[split],
            )

        ax.set_title(f"{state_col} = {state}", size=18)
        ax.set_ylabel("nombre de mostres", size=18)
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=45, ha="right", size=18)
        ax.legend(title=split_col)

    axes[-1].set_xlabel(attack_col, size=18)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ----------------------------------------------------------------------
# Plot training history (LOSS + ACC) and save PNGs
# ----------------------------------------------------------------------
def plot_training_history(history: Any, output_dir: Path, prefix: str = "training") -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    hist = history.history if hasattr(history, "history") else dict(history)

    # Loss
    loss = hist.get("loss", None)
    val_loss = hist.get("val_loss", None)
    if loss is None:
        print("[plot_training_history] No s'ha trobat la clau 'loss' al history.")
        return

    # Accuracy
    acc_keys = ["accuracy", "acc", "binary_accuracy", "sparse_categorical_accuracy", "categorical_accuracy"]
    val_acc_keys = ["val_accuracy", "val_acc", "val_binary_accuracy", "val_sparse_categorical_accuracy", "val_categorical_accuracy"]

    acc = next((hist.get(k) for k in acc_keys if k in hist), None)
    val_acc = next((hist.get(k) for k in val_acc_keys if k in hist), None)

    epochs = range(1, len(loss) + 1)

    # Loss plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss, label="Train loss")
    if val_loss is not None:
        plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Evolució de la loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_loss.png", dpi=150)
    plt.close()

    # Accuracy plot
    if acc is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, acc, label="Train accuracy")
        if val_acc is not None:
            plt.plot(epochs, val_acc, label="Val accuracy")
        else:
            print("[plot_training_history] No s'ha trobat cap clau de validació per accuracy (p.ex. 'val_accuracy').")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Evolució de l'accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_accuracy.png", dpi=150)
        plt.close()
    else:
        print("[plot_training_history] No s'han trobat claus d'accuracy (accuracy/acc/binary_accuracy/...).")



# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------


def save_training_history(history, out_path_prefix: Path) -> None:
    """Save training/validation loss and accuracy to PNG files and JSON data."""
    hist = history.history
    cleaned_hist = {k: [float(v) for v in values] for k, values in hist.items()}

    hist_json_path = out_path_prefix.with_name(out_path_prefix.name + "_history.json")
    with open(hist_json_path, "w") as f:
        json.dump(cleaned_hist, f, indent=2)

    acc = hist.get("accuracy", hist.get("acc"))
    val_acc = hist.get("val_accuracy", hist.get("val_acc"))
    loss = hist["loss"]
    val_loss = hist["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss, label="Train loss", color="black")
    plt.plot(epochs, val_loss, label="Val loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_prefix.with_name(out_path_prefix.name + "_loss.png"), dpi=150)
    plt.close()

    if acc is not None and val_acc is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, acc, label="Train acc", color="black")
        plt.plot(epochs, val_acc, label="Val acc", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training / Validation accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_path_prefix.with_name(out_path_prefix.name + "_acc.png"), dpi=150
        )
        plt.close()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: Path,
) -> None:
    """Save absolute confusion matrix to PNG and JSON."""
    cm = confusion_matrix(y_true, y_pred)

    cm_json_path = out_path.with_suffix(".json")
    cm_data = {
        "class_names": list(class_names),
        "matrix": cm.tolist(),
    }
    with open(cm_json_path, "w") as f:
        json.dump(cm_data, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (counts)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2.0 else "black",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _safe_filename(name: str) -> str:
    """Return a filesystem-friendly filename."""
    return "".join(ch if ch.isalnum() or ch in "._-+=" else "_" for ch in str(name))


def _infer_step_kind(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series with values in {"1", "seq", None} depending on df['step'].

    Accepts:
      - step == 1  -> "1"
      - step == "seq" (case-insensitive) -> "seq"
      - step == seq_len (numeric encoding of "seq") -> "seq"
    """
    def kind(row: pd.Series) -> str | None:
        s = row.get("step", None)
        if s is None:
            return None

        # step=1
        try:
            if float(s) == 1.0:
                return "1"
        except Exception:
            pass

        # step="seq"
        if isinstance(s, str) and s.strip().lower() == "seq":
            return "seq"

        # step == seq_len (numeric encoding of "seq")
        try:
            sl = row.get("seq_len", None)
            if sl is not None and float(s) == float(sl):
                return "seq"
        except Exception:
            pass

        return None

    return df.apply(kind, axis=1)


def _flatten_metrics(obj: Any, prefix: str = "", out: dict[str, float] | None = None) -> dict[str, float]:
    """
    Flatten nested dict metrics into a single-level dict.

    Rules:
    - Scalar numbers become columns.
    - Booleans become 0/1 columns.
    - Lists of length 2 with numeric values become *_low and *_high (for CI/error bars).
    - Other lists (curves) are ignored to avoid huge columns.
    """
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_metrics(v, new_key, out)

    elif isinstance(obj, list):
        if len(obj) == 2 and all(isinstance(x, (int, float, np.number)) for x in obj):
            out[f"{prefix}_low"] = float(obj[0])
            out[f"{prefix}_high"] = float(obj[1])
        else:
            # ignore long arrays (curves) or non-numeric lists
            pass

    else:
        if isinstance(obj, bool):
            out[prefix] = int(obj)
        elif isinstance(obj, (int, float, np.number)):
            out[prefix] = float(obj)

    return out


def load_all_metrics_flat(base_output_root):
    """
    Recursively loads every **/metrics.json under base_output_root and flattens metrics + config.

    Adds a few key run columns:
      run_name, run_dir, task, model_type, target_col, seq_len, step

    Returns:
      pd.DataFrame
    """
    base_output_root = Path(base_output_root)
    rows: list[dict[str, Any]] = []

    for metrics_path in base_output_root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        run_name = run_dir.name

        with open(metrics_path, "r") as f:
            m = json.load(f)

        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        else:
            cfg = {}

        row: dict[str, Any] = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "task": m.get("task", cfg.get("task")),
            "model_type": m.get("model_type", cfg.get("model_type")),
            "target_col": m.get("target_col", cfg.get("target_col")),
            "seq_len": cfg.get("seq_len"),
            "step": cfg.get("step"),
        }

        # Flatten EVERYTHING from metrics + config (prefix config keys to avoid collisions)
        row.update(_flatten_metrics(m))
        row.update({f"cfg.{k}": v for k, v in _flatten_metrics(cfg).items()})

        rows.append(row)

    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["task", "model_type", "seq_len", "step"] if c in df.columns]
    if sort_cols and not df.empty:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def _find_ci_cols(df: pd.DataFrame, metric: str) -> tuple[str | None, str | None]:
    """
    For a given metric column, try to locate corresponding CI columns in flattened data.

    Patterns supported (as seen in metrics.json):
      - ...FPR -> ...FPR_CI_clopper_pearson_low/high (fallback normal)
      - ...false_positives_per_hour / ...average_FP_per_hour -> ...FPH_CI_clopper_pearson_low/high (fallback normal)
    """
    parent = metric.rsplit(".", 1)[0] + "." if "." in metric else ""

    candidates: list[str] = []

    if metric.endswith("FPR") or metric.endswith(".FPR"):
        candidates = [
            f"{parent}FPR_CI_clopper_pearson",
            f"{parent}FPR_CI_normal",
        ]

    if metric.endswith("false_positives_per_hour") or metric.endswith("average_FP_per_hour"):
        candidates = [
            f"{parent}FPH_CI_clopper_pearson",
            f"{parent}FPH_CI_normal",
        ]

    for base in candidates:
        lo, hi = f"{base}_low", f"{base}_high"
        if lo in df.columns and hi in df.columns:
            return lo, hi

    return None, None


def build_metric_triplets(df: pd.DataFrame, task: str) -> list[dict[str, str | None]]:
    """
    Returns list of dicts:
      { "name": <base name>, "all": <col or None>, "charge": <col or None>, "idle": <col or None> }
    """
    sub = df[df["task"] == task].copy()

    triplets: list[dict[str, str | None]] = []
    seen_names: set[str] = set()

    def add_triplet(name: str, all_col: str | None, charge_col: str | None, idle_col: str | None) -> None:
        if name in seen_names:
            return
        seen_names.add(name)
        triplets.append({"name": name, "all": all_col, "charge": charge_col, "idle": idle_col})

    # top-level scalar metrics (ALL only)
    for col in ["training_time_seconds", "avg_inference_latency_per_window_sec"]:
        if col in sub.columns and pd.api.types.is_numeric_dtype(sub[col]):
            add_triplet(col, col, None, None)

    # RESEARCH: test_global_metrics.<suffix>  vs  binary_state_metrics.<state>.research.<suffix>
    all_pref = "test_global_metrics."
    ch_pref = "binary_state_metrics.charging.research."
    id_pref = "binary_state_metrics.idle.research."

    for col in sub.columns:
        if col.startswith(all_pref) and pd.api.types.is_numeric_dtype(sub[col]):
            suffix = col[len(all_pref):]
            add_triplet(
                f"research.{suffix}",
                col,
                ch_pref + suffix if (ch_pref + suffix) in sub.columns else None,
                id_pref + suffix if (id_pref + suffix) in sub.columns else None,
            )

    # OPERATIONAL: binary_operational_metrics.<suffix> (threshold/fpr_target) with state analogs
    for suffix in ["threshold", "fpr_target"]:
        all_col = f"binary_operational_metrics.{suffix}"
        ch_col = f"binary_state_metrics.charging.operational.{suffix}"
        id_col = f"binary_state_metrics.idle.operational.{suffix}"
        if all_col in sub.columns and pd.api.types.is_numeric_dtype(sub[all_col]):
            add_triplet(
                f"operational.{suffix}",
                all_col,
                ch_col if ch_col in sub.columns else None,
                id_col if id_col in sub.columns else None,
            )

    # OPERATIONAL: validation/test blocks
    for block in ["validation", "test"]:
        all_pref = f"binary_operational_metrics.{block}."
        ch_pref = f"binary_state_metrics.charging.operational.{block}."
        id_pref = f"binary_state_metrics.idle.operational.{block}."

        for col in sub.columns:
            if col.startswith(all_pref) and pd.api.types.is_numeric_dtype(sub[col]):
                suffix = col[len(all_pref):]
                add_triplet(
                    f"operational.{block}.{suffix}",
                    col,
                    ch_pref + suffix if (ch_pref + suffix) in sub.columns else None,
                    id_pref + suffix if (id_pref + suffix) in sub.columns else None,
                )

    return triplets


def plot_metric_triplet_row(
    df: pd.DataFrame,
    task: str,
    triplet: dict[str, str | None],
    out_root: str | Path = "plots_power",
    dpi: int = 160,
    show: bool = False,
) -> None:
    """
    Plot one triplet (ALL/CHARGE/IDLE) as 1 row of subplots for a given task.
    Saves a PNG under out_root/<task>/<triplet_name>.png.
    """
    data = df[df["task"] == task].copy()
    data["step_kind"] = _infer_step_kind(data)
    data = data[data["step_kind"].isin(["1", "seq"])].copy()

    # Decide which panels are plottable (column exists AND has data)
    candidate_panels = [("all", "ALL"), ("charge", "CHARGE"), ("idle", "IDLE")]
    panels: list[tuple[str, str, str]] = []
    for k, title in candidate_panels:
        col = triplet.get(k)
        if not col or col not in data.columns:
            continue
        tmp = data.dropna(subset=["seq_len", col, "model_type"])
        if tmp.empty:
            continue
        panels.append((k, title, col))

    if not panels:
        return

    model_types = sorted(data["model_type"].dropna().unique())
    cmap = plt.get_cmap("magma")
    n = 4
    color_map = {mt: cmap(0.0 if n == 1 else i / (n - 1)) for i, mt in enumerate(model_types)}
    linestyle_map = {"1": "-", "seq": ":"}

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6), sharex=True)
    if ncols == 1:
        axes = [axes]

    handles_labels: tuple[list[Any], list[str]] | None = None

    for ax, (_k, title, metric_col) in zip(axes, panels):
        sub = data.dropna(subset=["seq_len", metric_col, "model_type"])

        ci_lo, ci_hi = _find_ci_cols(sub, metric_col)

        # TTD fallback SEM if no CI and episodes_detected exists
        ttd_sem_col: str | None = None
        if ci_lo is None and ("avg_ttd_seconds" in metric_col or "avg_ttd_hours" in metric_col):
            parent = metric_col.rsplit(".", 1)[0] + "."
            ep = f"{parent}episodes_detected"
            if ep in sub.columns:
                ttd_sem_col = ep

        ax.set_title(title)

        for mt in model_types:
            for sk in ["1", "seq"]:
                s2 = sub[(sub["model_type"] == mt) & (sub["step_kind"] == sk)]
                if s2.empty:
                    continue

                cols_to_take = [metric_col]
                if ci_lo and ci_hi:
                    cols_to_take += [ci_lo, ci_hi]
                if ttd_sem_col:
                    cols_to_take += [ttd_sem_col]

                agg = (
                    s2.groupby("seq_len", as_index=False)[cols_to_take]
                    .mean()
                    .sort_values("seq_len")
                )
                x = agg["seq_len"].to_numpy()
                y = agg[metric_col].to_numpy()

                label = f"{mt} | step={sk}"

                if ci_lo and ci_hi:
                    lo = agg[ci_lo].to_numpy()
                    hi = agg[ci_hi].to_numpy()
                    yerr = np.vstack([np.clip(y - lo, 0, np.inf), np.clip(hi - y, 0, np.inf)])
                    ax.errorbar(
                        x, y, yerr=yerr,
                        linestyle=linestyle_map[sk],
                        marker="o",
                        color=color_map[mt],
                        capsize=3,
                        label=label,
                    )
                elif ttd_sem_col:
                    n_ep = agg[ttd_sem_col].to_numpy()
                    sem = np.where(n_ep > 0, np.abs(y) / np.sqrt(n_ep), 0.0)
                    ax.errorbar(
                        x, y, yerr=sem,
                        linestyle=linestyle_map[sk],
                        marker="o",
                        color=color_map[mt],
                        capsize=3,
                        label=label,
                    )
                else:
                    ax.plot(
                        x, y,
                        linestyle=linestyle_map[sk],
                        marker="o",
                        color=color_map[mt],
                        label=label,
                    )

        ax.set_xlabel("seq_len")
        ax.set_ylabel(str(triplet["name"]))

        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    fig.suptitle(f"{triplet['name']} | task={task} (step=1 solid, step=seq dotted)", y=1.02)

    if handles_labels:
        handles, labels = handles_labels
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()

    out_root = Path(out_root)
    save_path = out_root / _safe_filename(task) / f"{_safe_filename(str(triplet['name']))}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_all_triplets_for_tasks(
    df: pd.DataFrame,
    tasks: list[str] | None = None,
    out_root: str | Path = "plots_power",
    show: bool = False,
) -> None:
    """
    Build triplets per task and plot them all.
    """
    available_tasks = sorted(df["task"].dropna().unique())
    tasks_to_plot = available_tasks if tasks is None else [t for t in tasks if t in available_tasks]

    for task in tasks_to_plot:
        triplets = build_metric_triplets(df, task)
        for trip in triplets:
            plot_metric_triplet_row(df, task, trip, out_root=out_root, show=show)



def plot_generalization_metric_all_and_states(
    df_global,
    df_state,
    metric_base_name,
    title=None,
    states_order=("ALL", "charging", "idle"),
    save_path=None,
):
    metric_mean = f"{metric_base_name}_mean"
    metric_std  = f"{metric_base_name}_std"

    # same attacks on all subplots
    attacks = sorted(df_global["attack"].unique())
    x = np.arange(len(attacks))
    subsets = ["all", "seen_attacks", "heldout_attack"]

    # magma palette for the 3 subsets
    cmap = plt.cm.get_cmap("magma")
    colors = cmap(np.linspace(0.2, 0.8, len(subsets)))

    fig, axes = plt.subplots(
        1, len(states_order),
        figsize=(5 * len(states_order), 4),
        sharey=True,
    )

    for ax, state in zip(axes, states_order):
        # data for this state
        if state == "ALL":
            df_state_cur = df_global
        else:
            df_state_cur = df_state[df_state["state"] == state]

        # attacks with a non-NaN heldout value in this state
        mask_valid_heldout = (
            (df_state_cur["subset"] == "heldout_attack")
            & df_state_cur[metric_mean].notna()
        )
        attacks_with_heldout = set(
            df_state_cur.loc[mask_valid_heldout, "attack"].unique()
        )

        valid_mask = np.array([a in attacks_with_heldout for a in attacks])

        for i, subset in enumerate(subsets):
            df_s_raw = df_state_cur[df_state_cur["subset"] == subset]
            if df_s_raw.empty:
                continue

            df_s = df_s_raw.set_index("attack").reindex(attacks)

            means = df_s[metric_mean].to_numpy()
            stds  = df_s[metric_std].to_numpy()

            # hide attacks without valid heldout in this state
            means = np.where(valid_mask, means, np.nan)
            stds  = np.where(valid_mask, stds,  np.nan)

            if np.all(np.isnan(means)):
                continue

            offset = (i - 1) * 0.08

            ax.errorbar(
                x + offset,
                means,
                yerr=stds,
                fmt="o-",
                capsize=4,
                label=subset.replace("_", " "),
                color=colors[i],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=45, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"State = {state}")
        ax.set_ylabel(metric_base_name)

        ax.grid(False)

    if title is None:
        title = f"{metric_base_name} (leave-one-attack-out, ALL vs charging/idle)"
    fig.suptitle(title)
    plt.tight_layout()
    axes[-1].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()




def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+=" else "_" for ch in str(name))


def _collect_metric_bases(curves: dict) -> list[str]:
    """
    From curves[kind], collect base metrics that have *_mean keys.
    Handles:
      - ALL:                <base>_mean
      - CHARGE state keys:  state_charging_<base>_mean
      - IDLE state keys:    state_idle_<base>_mean
    """
    bases = set()
    for kind, d in curves.items():
        for k in d.keys():
            if not k.endswith("_mean"):
                continue
            if k == "severity_mean":
                continue

            if k.startswith("state_charging_"):
                bases.add(k[len("state_charging_") : -len("_mean")])
            elif k.startswith("state_idle_"):
                bases.add(k[len("state_idle_") : -len("_mean")])
            elif k != "severity":
                bases.add(k[:-len("_mean")])
    return sorted(bases)


def plot_robustness_all_charge_idle(
    json_path: str | Path,
    out_dir: str | Path = "plots_robustness",
    dpi: int = 160,
    show: bool = False,
):
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        obj = json.load(f)

    curves = obj.get("curves", {})
    if not curves:
        raise ValueError("No 'curves' found in robustness_curves.json")

    kinds = list(curves.keys())
    metric_bases = _collect_metric_bases(curves)

    # magma colors per kind
    cmap = plt.get_cmap("magma")
    n = 3
    kind_color = {k: cmap(0.0 if n == 1 else i / (n - 1)) for i, k in enumerate(kinds)}

    # helper: build panel keys for a given base metric
    def panel_defs(base: str):
        return [
            ("ALL", base, f"{base}_mean", f"{base}_std"),
            ("CHARGE", f"state_charging_{base}", f"state_charging_{base}_mean", f"state_charging_{base}_std"),
            ("IDLE", f"state_idle_{base}", f"state_idle_{base}_mean", f"state_idle_{base}_std"),
        ]

    for base in metric_bases:
        # keep only panels that actually have data in at least one kind
        panels = []
        for title, _, mean_key, std_key in panel_defs(base):
            has_any = any(mean_key in curves[kind] for kind in kinds)
            if has_any:
                panels.append((title, mean_key, std_key))

        if not panels:
            continue

        # 6x6 per panel => total width = 6 * n_panels
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), sharex=True)
        if len(panels) == 1:
            axes = [axes]

        handles_labels = None

        for ax, (panel_title, mean_key, std_key) in zip(axes, panels):
            ax.set_title(panel_title)

            any_plotted = False
            for kind in kinds:
                d = curves[kind]
                sev = np.asarray(d.get("severity", []), dtype=float)

                y_mean = d.get(mean_key, None)
                if y_mean is None:
                    continue
                y_mean = np.asarray(y_mean, dtype=float)

                y_std = d.get(std_key, None)
                yerr = np.asarray(y_std, dtype=float) if y_std is not None else None

                # shape guard
                if len(sev) != len(y_mean) or len(sev) == 0:
                    continue

                ax.errorbar(
                    sev,
                    y_mean,
                    yerr=yerr,
                    marker="o",
                    capsize=3,
                    linestyle="-",
                    color=kind_color[kind],
                    label=kind,
                )
                any_plotted = True

            ax.set_xlabel("Degradation severity")
            ax.set_ylabel(base)

            if any_plotted and handles_labels is None:
                handles_labels = ax.get_legend_handles_labels()

        # one shared legend for the whole row
        if handles_labels:
            handles, labels = handles_labels
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)),
                       bbox_to_anchor=(0.5, 1.05))

        fig.suptitle(f"{base} vs severity", y=1.10)
        fig.tight_layout()

        save_path = out_dir / f"{_safe_filename(base)}.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save_path}")

        if show:
            plt.show()
        plt.close(fig)




def plot_all_confusion_matrices_for_tasks(
    df: pd.DataFrame,
    tasks: list[str] | None = None,
    out_root: str | Path = "plots_power",
    show: bool = False,
) -> None:
    """
    For each run in df (optionally filtered by tasks), load metrics.json and plot:
      - Global confusion matrix (test_confusion_matrix)
      - For scenario/multiattack: per-state confusion matrices (idle/charging), if present.

    Output structure:
      <out_root>/<task>/<run_name>/cm_global.png
      <out_root>/<task>/<run_name>/cm_idle.png
      <out_root>/<task>/<run_name>/cm_charging.png
    """
    if df.empty:
        print("[plot_all_confusion_matrices_for_tasks] Empty dataframe, nothing to plot.")
        return

    base_out = Path(out_root)

    available_tasks = sorted(df["task"].dropna().unique())
    tasks_to_plot = available_tasks if tasks is None else [t for t in tasks if t in available_tasks]

    for task in tasks_to_plot:
        df_task = df[df["task"] == task].copy()

        for idx, row in df_task.iterrows():
            run_name = row.get("run_name", None) or f"run_{idx}"
            run_dir = Path(row["run_dir"])

            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                print(f"[plot_all_confusion_matrices_for_tasks] metrics.json not found for run_dir={run_dir}")
                continue

            with open(metrics_path, "r") as f:
                m = json.load(f)

            task_out_dir = base_out / task / run_name
            task_out_dir.mkdir(parents=True, exist_ok=True)

            # Global confusion matrix
            cm_info = m.get("test_confusion_matrix")
            if cm_info and "matrix" in cm_info and "labels" in cm_info:
                cm = np.array(cm_info["matrix"])
                labels = list(cm_info["labels"])
                out_path = task_out_dir / "cm_global.png"
                title = f"{task} | {run_name} | GLOBAL"
                plot_confusion_matrix_from_data(
                    cm,
                    labels,
                    out_path=out_path,
                    title=title,
                    show=show,
                )

            # Per-state confusion matrices for scenario & multiattack
            if task in {"scenario", "multiattack"}:
                state_metrics = m.get("state_multiclass_metrics") or {}
                for state, info in state_metrics.items():
                    if not isinstance(info, dict):
                        continue
                    cm_state_info = info.get("confusion_matrix")
                    if (
                        not cm_state_info
                        or "matrix" not in cm_state_info
                        or "labels" not in cm_state_info
                    ):
                        continue

                    cm_state = np.array(cm_state_info["matrix"])
                    labels_state = list(cm_state_info["labels"])
                    out_path_state = task_out_dir / f"cm_{state}.png"
                    title_state = f"{task} | {run_name} | {state}"
                    plot_confusion_matrix_from_data(
                        cm_state,
                        labels_state,
                        out_path=out_path_state,
                        title=title_state,
                        show=show,
                    )

def plot_confusion_matrix_from_data(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path | None = None,
    title: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot (and optionally save) a confusion matrix given the already-computed matrix
    and the corresponding class_names.

    This mirrors the style of save_confusion_matrix, but assumes cm is already computed.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title or "Confusion matrix (counts)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2.0 else "black",
            )

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
