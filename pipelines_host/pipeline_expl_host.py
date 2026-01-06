#!/usr/bin/env python

"""
HOST surrogate explainability pipeline (post-hoc decision-tree surrogate).

This script loads a trained HOST run (pipeline1 artifacts) and builds an interpretable
DecisionTree surrogate over window-level summary statistics to explain either:
- the base model’s behavior (label_source="model" for fidelity), or
- the true labels (label_source="true" for accuracy).

Key steps:
- Read config + load artifacts from run_dir (model.keras or xgb_model.pkl, scaler.pkl, label_encoder.pkl).
- Load and clean the HOST CSV, create task labels (binary/scenario/multiattack), and apply the same filtering rules.
- Select features from config.json or optionally pick top-K from feature_importances.json.
- Encode targets using the saved label_encoder, split chronologically, rebuild scaled sliding windows, and save them.
- Get base-model probabilities on windows, choose surrogate labels (model vs true), then summarize each window
  into tabular stats (mean/std/min/max/last) and train a shallow DecisionTree.
- Evaluate surrogate vs target, fidelity to base model, and vs true labels; save tree rules, importances, and metrics
  (JSON + PNG) back into run_dir.

"""


import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

# Add project root (TFM/) to sys.path (same pattern as pipeline_trainingmodels_host.py)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from func_aux.func_preproc import json_sanitize, set_global_seed  # noqa: E402
from func_aux.func_models import train_surrogate_tree  # noqa: E402


def _resolve_path_maybe_relative(p: Path, base: Path) -> Path:
    if p.is_absolute():
        return p
    return (base / p).resolve()


def run_pipeline2_surrogate(
    run_dir: Path,
    csv_path: Optional[Path] = None,
    col_time: Optional[str] = None,
    max_depth: int = 4,
    label_source: str = "model",
    seed: int = 42,
) -> None:
    set_global_seed(seed)

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")

    with open(config_path, "r") as f:
        config = json.load(f)

    task = config["task"]
    target_col = config["target_col"]
    seq_len = int(config["seq_len"])
    step = int(config["step"])
    num_features = list(config["num_features"])
    cat_features = list(config["cat_features"])

    # Keep these for logging parity with pipeline1 (even if we reuse windows.npz)
    if csv_path is None:
        csv_path_cfg = Path(config["csv_path"])
        csv_path = _resolve_path_maybe_relative(csv_path_cfg, base=run_dir)
        if not csv_path.exists():
            csv_path_alt = _resolve_path_maybe_relative(csv_path_cfg, base=ROOT)
            if csv_path_alt.exists():
                csv_path = csv_path_alt
    else:
        csv_path = Path(csv_path)

    if col_time is None:
        col_time = str(config.get("col_time", "timestamp"))

    print(f"[Surrogate pipeline] Using run_dir: {run_dir}")
    print(f"[Surrogate pipeline] Task: {task}")
    print(f"[Surrogate pipeline] CSV: {csv_path}")
    print(f"[Surrogate pipeline] Time column: {col_time}")
    print(f"[Surrogate pipeline] max_depth: {max_depth}")
    print(f"[Surrogate pipeline] label_source: {label_source}")
    print(f"[Surrogate pipeline] seed: {seed}")

    # Load base model (same convention as pipeline1 outputs)
    keras_path = run_dir / "model.keras"
    xgb_path = run_dir / "xgb_model.pkl"

    if keras_path.exists():
        print("[Surrogate] Detected Keras base model (model.keras).")
        model = keras_load_model(keras_path)
    elif xgb_path.exists():
        print("[Surrogate] Detected XGBoost base model (xgb_model.pkl).")
        model = joblib.load(xgb_path)
    else:
        raise FileNotFoundError(
            f"No base model found in {run_dir}. Expected model.keras or xgb_model.pkl."
        )

    scaler_path = run_dir / "scaler.pkl"
    le_path = run_dir / "label_encoder.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler.pkl in {run_dir}")
    if not le_path.exists():
        raise FileNotFoundError(f"Missing label_encoder.pkl in {run_dir}")

    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(le_path)
    class_names = list(label_encoder.classes_)

    # Reuse EXACT windows from pipeline1 (guarantees same splits/scaling)
    windows_path = run_dir / "windows.npz"
    if not windows_path.exists():
        raise FileNotFoundError(
            f"windows.npz not found in {run_dir}. Run pipeline1_host first."
        )

    w = np.load(windows_path, allow_pickle=False)
    X_train = w["X_train"]
    y_train = w["y_train"]
    X_test = w["X_test"]
    y_test = w["y_test"]

    print("[Surrogate] Windows train/test:", X_train.shape, X_test.shape)

    np.savez_compressed(
        run_dir / "windows_surrogate.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    feature_names = num_features + cat_features

    tree, stat_names, tree_metrics = train_surrogate_tree(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        class_names=class_names,
        feature_names=feature_names,
        max_depth=max_depth,
        label_source=label_source,
        output_dir=run_dir,
    )

    joblib.dump(tree, run_dir / "surrogate_tree.joblib")

    with open(run_dir / "surrogate_feature_stats.json", "w") as f:
        json.dump(
            json_sanitize({"stat_names": stat_names, "feature_names": feature_names}),
            f,
            indent=2,
            allow_nan=False,
        )

    with open(run_dir / "surrogate_metrics.json", "w") as f:
        json.dump(json_sanitize(tree_metrics), f, indent=2, allow_nan=False)

    with open(run_dir / "surrogate_config.json", "w") as f:
        json.dump(
            json_sanitize(
                {
                    "seed": int(seed),
                    "max_depth": int(max_depth),
                    "label_source": str(label_source),
                    "csv_path_used": str(csv_path),
                    "col_time": str(col_time),
                    "source_windows": "windows.npz",
                    "task": str(task),
                    "target_col": str(target_col),
                    "seq_len": int(seq_len),
                    "step": int(step),
                }
            ),
            f,
            indent=2,
            allow_nan=False,
        )

    print(f"[Surrogate] Saved surrogate tree and metrics in: {run_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline 2 (HOST): train surrogates for a pipeline1 run.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--csv-path", type=Path, default=None)
    p.add_argument("--col-time", type=str, default=None)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--label-source", type=str, choices=["model", "true"], default="model")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline2_surrogate(
        run_dir=args.run_dir,
        csv_path=args.csv_path,
        col_time=args.col_time,
        max_depth=args.max_depth,
        label_source=args.label_source,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
