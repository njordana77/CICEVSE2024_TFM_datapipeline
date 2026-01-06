# pipeline_explicability_power.py

#!/usr/bin/env python

"""
Surrogate explainability pipeline for EVSE power-based attack detection (post-hoc).

This script consumes a completed pipeline1 run directory and builds a simple,
interpretable decision-tree surrogate that approximates either:
- the base model’s predictions (label_source="model"), or
- the true labels (label_source="true").

Workflow:
- Load pipeline1 artifacts from run_dir (config.json, scaler.pkl, label_encoder.pkl, and the trained model:
  model.keras for Keras models or xgb_model.pkl for XGBoost).
- Reload the original power CSV and reproduce the *exact* preprocessing used in pipeline1:
  label preparation for the same task, NaN cleaning, and chronological split per (Attack, State).
- Rebuild sliding-window sequences using the *same* scaler, seq_len, and step from config.json.
  Save these windows to windows_surrogate.npz for later notebook-based explanations.
- Train and evaluate a shallow decision tree surrogate (max_depth) via train_surrogate_tree,
  which summarizes windows into tabular statistics, fits the tree, and reports metrics.
- Save surrogate artifacts into run_dir:
  surrogate_tree.joblib, surrogate_feature_stats.json, and surrogate_metrics.json.

Main entry point: run_pipeline2_surrogate(...). Also provides a CLI wrapper (parse_args/main).
"""


import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import joblib

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, balanced_accuracy_score
from tensorflow.keras.models import load_model as keras_load_model

from func_aux.func_preproc import (
    set_global_seed,
    load_power_data,
    encode_y_data,
    prepare_labels_for_task,
    split_df_per_attack_and_state_chronologically,
    make_sequences_from_df_single_class,
)

from func_aux.func_models import (
    train_surrogate_tree,
)


def run_pipeline2_surrogate(
    run_dir: Path,
    csv_path: Path | None = None,
    col_time: str | None = None,
    max_depth: int = 3,
    label_source: str = "model",
    seed: int = 42,
) -> None:
    """
    Pipeline 2:

    - Load config and artifacts from a pipeline1 run directory.
    - Rebuild train/test windows using the SAME scaler and splits.
    - Summarise windows into tabular stats.
    - Train and evaluate a decision tree surrogate.
    - Save tree, metrics and feature importance plots.
    """
    set_global_seed(seed)
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    # Load config and artifacts from pipeline1
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")

    with open(config_path, "r") as f:
        config = json.load(f)

    task = config["task"]
    target_col = config["target_col"]
    seq_len = config["seq_len"]
    step = config["step"]
    num_features = config["num_features"]
    cat_features = config["cat_features"]

    # If not provided, use the same CSV and time column as pipeline1
    if csv_path is None:
        csv_path = Path(config["csv_path"])
    else:
        csv_path = Path(csv_path)

    if col_time is None:
        col_time = config["col_time"]

    print(f"[Surrogate pipeline] Using run_dir: {run_dir}")
    print(f"[Surrogate pipeline] Task: {task}")
    print(f"[Surrogate pipeline] CSV: {csv_path}")
    print(f"[Surrogate pipeline] Time column: {col_time}")
    print(f"[Surrogate pipeline] max_depth: {max_depth}")
    print(f"[Surrogate pipeline] label_source: {label_source}")

    # Load base model, scaler, and label encoder
    from tensorflow.keras.models import load_model as keras_load_model

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
            f"No base model found in {run_dir}. "
            "Expected either model.keras (Keras) or xgb_model.pkl (XGBoost)."
        )

    scaler = joblib.load(run_dir / "scaler.pkl")
    label_encoder = joblib.load(run_dir / "label_encoder.pkl")
    class_names = list(label_encoder.classes_)


    # Load data and rebuild splits exactly like pipeline1
    df_power = load_power_data(str(csv_path), col_time=col_time)

    # Make sure the correct target column exists
    df_power, target_col_pipeline2 = prepare_labels_for_task(df_power, task)
    if target_col_pipeline2 != target_col:
        print(
            f"[Warning] target_col from config={target_col}, "
            f"but prepare_labels_for_task returned {target_col_pipeline2}"
        )

    # Reuse encode_y_data just to reproduce the same cleaning (drop NaNs)
    feature_cols_all = num_features + cat_features
    df_clean, feature_cols_used, le_tmp = encode_y_data(
        df_power,
        feature_cols_all,
        target_col,
    )

    # Overwrite 'y' with the label encoder used in pipeline1,
    # to keep the same class<->index mapping as the base model.
    df_clean["y"] = label_encoder.transform(df_clean[target_col])

    # Chronological split per (Attack, State), same fractions as pipeline1
    df_train, df_val, df_test = split_df_per_attack_and_state_chronologically(
        df_clean,
        attack_col="Attack",
        state_col="State",
        time_col="timestamp",
        train_frac=0.7,
        val_frac=0.15,
    )

    print("[Surrogate] Train/val/test shapes:", df_train.shape, df_val.shape, df_test.shape)

    # Build windows using the scaler from pipeline1
    X_train, y_train = make_sequences_from_df_single_class(
        df=df_train,
        num_features=num_features,
        cat_features=cat_features,
        scaler=scaler,
        seq_len=seq_len,
        step=step,
        label_col="y",
    )

    X_test, y_test = make_sequences_from_df_single_class(
        df=df_test,
        num_features=num_features,
        cat_features=cat_features,
        scaler=scaler,
        seq_len=seq_len,
        step=step,
        label_col="y",
    )

    print("[Surrogate] Windows train/test:", X_train.shape, X_test.shape)

    # Save windows and labels so we can explain alerts later from a notebook
    np.savez_compressed(
        run_dir / "windows_surrogate.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


    feature_names = num_features + cat_features

    # Train surrogate tree
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

    # Save surrogate tree and metadata
    joblib.dump(tree, run_dir / "surrogate_tree.joblib")

    with open(run_dir / "surrogate_feature_stats.json", "w") as f:
        json.dump(
            {
                "stat_names": stat_names,
                "feature_names": feature_names,
            },
            f,
            indent=2,
        )


    with open(run_dir / "surrogate_metrics.json", "w") as f:
        json.dump(tree_metrics, f, indent=2)

    print(f"[Surrogate] Saved surrogate tree and metrics in: {run_dir}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline 2: train decision-tree surrogates for a pipeline1 run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a pipeline1 run directory (contains model.keras, config.json, etc.).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to the CSV with power data. If not given, uses the one from config.json.",
    )
    parser.add_argument(
        "--col-time",
        type=str,
        default=None,
        help="Name of the time column. If not given, uses the one from config.json.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum depth of the surrogate decision tree.",
    )
    parser.add_argument(
        "--label-source",
        type=str,
        choices=["model", "true"],
        default="model",
        help="Labels used to train the tree: "
             "'model' = imitate base model, 'true' = use true labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline2_surrogate(
        run_dir=Path(args.run_dir),
        csv_path=Path(args.csv_path) if args.csv_path is not None else None,
        col_time=args.col_time,
        max_depth=args.max_depth,
        label_source=args.label_source,
    )


if __name__ == "__main__":
    main()
