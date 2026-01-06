# func_models.py

# !/usr/bin/env python

"""
Model-building utilities for sequence classification.

This module defines:
- LSTM and temporal convolutional (TCN) architectures for windowed time-series.
- A generic training routine for unimodal Keras models with class weighting.
- Training of a tabular surrogate decision tree over window statistics.
- Training of an XGBoost classifier on flattened or summarised windows.
- Helpers to summarise sliding windows into simple per-feature statistics.
"""


import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from xgboost import XGBClassifier

from pathlib import Path
from typing import List, Dict, Tuple, Optional

from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt


def build_lstm_model(seq_len: int, n_features: int, n_classes: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(seq_len, n_features)),
            layers.LSTM(128),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    return model


def train_unimodal_model(
    X_train, y_train, X_val, y_val, label_encoder, build_model, initial_lr=1e-3
):
    seq_len_ = X_train.shape[1]
    n_features = X_train.shape[2]
    n_classes = len(label_encoder.classes_)

    model = build_model(seq_len_, n_features, n_classes)
    model.summary()

    classes = np.unique(y_train)
    class_weights_arr = compute_class_weight("balanced", classes=classes, y=y_train)

    max_w = 10
    for i, w in enumerate(class_weights_arr):
        class_weights_arr[i] = min(w, max_w)

    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
    print("Class weights:", class_weights)

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    callbacks = [early_stopping, lr_scheduler]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )

    return history, model, class_weights


def residual_tcn_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout: float = 0.1,
) -> tf.Tensor:
    shortcut = x

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
    )(x)
    x = layers.SpatialDropout1D(dropout)(x)

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
    )(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
        )(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def build_tcn_model(
    seq_len: int,
    n_features: int,
    n_classes: int,
) -> tf.keras.Model:
    inputs = layers.Input(shape=(seq_len, n_features))
    x = inputs

    dilations = [1, 2, 4]
    for d in dilations:
        x = residual_tcn_block(
            x,
            filters=64,
            kernel_size=3,
            dilation_rate=d,
            dropout=0.1,
        )

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="tcn_power_classifier",
    )
    return model


# ----------------------------------------------------------------------
# Surrogate tree training
# ----------------------------------------------------------------------


def train_surrogate_tree(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    feature_names: List[str],
    max_depth: int = 3,
    label_source: str = "model",
    output_dir: Path | None = None,
) -> Tuple[DecisionTreeClassifier, List[str], Dict]:
    label_source = label_source.lower()
    if label_source not in {"model", "true"}:
        raise ValueError("label_source must be 'model' or 'true'.")

    print("\n[Surrogate] Summarising windows into tabular features...")
    X_train_tab, stat_names = summarize_windows_stats(X_train, feature_names)
    X_test_tab, _ = summarize_windows_stats(X_test, feature_names)

    # Labels used to train the tree
    if label_source == "model":
        print("[Surrogate] Using base model predictions as labels (surrogate mode).")

        # Check whether this is an XGBoost model (has predict_proba on 2D data)
        is_xgb = hasattr(model, "predict_proba")

        if is_xgb:
            # Flatten windows: (N, T, F) -> (N, T*F)
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)

            proba_train = model.predict_proba(X_train_flat)
            proba_test = model.predict_proba(X_test_flat)

            y_train_tree = np.argmax(proba_train, axis=1)
            y_test_ref = np.argmax(proba_test, axis=1)
        else:
            # Keras / deep model: 3D windows
            proba_train = model.predict(X_train, verbose=0)
            proba_test = model.predict(X_test, verbose=0)

            y_train_tree = np.argmax(proba_train, axis=1)
            y_test_ref = np.argmax(proba_test, axis=1)
    else:
        print("[Surrogate] Using true labels as labels (direct mode).")
        y_train_tree = y_train
        y_test_ref = y_test

    print("[Surrogate] Training decision tree...")
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X_train_tab, y_train_tree)

    # Evaluate surrogate
    y_tree_test = tree.predict(X_test_tab)

    report = classification_report(
        y_test_ref,
        y_tree_test,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    bal_acc = balanced_accuracy_score(y_test_ref, y_tree_test)

    metrics = {
        "label_source": label_source,
        "max_depth": max_depth,
        "classification_report": report,
        "balanced_accuracy": float(bal_acc),
    }

    print("\n[Surrogate] Test performance (balanced accuracy):", bal_acc)

    # Feature importances plot (if output_dir is provided)
    if output_dir is not None:
        importances = tree.feature_importances_
        idx_sorted = np.argsort(importances)[::-1][: min(20, len(importances))]

        plt.figure(figsize=(9, 4))
        plt.bar(
            [stat_names[i] for i in idx_sorted],
            importances[idx_sorted],
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Feature importance")
        plt.title("Surrogate tree feature importances")
        plt.tight_layout()
        plt.savefig(output_dir / "surrogate_feature_importances.png", dpi=150)
        plt.close()

        # Export tree as text
        tree_rules = export_text(tree, feature_names=stat_names)
        with open(output_dir / "surrogate_tree.txt", "w") as f:
            f.write(tree_rules)

    return tree, stat_names, metrics


def train_xgb_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    label_encoder,
    use_class_weights: bool = True,
    max_class_weight: float = 10.0,
) -> Tuple[XGBClassifier, np.ndarray, float, float, Dict[int, float]]:
    n_classes = len(label_encoder.classes_)


    # Class weights
    if use_class_weights:
        classes = np.unique(y_train)

        class_weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )

        # cap weights a max_class_weight (p.ex. 50)
        if max_class_weight is not None:
            max_w = float(max_class_weight)
            class_weights_arr = np.minimum(class_weights_arr, max_w)

        # Diccionari {classe: pes}
        class_weights = {
            int(c): float(w) for c, w in zip(classes, class_weights_arr)
        }

        # Vector de sample_weight per mostra
        sample_weight = np.zeros_like(y_train, dtype=float)
        for c, w in class_weights.items():
            sample_weight[y_train == c] = w

    else:
        sample_weight = None
        class_weights = {int(c): 1.0 for c in range(n_classes)}

    # model XGBoost
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",  # binari i multiclasse
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
    )

    # Training
    t0 = time.perf_counter()
    fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    xgb.fit(X_train, y_train, **fit_kwargs)
    t1 = time.perf_counter()
    training_time_seconds = t1 - t0


    # TEST
    p0 = time.perf_counter()
    y_test_proba = xgb.predict_proba(X_test)
    p1 = time.perf_counter()
    total_inference_time_test = p1 - p0

    return xgb, y_test_proba, training_time_seconds, total_inference_time_test, class_weights


def summarize_windows_stats(
    X: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Summarise each window into simple statistics per feature.

    X : (n_windows, seq_len, n_features)

    For each original feature f, we compute:
        - mean over time
        - std over time
        - min over time
        - max over time

    This gives a tabular representation for decision trees.
    """
    n_windows, seq_len, n_features = X.shape
    if n_features != len(feature_names):
        raise ValueError("feature_names length does not match X.shape[-1].")

    stats = []
    stat_names = []

    for j, fname in enumerate(feature_names):
        x_f = X[:, :, j]  # (n_windows, seq_len)

        stats.append(x_f.mean(axis=1))
        stat_names.append(f"mean_{fname}")

        stats.append(x_f.std(axis=1))
        stat_names.append(f"std_{fname}")

        stats.append(x_f.min(axis=1))
        stat_names.append(f"min_{fname}")

        stats.append(x_f.max(axis=1))
        stat_names.append(f"max_{fname}")

    X_tab = np.column_stack(stats)  # (n_windows, n_stats)
    return X_tab, stat_names
