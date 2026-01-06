#!/usr/bin/env python
"""
Batch runner for HOST training experiments.

What it does:
- Loads the HOST kernel-events CSV.
- For each (task, model_type, seq_len, step) configuration, trains a model using
  TOP-K features defined in a precomputed feature_importances.json.
- Writes all artifacts (windows, model, encoders/scaler, metrics/config, plots)
  under outputs_host/<task>_<model>_seq<...>_step<...>_topK...
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root (TFM/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines_host.pipeline_trainingmodels_host import run_pipeline1_host  # noqa: E402


def main():
    project_root = Path(__file__).resolve().parents[1]

    # Input CSV
    csv_path = (
        project_root
        / "CICEVSE2024_Dataset"
        / "Host Events"
        / "EVSE-B-HPC-Kernel-Events-Combined.csv"
    )

    # Output root for all experiments
    base_output_root = project_root / "outputs_host"

    # Root containing feature_importances.json produced by the importance pipeline
    base_importance_root = project_root / "outputs_host" / "feature_selection"

    tasks = ["binary", "scenario", "multiattack"]
    model_types = ["xgb", "tcn", "lstm"]
    seq_lens = [2, 5, 10, 15, 20, 25, 30]

    # Subfolder names where importances live (fixed seq10_step1)
    importance_subdir_by_task = {
        "binary": "binary_seq10_step1",
        "scenario": "scenario_seq10_step1",
        "multiattack": "multiattack_seq10_step1",
    }

    # TOP-K per (task, model_type)
    k_by_task_model = {
        "binary": {"xgb": 70, "tcn": 70, "lstm": 70},
        "scenario": {"xgb": 70, "tcn": 70, "lstm": 70},
        "multiattack": {"xgb": 70, "tcn": 70, "lstm": 70},
    }

    base_output_root.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        for model_type in model_types:
            K = k_by_task_model.get(task, {}).get(model_type)
            if K is None:
                print(f"[SKIP] Missing K for task={task}, model={model_type}")
                continue

            subdir = importance_subdir_by_task[task]
            importance_dir = base_importance_root / model_type / subdir
            fi_json = importance_dir / "feature_importances.json"

            if not fi_json.exists():
                print(f"[WARN] Missing importances: {fi_json} (skip task/model)")
                continue

            for seq_len in seq_lens:
                for step in [1, seq_len]:
                    exp_name = f"{task}_{model_type}_seq{seq_len}_step{step}"
                    print(f"\n=== HOST TOP-K: {exp_name} | K={K} ===")
                    print(f"    importances: {fi_json}")

                    run_pipeline1_host(
                        task=task,
                        csv_path=csv_path,
                        output_root=base_output_root,
                        feature_importances_path=fi_json,
                        model_type=model_type,
                        seq_len=seq_len,
                        step=step,
                        K=K,
                    )


if __name__ == "__main__":
    main()
