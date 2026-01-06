#!/usr/bin/env python
"""
Grid runner for NETWORK-modality training experiments.

- Ensures the project root is on sys.path so pipeline modules can be imported.
- Loads the preprocessed network dataset (CICEVSE2024_network_preprocessed.csv) and creates outputs_network/.
- Runs pipeline1 over a small configuration grid:
  - tasks: scenario, multiattack
  - model types: xgb, tcn, lstm
  - sequence lengths: [1, 2, 10, 20]
  - step sizes: 1 and seq_len (overlapping vs non-overlapping windows)
  - fixed fpr_target = 1e-3 (passed through to the pipeline, mainly relevant for binary-style operational eval).
- Prints an experiment name per run and wraps each call in try/except so failures don’t stop the full grid.

Entry point: main().
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    from pipelines_network.pipeline_trainingmodels_network import run_pipeline1
except Exception:
    from pipeline_trainingmodels_network import run_pipeline1


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    csv_path = project_root / "CICEVSE2024_Dataset" / "CICEVSE2024_network_preprocessed.csv"
    print(csv_path)

    base_output_root = project_root / "outputs_network"
    base_output_root.mkdir(parents=True, exist_ok=True)

    tasks = ["scenario", "multiattack"]
    model_types = ["xgb", "tcn", "lstm"]
    seq_lens = [1, 2, 10, 20]

    fpr_target = 1e-3

    for task in tasks:
        for model_type in model_types:
            for seq_len in seq_lens:
                for step in [1, seq_len]:
                    exp_name = f"{task}_{model_type}_seq{seq_len}_step{step}"
                    print(f"\n=== Running experiment: {exp_name} ===")

                    try:
                        run_pipeline1(
                            task=task,
                            csv_path=csv_path,
                            col_time="timestamp",
                            output_root=base_output_root,
                            model_type=model_type,
                            seq_len=seq_len,
                            step=step,
                            fpr_target=fpr_target,
                        )
                    except Exception as e:
                        print(
                            f"[runner][error] {exp_name} failed: "
                            f"{type(e).__name__}: {e}"
                        )
                        continue


if __name__ == "__main__":
    main()
