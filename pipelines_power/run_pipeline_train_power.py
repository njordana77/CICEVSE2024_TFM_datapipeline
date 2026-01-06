# pipelines_power/run_pipeline_train_power.py

#!/usr/bin/env python

"""
Experiment runner for EVSE power-model training (pipeline1 grid search).

This script:
- Ensures the project root (TFM/) is on sys.path so local modules can be imported.
- Defines the power dataset CSV path and a shared output directory (outputs_power/).
- Runs pipeline1 over a small grid of configurations:
  - tasks: binary, scenario, multiattack
  - model types: xgb, tcn, lstm
  - sequence lengths: [2, 5, 10, 15, 20, 25, 30]
  - step sizes: 1 and seq_len (non-overlapping windows)
  - fixed operational target for binary runs: fpr_target = 1e-3
- For each configuration, prints the experiment name and invokes run_pipeline1(...),
  which trains the model and writes artifacts/metrics under outputs_power/<run_name>/.

Entry point: main().
"""


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # TFM/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from pathlib import Path
from pipelines_power.pipeline_trainingmodels_power import run_pipeline1


def main():
    # Path to  CSV
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    csv_path = PROJECT_ROOT / "CICEVSE2024_Dataset" / "Power Consumption" / "EVSE-B-PowerCombined.csv"

    # Root folder where all experiment outputs will be stored
    base_output_root =  PROJECT_ROOT / "outputs_power"

    # Grid of configs
    tasks = ["binary", "scenario", "multiattack"]
    model_types = ["xgb", "tcn", "lstm"]
    seq_lens = [2, 5, 10, 15, 20, 25, 30]

    # Common hyperparameters
    fpr_target = 1e-3

    for task in tasks:
        for model_type in model_types:
            for seq_len in seq_lens:
                step_list = [1, seq_len]
                for step in step_list:
                    exp_name = f"{task}_{model_type}_seq{seq_len}_step{step}"
                    print(f"\n=== Running experiment: {exp_name} ===")

                    # Create a specific output directory per experiment
                    output_root = base_output_root
                    output_root.mkdir(parents=True, exist_ok=True)

                    run_pipeline1(
                        task=task,
                        csv_path=csv_path,
                        col_time="time",
                        output_root=output_root,
                        model_type=model_type,
                        seq_len=seq_len,
                        step=step,
                        fpr_target=fpr_target,
                    )


if __name__ == "__main__":
    main()
