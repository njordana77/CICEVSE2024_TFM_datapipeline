# CICEVSE2024 TFM Pipeline

Training and evaluation pipelines for the **CICEVSE2024** dataset, organized by modality:

- **Power consumption** (`pipelines_power/`)
- **Host events** (`pipelines_host/`)
- **Network traffic** (`pipelines_network/`)

This repository is intentionally **script-based**: you clone it, install the dependencies, place the dataset in the expected folder, and run the provided `run_pipeline_train_*.py` entry scripts.

---

## Repository layout

```
CICEVSE2024_TFM_datapipeline/
  func_aux/                 # helper functions
  pipelines_power/          # power modality experiments
  pipelines_host/           # host modality experiments
  pipelines_network/        # network modality experiments
  CICEVSE2024_Dataset/      # (NOT included) put the dataset here
  outputs_power/            # created automatically
  outputs_host/             # created automatically
  outputs_network/          # created automatically
```

> `CICEVSE2024_Dataset/` is ignored by git on purpose (large data files).

---

## Quickstart (pip + venv)

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

### TensorFlow note (CPU vs GPU)
- If you want a straightforward install, use **`tensorflow`** (default in `requirements.txt`).
- If you want **CPU-only** (often simpler on servers/CI), replace `tensorflow` with **`tensorflow-cpu`** in `requirements.txt`.

---

## Quickstart (conda)

```bash
conda env create -f environment.min.yml
conda activate cicevse2024-tfm
```

---

## Dataset: expected paths

Place the dataset files so the runner scripts can find them:

### Power modality
```
CICEVSE2024_Dataset/Power Consumption/EVSE-B-PowerCombined.csv
```

### Host modality
```
CICEVSE2024_Dataset/Host Events/EVSE-B-HPC-Kernel-Events-Combined.csv
```

### Network modality
```
CICEVSE2024_Dataset/CICEVSE2024_network_preprocessed.csv
```

If your filenames/folders differ, edit the `csv_path = ...` line inside the corresponding runner script:

- `pipelines_power/run_pipeline_train_power.py`
- `pipelines_host/run_pipeline_train_host.py`
- `pipelines_network/run_pipeline_train_network.py`

---

## Run the pipelines

### Power
```bash
python pipelines_power/run_pipeline_train_power.py
```

### Host
```bash
python pipelines_host/run_pipeline_train_host.py
```

### Network
```bash
python pipelines_network/run_pipeline_train_network.py
```

Outputs are written under:
- `outputs_power/`
- `outputs_host/`
- `outputs_network/`

---

## Reproducibility tips

- The repo includes an `environment.yml` (full, pinned). It can be useful if it matches your OS, but it may be **too strict** for other machines.
- For portability, prefer the provided **`environment.min.yml`** or **`requirements.txt`**.

---

## Editable install (optional)

You can run everything without installing the project. If you prefer an editable install (useful for notebooks or external scripts), run:

```bash
pip install -e .
```

This keeps your current imports (e.g. `from func_aux...`) working.

---

## License
MIT (see `LICENSE`).
