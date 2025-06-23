Here’s a revised and polished version of your README with improved clarity, formatting, and structure:

---

# Evaluation and Comparison of EEG Encoding Methods for Seizure Classification in Spiking Neural Networks

This is a bachelor thesis project focused on evaluating and comparing EEG encoding methods for seizure classification using spiking neural networks (SNNs). It includes utilities for dataset processing and Jupyter notebooks for experimentation and visualization.

---

## 📊 Dataset

The dataset used is the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/), a publicly available EEG dataset containing recordings from pediatric epilepsy patients.

After downloading the dataset from PhysioNet, place it in the following directory:

```
data/raw/CHB-MIT/chb01, chb02, ..., chb24
```

Each subdirectory (e.g. `chb01`) contains EEG recordings in `.edf` format.

---

## ⚙️ Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management and environment setup.

### 1. Create a virtual environment

```bash
uv venv
```

### 2. Install the project in editable mode

```bash
uv pip install -e .
```

This installs the project with all required dependencies and allows for live editing of the source code.

---

Perfect — here’s an updated version of the **Environment Variables** section that specifically includes `OPTUNA_CONN_STRING`, and references the `.env.example` file correctly.

---

## ⚡ Environment Variables

This project requires one environment variable for hyperparameter tuning:

```env
OPTUNA_CONN_STRING
```

This is used by [Optuna](https://optuna.org/) to connect to a persistent database for storing trials (e.g., PostgreSQL or SQLite).

You can create a `.env` file at the root of the project, or copy the included example:

```bash
cp .env.example .env
```

Example `.env` content:

```ini
# PostgreSQL (recommended for parallel/multi-process tuning)
OPTUNA_CONN_STRING="postgresql+psycopg2://user:password@localhost/dbname"

# or SQLite (for quick local testing)
OPTUNA_CONN_STRING="sqlite:///optuna.db"
```

---