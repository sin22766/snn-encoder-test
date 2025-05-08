import gc
from pathlib import Path

import papermill as pm

from eeg_snn_encoder.config import PROJ_ROOT

notebooks = [
    PROJ_ROOT / "notebooks" / "3-model-tuning-Dummy.ipynb",
    PROJ_ROOT / "notebooks" / "3-model-tuning-SF.ipynb",
    PROJ_ROOT / "notebooks" / "3-model-tuning-PE.ipynb",
]

def run_notebook(notebook_path: Path):
    output_path = notebook_path.parent / f"output-{notebook_path.stem}.ipynb"
    print(f"Running {notebook_path}...")
    pm.execute_notebook(
        notebook_path,
        output_path,
        kernel_name="python3"
    )
    print(f"Finished {notebook_path}.")


for notebook in notebooks:
    run_notebook(notebook)

    # Manual memory cleanup
    gc.collect()