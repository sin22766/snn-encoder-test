import gc

from loguru import logger
import papermill as pm

from eeg_snn_encoder.config import PROJ_ROOT

notebooks = PROJ_ROOT / "notebooks" / "5-model-cv-All.ipynb"

params_list = [
    {"tuning_encoder": "pe"},
    {"tuning_encoder": "be"},
    {"tuning_encoder": "poisson"},
    {"tuning_encoder": "bsa"},
    {"tuning_encoder": "sf"},
    {"tuning_encoder": "tbr"},
    {"tuning_encoder": "dummy"},
]


for param in params_list:
    output_path = notebooks.parent / f"{notebooks.stem}-output.ipynb"
    logger.info(f"Running {notebooks}...")

    pm.execute_notebook(
        notebooks,
        output_path,
        kernel_name="snn-eeg",
        parameters=param
    )

    logger.info(f"Finished {notebooks}.")

    # Manual memory cleanup
    gc.collect()