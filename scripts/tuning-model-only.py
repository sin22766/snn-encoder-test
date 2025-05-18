import gc

from loguru import logger
import papermill as pm

from eeg_snn_encoder.config import PROJ_ROOT

notebooks = PROJ_ROOT / "notebooks" / "3-model-tuning-All-model-only.ipynb"

params_list = [
    # {"tuning_list": ["be"]},
    {"tuning_list": ["pe"]},
    {"tuning_list": ["poisson"]},
    {"tuning_list": ["bsa"]},
    {"tuning_list": ["sf"]},
    {"tuning_list": ["tbr"]},
]


for param in params_list:
    output_path = notebooks.parent / f"{notebooks.stem}-output.ipynb"
    logger.info(f"Running {notebooks}...")

    pm.execute_notebook(
        notebooks,
        output_path,
        kernel_name="snn-encoder-test",
        parameters=param
    )

    logger.info(f"Finished {notebooks}.")

    # Manual memory cleanup
    gc.collect()