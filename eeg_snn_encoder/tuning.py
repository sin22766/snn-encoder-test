import os
from typing import Callable, List, TypedDict

import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import optuna
from optuna_integration import PyTorchLightningPruningCallback

from eeg_snn_encoder.callback import TrackBestMetric
from eeg_snn_encoder.encoders import (
    BSAEncoder,
    BurstEncoder,
    PhaseEncoder,
    PoissonEncoder,
    SpikeEncoder,
    StepForwardEncoder,
    TBREncoder,
)
from eeg_snn_encoder.models.classifier import ModelConfig
from eeg_snn_encoder.models.lightning import LitSeizureClassifier, OptimizerConfig


class TrialFilter(TypedDict):
    """
    A dictionary to define the filter function for trials.

    Attributes
    ----------
    name : str
        The name of the study.
    filter_fn : callable
        A function that takes a trial and returns True if it should be included.
    """

    name: str
    filter_fn: callable


def filter_and_sort_trials(study_configs: List[TrialFilter], storage_url=None):
    """
    Load, filter, and sort trials from multiple Optuna studies based on custom filters.

    Parameters
    ----------
    study_configs : list of dict
        List of dictionaries containing study configurations.
        Each dict should have 'name' and 'filter_fn' keys.
    storage_url : str, optional
        The Optuna storage URL. If None, uses os.environ["OPTUNA_CONN_STRING"].

    Returns
    -------
    list
        Sorted filtered trials from all studies.

    Raises
    ------
    ValueError
        If no storage URL is provided and OPTUNA_CONN_STRING environment variable is not set.
    """

    # Use provided storage URL or fall back to environment variable
    storage = storage_url or os.environ.get("OPTUNA_CONN_STRING")
    if not storage:
        raise ValueError(
            "No storage URL provided and OPTUNA_CONN_STRING environment variable not set"
        )

    qualifying_trials = []

    for config in study_configs:
        study_name = config["name"]
        filter_fn = config["filter_fn"]

        print(f"Loading trials from {study_name}")
        current_study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )

        completed_trials = current_study.get_trials(
            False, states=(optuna.trial.TrialState.COMPLETE,)
        )
        qualifying_study_trials = list(filter(filter_fn, completed_trials))
        ranked_trials = sorted(qualifying_study_trials, key=lambda t: t.value)

        qualifying_trials.extend(ranked_trials)
        print(f"Found {len(ranked_trials)} qualifying trials in {study_name}")

    return qualifying_trials


def burst_encoder_tuning(trial: optuna.Trial) -> SpikeEncoder:
    """
    Create a BurstEncoder with parameters tuned by Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial for hyperparameter optimization.

    Returns
    -------
    BurstEncoder
        Configured BurstEncoder instance.
    """
    max_window = trial.suggest_int("burst_max_window", 4, 8)
    n_max = trial.suggest_int("burst_n_max", 1, max_window)
    t_max = trial.suggest_int("burst_t_max", 0, max_window // n_max)
    t_min = trial.suggest_int("burst_t_min", 0, t_max)

    encoder_params = {
        "max_window": max_window,
        "n_max": n_max,
        "t_max": t_max,
        "t_min": t_min,
    }

    return BurstEncoder(**encoder_params)


def phase_encoder_tuning(trial: optuna.Trial) -> SpikeEncoder:
    """
    Create a PhaseEncoder with parameters tuned by Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial for hyperparameter optimization.

    Returns
    -------
    PhaseEncoder
        Configured PhaseEncoder instance.
    """
    encoder_params = {
        "phase_window": trial.suggest_int("phase_window", 1, 4),
    }

    return PhaseEncoder(**encoder_params)


def poisson_encoder_tuning(trial: optuna.Trial) -> SpikeEncoder:
    """
    Create a PoissonEncoder with parameters tuned by Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial for hyperparameter optimization.

    Returns
    -------
    PoissonEncoder
        Configured PoissonEncoder instance.
    """
    encoder_params = {
        "interval_freq": trial.suggest_int("poisson_interval_freq", 1, 8),
        "random_seed": 47,
    }

    return PoissonEncoder(**encoder_params)


def bsa_encoder_tuning(trial: optuna.Trial) -> SpikeEncoder:
    """
    Create a BSAEncoder with parameters tuned by Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial for hyperparameter optimization.

    Returns
    -------
    BSAEncoder
        Configured BSAEncoder instance.
    """
    encoder_params = {
        "win_size": trial.suggest_int("bsa_win_size", 1, 16),
        "cutoff": trial.suggest_float("bsa_cutoff", 0.01, 1),
        "threshold": trial.suggest_float("bsa_threshold", 0.01, 4),
    }

    return BSAEncoder(**encoder_params)


def step_forward_encoder_tuning(trial: optuna.Trial) -> SpikeEncoder:
    """
    Create a StepForwardEncoder with parameters tuned by Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial for hyperparameter optimization.

    Returns
    -------
    StepForwardEncoder
        Configured StepForwardEncoder instance.
    """
    encoder_params = {
        "threshold": trial.suggest_float("sf_threshold", 0.01, 4),
    }

    return StepForwardEncoder(**encoder_params)


def tbr_encoder_tuning(trial: optuna.Trial) -> SpikeEncoder:
    """
    Create a TBREncoder with parameters tuned by Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial for hyperparameter optimization.

    Returns
    -------
    TBREncoder
        Configured TBREncoder instance.
    """
    encoder_params = {
        "threshold": trial.suggest_float("tbr_threshold", 0.01, 4),
    }

    return TBREncoder(**encoder_params)


# Dictionary mapping encoder types to their respective tuning functions
ENCODER_TUNING_FUNCTIONS = {
    "be": burst_encoder_tuning,
    "pe": phase_encoder_tuning,
    "poisson": poisson_encoder_tuning,
    "bsa": bsa_encoder_tuning,
    "sf": step_forward_encoder_tuning,
    "tbr": tbr_encoder_tuning,
}


def create_objective(
    encoder_type: str,
    datamodule: LightningDataModule,
    moniter_metric: str = "val_mse",
    moniter_mode: str = "min",
    model_config: ModelConfig = None,
    optimizer_config: OptimizerConfig = None,
) -> Callable[[optuna.Trial], float]:
    """
    Create an objective function for Optuna based on the encoder type.

    Parameters
    ----------
    encoder_type : str
        The type of encoder to tune.
    datamodule : LightningDataModule
        The data module for training and validation.
    moniter_metric : str, optional
        The metric to monitor for optimization (default is "val_mse").
    moniter_mode : str, optional
        The mode for the monitored metric, "min" or "max" (default is "min").
    model_config : ModelConfig, optional
        Model configuration dictionary. If None, will be suggested by Optuna.
    optimizer_config : OptimizerConfig, optional
        Optimizer configuration dictionary. If None, will be suggested by Optuna.

    Returns
    -------
    callable
        The objective function for Optuna.
    """
    if encoder_type not in ENCODER_TUNING_FUNCTIONS:
        valid_types = list(ENCODER_TUNING_FUNCTIONS.keys())
        raise ValueError(f"Unsupported encoder type: {encoder_type}. Choose from: {valid_types}")

    def objective(trial: optuna.Trial) -> float:
        if model_config is None:
            model_params: ModelConfig = {
                "threshold": trial.suggest_float("threshold", 0.01, 0.5),
                "slope": trial.suggest_float("slope", 1.0, 20.0),
                "beta": trial.suggest_float("beta", 0.1, 0.99),
                "dropout_rate1": trial.suggest_float("dropout_rate1", 0.1, 0.99),
                "dropout_rate2": trial.suggest_float("dropout_rate2", 0.1, 0.99),
            }
        else:
            model_params = model_config

        if optimizer_config is None:
            optimizer_params: OptimizerConfig = {
                "lr": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
                "scheduler_factor": trial.suggest_float("scheduler_factor", 0.1, 0.99),
                "scheduler_patience": trial.suggest_int("scheduler_patience", 1, 10),
            }
        else:
            optimizer_params = optimizer_config

        spike_encoder = ENCODER_TUNING_FUNCTIONS[encoder_type](trial)

        lit_model = LitSeizureClassifier(
            model_config=model_params,
            optimizer_config=optimizer_params,
            spike_encoder=spike_encoder,
        )

        tracker = TrackBestMetric(
            monitor=moniter_metric,
            mode=moniter_mode,
        )

        trainer = pl.Trainer(
            max_epochs=20,
            accelerator="auto",
            devices="auto",
            strategy="auto",
            enable_model_summary=False,
            enable_checkpointing=False,
            callbacks=[
                tracker,
                PyTorchLightningPruningCallback(trial, monitor=moniter_metric),
                EarlyStopping(monitor="val_loss", mode="min", patience=5),
            ],
            logger=False,
        )

        trainer.fit(lit_model, datamodule=datamodule)

        return tracker.best_metric

    return objective
