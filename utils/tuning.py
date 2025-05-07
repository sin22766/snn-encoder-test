from typing import Callable

import lightning.pytorch as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from dataset import CHBMITPreprocessedDataset
from dataset.lightning import CHBMITPreprocessedDataModule
from encoder.base import DummyEncoder, SpikeEncoder
from encoder.deconv import BSAEncoder
from encoder.global_refer import PhaseEncoder, PhaseEncoderExpand
from encoder.latency import BurstEncoder, BurstEncoderExpand
from encoder.rate import PoissonEncoder, PoissonEncoderExpand
from encoder.temporal import StepForwardEncoder, TBREncoder
from model.classifier import ModelConfig
from model.lightning import LitSTFTPreprocessedSeizureClassifier, OptimizerConfig

dataset = CHBMITPreprocessedDataset("./CHB-MIT/processed_data.h5")
datamodule = CHBMITPreprocessedDataModule(dataset, batch_size=32)


def create_objective(
    create_encoder: Callable[[optuna.Trial], SpikeEncoder],
) -> Callable[[optuna.Trial], float]:
    """
    Create an objective function for Optuna to optimize.

    Args:
        encoder: A callable that takes an Optuna trial and returns a SpikeEncoder.

    Returns:
        An objective function for Optuna.
    """

    def objective(trial: optuna.Trial) -> float:
        model_params: ModelConfig = {
            "threshold": trial.suggest_float("threshold", 0.01, 0.5),
            "slope": trial.suggest_float("slope", 1.0, 20.0),
            "beta": trial.suggest_float("beta", 0.1, 0.99),
            "dropout_rate1": trial.suggest_float("dropout_rate1", 0.1, 0.99),
            "dropout_rate2": trial.suggest_float("dropout_rate2", 0.1, 0.99),
        }

        optimizer_params: OptimizerConfig = {
            "lr": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
            "scheduler_factor": trial.suggest_float("scheduler_factor", 0.1, 0.99),
            "scheduler_patience": trial.suggest_int("scheduler_patience", 1, 10),
        }

        encoder = create_encoder(trial)

        model = LitSTFTPreprocessedSeizureClassifier(
            model_config=model_params,
            optimizer_config=optimizer_params,
            spike_encoder=encoder,
        )

        # Create the trainer
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="gpu",
            devices=1,
            logger=False,
            callbacks=[
                PyTorchLightningPruningCallback(trial, monitor="val_mse"),
            ],
        )

        # Train the model
        trainer.fit(model, datamodule=datamodule)

        # Return the validation loss as the objective value
        return trainer.callback_metrics["val_loss"].item()

    return objective


def create_dummy_encoder(trial: optuna.Trial) -> SpikeEncoder:
    return DummyEncoder()


def create_poisson_encoder(trial: optuna.Trial) -> SpikeEncoder:
    params = {
        "interval_freq": trial.suggest_int("encoder_interval_freq", 1, 16),
        "normalize": False,
    }
    return PoissonEncoder(**params)


def create_sf_encoder(trial: optuna.Trial) -> SpikeEncoder:
    params = {
        "threshold": trial.suggest_float("encoder_threshold", 0.1, 0.99),
        "normalize": False,
    }
    return StepForwardEncoder(**params)


def create_tbr_encoder(trial: optuna.Trial) -> SpikeEncoder:
    params = {
        "threshold": trial.suggest_float("encoder_threshold", 0.1, 0.99),
        "normalize": False,
    }
    return TBREncoder(**params)


def create_bsa_encoder(trial: optuna.Trial) -> SpikeEncoder:
    params = {
        "win_size": trial.suggest_int("encoder_win_size", 1, 16),
        "cutoff": trial.suggest_float("encoder_cutoff", 0.01, 0.99),
        "threshold": trial.suggest_float("encoder_threshold", 0.1, 0.99),
        "normalize": False,
    }
    return BSAEncoder(**params)

def create_phase_encoder(trial: optuna.Trial) -> SpikeEncoder:
    params = {
        "phase_window": trial.suggest_int("encoder_windows", 1, 16),
        "normalize": False,
    }
    return PhaseEncoder(**params)

def create_burst_encoder(trial: optuna.Trial) -> SpikeEncoder:
    max_window = trial.suggest_int("encoder_max_window", 1, 16)
    n_max = trial.suggest_int("encoder_n_max", 1, max_window)
    t_min = trial.suggest_int("encoder_t_min", 0, max_window)
    t_max = trial.suggest_int("encoder_t_max", t_min, max_window)

    params = {
        "max_window": max_window,
        "n_max": n_max,
        "t_min": t_min,
        "t_max": t_max,
        "normalize": False,
    }
    return BurstEncoder(**params)

def create_expand_poisson_encoder(trial: optuna.Trial) -> SpikeEncoder:
    params = {
        "interval_freq": trial.suggest_int("encoder_interval_freq", 1, 8),
        "normalize": False,
    }
    return PoissonEncoderExpand(**params)

def create_expand_phase_encoder(trial: optuna.Trial) -> SpikeEncoder:
    params = {
        "phase_window": trial.suggest_int("encoder_windows", 1, 8),
        "normalize": False,
    }
    return PhaseEncoderExpand(**params)

def create_expand_burst_encoder(trial: optuna.Trial) -> SpikeEncoder:
    max_window = trial.suggest_int("encoder_max_window", 1, 8)
    n_max = trial.suggest_int("encoder_n_max", 1, max_window)
    t_min = trial.suggest_int("encoder_t_min", 0, max_window)
    t_max = trial.suggest_int("encoder_t_max", t_min, max_window)

    params = {
        "max_window": max_window,
        "n_max": n_max,
        "t_min": t_min,
        "t_max": t_max,
        "normalize": False,
    }
    return BurstEncoderExpand(**params)