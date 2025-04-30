from typing import TypedDict

import lightning as L
import snntorch.functional as SF
import torch
from torch.optim import AdamW, lr_scheduler

from encoder.base import SpikeEncoder
from model.classifier import EEGSpikeClassifier, ModelConfig
from utils.preprocess import VectorizeSTFT


class OptimizerConfig(TypedDict):
    lr: float
    weight_decay: float
    scheduler_factor: float
    scheduler_patience: int


class LitSTFTSeizureClassifier(L.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        spike_encoder: SpikeEncoder,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["spike_encoder"])
        self.model = EEGSpikeClassifier(model_config)
        self.optimizer_config = optimizer_config
        self.spike_encoder = spike_encoder

        self.criterion = SF.mse_count_loss()
        self.accuracy = SF.accuracy_rate

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def preprocess_data(self, data: torch.Tensor) -> torch.Tensor:
        """Preprocess the input data by scaling and converting to spikes."""
        scaled_data = VectorizeSTFT(data)

        # Convert the data to non complex values
        scaled_data = torch.abs(scaled_data)

        spikes = self.spike_encoder.encode(scaled_data)

        return spikes

    def training_step(self, batch, batch_idx):
        data, targets = batch
        data_spike = self.preprocess_data(data)

        # Predict the spikes using the model
        spk_rec, _ = self.model(data_spike)

        # Compute the loss
        loss = self.criterion(spk_rec, targets)
        accuracy = self.accuracy(spk_rec, targets)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        data_spike = self.preprocess_data(data)

        # Predict the spikes using the model
        spk_rec, _ = self.model(data_spike)

        # Compute the loss
        loss = self.criterion(spk_rec, targets)
        accuracy = self.accuracy(spk_rec, targets)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        data, targets = batch
        data_spike = self.preprocess_data(data)

        # Predict the spikes using the model
        spk_rec, _ = self.model(data_spike)

        # Compute the loss
        loss = self.criterion(spk_rec, targets)
        accuracy = self.accuracy(spk_rec, targets)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.optimizer_config["lr"],
            betas=(0.9, 0.999),
            weight_decay=self.optimizer_config["weight_decay"],
        )

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.optimizer_config["scheduler_factor"],
            patience=self.optimizer_config["scheduler_patience"],
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class LitSTFTPreprocessedSeizureClassifier(L.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        spike_encoder: SpikeEncoder,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["spike_encoder"])
        self.model = EEGSpikeClassifier(model_config)
        self.optimizer_config = optimizer_config
        self.spike_encoder = spike_encoder

        self.criterion = SF.mse_count_loss()
        self.accuracy = SF.accuracy_rate

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data, targets = batch
        data_spike = self.spike_encoder.encode(data)

        # Predict the spikes using the model
        spk_rec, _ = self.model(data_spike)

        # Compute the loss
        loss = self.criterion(spk_rec, targets)
        accuracy = self.accuracy(spk_rec, targets)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        data_spike = self.spike_encoder.encode(data)

        # Predict the spikes using the model
        spk_rec, _ = self.model(data_spike)

        # Compute the loss
        loss = self.criterion(spk_rec, targets)
        accuracy = self.accuracy(spk_rec, targets)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        data, targets = batch
        data_spike = self.spike_encoder.encode(data)

        # Predict the spikes using the model
        spk_rec, _ = self.model(data_spike)

        # Compute the loss
        loss = self.criterion(spk_rec, targets)
        accuracy = self.accuracy(spk_rec, targets)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.optimizer_config["lr"],
            betas=(0.9, 0.999),
            weight_decay=self.optimizer_config["weight_decay"],
        )

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.optimizer_config["scheduler_factor"],
            patience=self.optimizer_config["scheduler_patience"],
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
