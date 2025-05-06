from typing import Tuple, TypedDict

import lightning as L
import snntorch.functional as SF
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

from eeg_snn_encoder.encoders.base import SpikeEncoder
from eeg_snn_encoder.metrics import (
    spike_count_accuracy,
    spike_count_f1,
    spike_count_mse,
    spike_count_precision,
    spike_count_recall,
)
from eeg_snn_encoder.models.classifier import EEGSTFTSpikeClassifier, ModelConfig


class OptimizerConfig(TypedDict):
    lr: float
    weight_decay: float
    scheduler_factor: float
    scheduler_patience: int


class LitSeizureClassifier(L.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        spike_encoder: SpikeEncoder,
    ):
        super().__init__()
        self.model = EEGSTFTSpikeClassifier(config=model_config).to(self.device)
        self.optimizer_config = optimizer_config
        self.spike_encoder = spike_encoder

        self.criterion = SF.mse_count_loss()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def training_step(self, batch: DataLoader, batch_idx: int) -> torch.Tensor:
        data, targets = batch
        data, targets = data.to(self.device), targets.to(self.device)
        spike_train = self.spike_encoder.encode(data)
        spk_rec, _ = self.model(spike_train)

        loss = self.criterion(spk_rec, targets)
        accuracy = SF.accuracy_rate(spk_rec, targets)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: DataLoader, batch_idx: int) -> torch.Tensor:
        data, targets = batch
        data, targets = data.to(self.device), targets.to(self.device)
        spike_train = self.spike_encoder.encode(data)
        spk_rec, _ = self.model(spike_train)

        loss = self.criterion(spk_rec, targets)
        accuracy = SF.accuracy_rate(spk_rec, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: DataLoader, batch_idx: int) -> torch.Tensor:
        data, targets = batch
        data, targets = data.to(self.device), targets.to(self.device)
        spike_train = self.spike_encoder.encode(data)
        spk_rec, _ = self.model(spike_train)

        loss = self.criterion(spk_rec, targets)
        accuracy = SF.accuracy_rate(spk_rec, targets)

        precision = spike_count_precision(spk_rec, targets)
        recall = spike_count_recall(spk_rec, targets)
        f1 = spike_count_f1(spk_rec, targets)
        mse = spike_count_mse(spk_rec, targets)
        total_input_spikes = spike_train.sum()

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test_total_spikes", total_input_spikes, on_step=False, on_epoch=True, prog_bar=True
        )

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