from pathlib import Path
from typing import Optional

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from eeg_snn_encoder.config import PROCESSED_DATA_DIR


class CHBMITDataset(Dataset):
    """
    Dataset for EEG data with seizure/non-seizure labels.

    This class handles EEG data loaded from HDF5 files. Each sample is a 2D array
    representing an 8-second EEG window with 22 channels and 2048 time points
    (sampled at 256 Hz). The dataset includes both ictal (seizure) and interictal
    (non-seizure) segments.

    Parameters
    ----------
    data_path : Path, optional
        Path to the HDF5 file containing preprocessed EEG data. Defaults to ``PROCESSED_DATA_DIR / "windowed.h5"``.
        The file is expected to contain EEG data and corresponding labels with the following structure:
        - "data": EEG recordings of shape (N, 22, 2048) or (N, 22, 129, 65) if preprocessed with STFT
        - "labels": Binary labels of shape (N, 1), where 1 indicates ictal (seizure) and 0 indicates interictal (non-seizure)
        - "info": Metadata (not used)
        - "channels": Channel names (not used)
    """

    def __init__(self, data_path: Path = PROCESSED_DATA_DIR / "windowed.h5"):
        data_file = h5py.File(data_path, "r")

        self.data = torch.tensor(np.array(data_file["data"]), dtype=torch.float32)
        self.labels = torch.tensor(np.array(data_file["labels"]), dtype=torch.float32)

        data_file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


class CHBMITDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for the CHB-MIT EEG dataset.

    This module wraps around a preprocessed CHBMITDataset and handles data loading
    for training, validation, and testing. It splits the full dataset into three
    subsets (70% train, 15% validation, 15% test) and provides corresponding dataloaders.

    Parameters
    ----------
    dataset : CHBMITDataset
        The complete dataset containing EEG data and labels.

    batch_size : int, optional
        Number of samples per batch to load. Defaults to 32.

    worker : int, optional
        Number of subprocesses to use for data loading. Defaults to 8.
    """

    def __init__(self, dataset: CHBMITDataset, batch_size: int = 32, worker=8):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.worker = worker

    def setup(self, stage: Optional[str] = None) -> None:
        # Split the dataset into train, validation, and test sets
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [0.7, 0.15, 0.15]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=self.worker)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, num_workers=self.worker)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, num_workers=self.worker)
