import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class CHBMITDataset(Dataset):
    """
    PyTorch Dataset for EEG data with seizure/non-seizure labels.
    Can load data directly from HDF5 files that has the preprocessed EEG data.
    The dataset contains two types of data: ictal (seizure) and interictal (non-seizure).
    Each data point is a 2D tensor of shape (22, 2048), where 22 is the number of EEG channels
    and 2048 is the number of time points in 256Hz sample rate 8 seconds records.

    Attributes:
        data (torch.Tensor): Combined EEG data
        labels (torch.Tensor): Binary labels (1 for ictal, 0 for interictal)
    """

    def __init__(
        self, data_dir, ictal_filename="ictal.h5", interictal_filename="interictal.h5"
    ):
        """
        Initialize the dataset with ictal and interictal data loaded from files.
        The data is assumed to be in HDF5 format with the following structure:
        - data: EEG data of shape (num_samples, 22, 2048)
        - info: Metadata about the data (not used in this dataset)
        - channels: Channel names (not used in this dataset)

        Parameters:
            data_dir (str): Directory containing HDF5 data files
            ictal_filename (str, optional): Filename for ictal data
            interictal_filename (str, optional): Filename for interictal data
        """

        ictal_path = os.path.join(data_dir, ictal_filename)
        interictal_path = os.path.join(data_dir, interictal_filename)

        ictal_file = h5py.File(ictal_path, "r")
        interictal_file = h5py.File(interictal_path, "r")

        ictal_data = torch.tensor(np.array(ictal_file["data"]), dtype=torch.float32)
        interictal_data = torch.tensor(
            np.array(interictal_file["data"]), dtype=torch.float32
        )

        # Ensure the data is converted to tensors
        self.data = torch.cat([ictal_data, interictal_data])
        # Labels for ictal and interictal data
        self.labels = torch.cat(
            [
                torch.ones(len(ictal_data)),  # Ictal = 1
                torch.zeros(len(interictal_data)),  # Interictal = 0
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_raw = self.data[idx]  # EEG data of shape (22, 2048)
        label = self.labels[idx].bool()  # Label: 0 (interictal) or 1 (ictal)
        return eeg_raw, label


class CHBMITPreprocessedDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed data and binary labels from an HDF5 file.

    The HDF5 file should contain:
    - 'data': array-like, shape (n_samples, ...)
    - 'labels': array-like, shape (n_samples,)
    """

    def __init__(self, data_file: str) -> None:
        """
        Initialize the dataset by loading data and labels from an HDF5 file.

        Parameters
        ----------
        data_file : str
            Path to the HDF5 file containing 'data' and 'labels' datasets.
        """
        data = h5py.File(data_file, "r")
        self.data = torch.tensor(np.array(data["data"]), dtype=torch.float32)
        self.labels = torch.tensor(np.array(data["labels"]), dtype=torch.float32)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the sample and label at the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple of torch.Tensor
            A tuple (data, label) where:
            - data is the input features (torch.float32)
            - label is the binary label converted to a boolean tensor
        """
        data = self.data[idx]
        label = self.labels[idx].bool()
        return data, label