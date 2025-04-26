from typing import Optional

import lightning as L
from torch.utils.data import DataLoader, random_split

from dataset.seizure import CHBMITDataset


class CHBMITDataModule(L.LightningDataModule):
    def __init__(self, dataset: CHBMITDataset, batch_size: int = 32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        # Split the dataset into train, validation, and test sets
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            self.dataset, [0.7, 0.15, 0.15]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, shuffle=True, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, shuffle=True, num_workers=10)