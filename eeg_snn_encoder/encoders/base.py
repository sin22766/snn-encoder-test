from abc import ABC, abstractmethod
from typing import Optional

import torch


class SpikeEncoder(ABC):
    """
    Base class for spike encoders.

    This abstract class provides the interface and common functionality
    for encoding continuous values into spike trains and decoding them back.
    """

    def __init__(self):
        """Initialize the SpikeEncoder."""
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous values into spikes.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).

        Returns
        -------
        torch.Tensor
            Encoded spike train.
        """
        pass

    @abstractmethod
    def decode(self, x: torch.Tensor, decode_params=None) -> torch.Tensor:
        """
        Decode spike trains back to continuous values.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).

        Returns
        -------
        torch.Tensor
            Decoded continuous values.
        """
        pass

    def get_decode_params(self, x: torch.Tensor) -> Optional[dict]:
        """
        Get parameters for decoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).

        Returns
        -------
        Optional[dict]
            Parameters needed for decoding.
        """
        return None
