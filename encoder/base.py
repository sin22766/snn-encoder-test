from abc import ABC, abstractmethod

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

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor to the range [0, 1].
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be normalized.
            
        Returns
        -------
        torch.Tensor
            Normalized tensor with values in range [0, 1].
        """
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values

        diff = x_max - x_min
        diff[diff == 0] = 1

        return (x - x_min) / diff

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
    def decode(self, x: torch.Tensor) -> torch.Tensor:
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