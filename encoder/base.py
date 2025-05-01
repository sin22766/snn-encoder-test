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

class DummyEncoder(SpikeEncoder):
    """
    Dummy encoder that does not perform any encoding or decoding.
    
    This class is used for testing purpose and does not modify the input data.
    """

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous values into spikes (no-op).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).
            
        Returns
        -------
        torch.Tensor
            Encoded spike train (same as input).
        """
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode spike trains back to continuous values (no-op).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).
            
        Returns
        -------
        torch.Tensor
            Decoded continuous values (same as input).
        """
        return x