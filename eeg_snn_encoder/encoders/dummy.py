import torch

from .base import SpikeEncoder


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
