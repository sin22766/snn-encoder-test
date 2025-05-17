from typing import Optional, TypedDict

import torch

from .base import SpikeEncoder


class SFDecodeParams(TypedDict):
    """
    Type definition for the parameters needed for decoding.

    Parameters
    ----------
    base : torch.Tensor
        Initial baseline values for decoding.
    """

    base: torch.Tensor


class StepForwardEncoder(SpikeEncoder):
    """
    Step Forward Encoder for converting continuous values to spike trains.

    This encoder generates spikes when the input signal crosses a threshold
    relative to an adaptive baseline that changes with each spike.

    Parameters
    ----------
    threshold : float, optional
        Threshold for spike generation used to modify the baseline on each timestep.
        Default is 0.1.
    """

    def __init__(
        self,
        threshold: float = 0.1,
    ):
        super().__init__()
        self._threshold = threshold

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train using Step Forward method.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded.
            Expected shape: (batch, channels, freqs, time_steps)
            Must be normalized to the range [0, 1].

        Returns
        -------
        torch.Tensor
            Encoded spike train with values {-1, 0, 1}.
        """
        spike_train = torch.zeros_like(x)

        # Initialize encoding base with the first time step
        encoding_base = x.select(-1, 0).clone()

        for t in range(1, x.shape[-1]):
            current = x.select(-1, t)

            upper_cross = current >= encoding_base + self._threshold
            lower_cross = current <= encoding_base - self._threshold

            spike_train.select(-1, t)[upper_cross] = 1
            spike_train.select(-1, t)[lower_cross] = -1

            encoding_base[upper_cross] += self._threshold
            encoding_base[lower_cross] -= self._threshold

        return spike_train

    def decode(self, x: torch.Tensor, decode_params: Optional[SFDecodeParams] = None) -> torch.Tensor:
        """
        Decode spike trains back to continuous values.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).
        base : torch.Tensor
            Initial baseline values for decoding.

        Returns
        -------
        torch.Tensor
            Decoded continuous values.
        """
        if decode_params is None:
            raise ValueError("decode_params must be provided for decoding.")
        
        if "base" not in decode_params:
            raise ValueError("decode_params must contain 'base'.")
        
        base = decode_params["base"]

        weighted_spikes = x * self._threshold

        spike_cumsum = torch.cumsum(weighted_spikes, dim=-1)

        decoded = base + spike_cumsum

        return decoded

    def get_decode_params(self, x: torch.Tensor) -> SFDecodeParams:
        """
        Get the parameters needed for decoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).

        Returns
        -------
        SFDecodeParams
            Dict containing the encoding base.
        """
        encoding_base = x.select(-1, 0).unsqueeze(-1)

        return {
            "base": encoding_base,
        }


class TBRDecodeParams(TypedDict):
    """
    Type definition for the parameters needed for decoding.

    Parameters
    ----------
    base : torch.Tensor
        Initial baseline values for decoding.
    threshold : torch.Tensor
        Threshold values used for decoding.
    """

    base: torch.Tensor
    threshold: torch.Tensor


class TBREncoder(SpikeEncoder):
    """
    Threshold-Based Representation Encoder for converting continuous values to spike trains.

    This encoder generates spikes when the difference between consecutive timesteps
    exceeds a precalculated threshold based on statistical properties of the input.

    Parameters
    ----------
    threshold : float, optional
        Threshold multiplier for spike generation. Default is 0.1.
    """

    def __init__(
        self,
        threshold: float = 0.1,
    ):
        super().__init__()
        self._threshold = threshold

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train by comparing with
        previous time-step using a fixed threshold.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded.
            Expected shape: (batch, channels, freqs, time_steps)
            Must be normalized to the range [0, 1].

        Returns
        -------
        torch.Tensor
            Encoded spike train with values {-1, 0, 1}.
        """
        padding = torch.zeros_like(x.select(-1, 0).unsqueeze(-1))
        x_diff = x.diff(dim=-1, prepend=padding)
        threshold = torch.mean(x_diff, dim=-1, keepdim=True) + (
            torch.std(x_diff, dim=-1, keepdim=True) * self._threshold
        )
        pos_spikes = x_diff > threshold
        neg_spikes = x_diff < -threshold

        spike_train = pos_spikes.float() - neg_spikes.float()

        return spike_train

    def decode(self, x: torch.Tensor, decode_params: Optional[TBRDecodeParams] = None) -> torch.Tensor:
        """
        Decode spike trains back to continuous values.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).
        base : torch.Tensor
            Initial baseline values for decoding.
        threshold : torch.Tensor
            Threshold values used for decoding.

        Returns
        -------
        torch.Tensor
            Decoded continuous values.
        """
        if decode_params is None:
            raise ValueError("decode_params must be provided for decoding.")
        
        if "base" not in decode_params or "threshold" not in decode_params:
            raise ValueError("decode_params must contain 'base' and 'threshold'.")
        
        threshold = decode_params["threshold"]
        base = decode_params["base"]

        decoded = (x.cumsum(dim=-1) * threshold) + base
        return decoded

    def get_decode_params(self, x: torch.Tensor) -> TBRDecodeParams:
        """
        Get the parameters needed for decoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).

        Returns
        -------
        TBRDecodeParams
            Dict containing the encoding base and threshold.
        """
        padding = torch.zeros_like(x.select(-1, 0).unsqueeze(-1))
        x_diff = x.diff(dim=-1, prepend=padding)
        threshold = torch.mean(x_diff, dim=-1, keepdim=True) + (
            torch.std(x_diff, dim=-1, keepdim=True) * self._threshold
        )
        encoding_base = x.select(-1, 0).clone().unsqueeze(-1)

        return {
            "base": encoding_base,
            "threshold": threshold,
        }
