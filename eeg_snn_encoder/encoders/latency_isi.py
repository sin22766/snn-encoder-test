import torch

from .base import SpikeEncoder


class BurstEncoder(SpikeEncoder):
    """
    BurstEncoder encodes continuous values into burst spike trains with configurable spike count and inter-spike interval (ISI).

    This encoder simulates biological bursting by mapping each input value to a burst of spikes.
    The number of spikes and the inter-spike interval are determined based on the input magnitude and configuration parameters.
    The time dimension is expanded by a factor of `max_window`, allowing for temporal burst representation.

    Parameters
    ----------
    max_window : int, optional
        Maximum length of the burst window (i.e., max number of time steps per encoded value). Default is 8.
    n_max : int, optional
        Maximum number of spikes that can be emitted per input value. Default is 4.
    t_max : int, optional
        Maximum inter-spike interval (ISI) in time steps. Default is 2.
    t_min : int, optional
        Minimum inter-spike interval (ISI) in time steps. Default is 0.
    normalize : bool, optional
        Whether to normalize the input tensor before encoding. Default is True.

    Notes
    -----
    Each input value is transformed into a spike burst where:
    - Spike count is proportional to the input magnitude, up to `n_max`.
    - ISI is inversely proportional to the input magnitude, between `t_min` and `t_max`.
    - The output has a time dimension expanded by `max_window`.
    """

    def __init__(
        self,
        max_window: int = 8,
        n_max: int = 4,
        t_max: int = 2,
        t_min: int = 0,
    ):
        super().__init__()
        self._max_window = max_window
        self._n_max = n_max
        self._t_max = t_max
        self._t_min = t_min

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous input tensor into burst spike trains.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, freqs, time_steps).
            Must be normalized to the range [0, 1].

        Returns
        -------
        torch.Tensor
            Encoded spike train with shape (batch, channels, freqs, time_steps * max_window).
        """
        batch, channels, freqs, time_steps = x.shape

        padding = -time_steps % self._max_window

        padded_x = torch.nn.functional.pad(x, (0, padding), value=float("nan")).to(device=x.device)

        downsampled_x = padded_x.reshape(batch * channels * freqs, -1, self._max_window).nanmean(
            dim=-1
        )

        spike_index = (
            torch.arange(0, self._max_window, 1)
            .expand((batch * channels * freqs, downsampled_x.shape[1], self._max_window))
            .to(device=x.device)
        )

        spike_count = torch.ceil(downsampled_x * self._n_max)
        isi = torch.ceil(self._t_max - (downsampled_x * (self._t_max - self._t_min)))
        burst_length = spike_count * (isi + 1)

        spike_train = (spike_index % (isi.unsqueeze(-1) + 1) == 0).float()
        spike_train[spike_index >= burst_length.unsqueeze(-1)] = 0
        spike_train = spike_train.reshape(
            batch, channels, freqs, downsampled_x.shape[1] * self._max_window
        )

        return spike_train

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode burst spike trains back to continuous values.

        Parameters
        ----------
        x : torch.Tensor
            Spike train tensor of shape (batch, channels, freqs, time_steps),
            where time_steps is expected to be a multiple of max_window.

        Returns
        -------
        torch.Tensor
            Decoded continuous tensor with shape (batch, channels, freqs, time_steps // max_window).
        """
        batch, channels, freqs, time_steps = x.shape

        time_steps = time_steps // self._max_window

        x_reshaped = x.reshape(batch * channels * freqs, time_steps, self._max_window)

        decoded = torch.sum(x_reshaped, dim=-1) / self._max_window
        decoded = decoded.repeat_interleave(self._max_window, dim=-1)
        decoded = decoded.reshape(batch, channels, freqs, time_steps * self._max_window)

        return decoded


class BurstEncoderExpand(SpikeEncoder):
    """
    BurstEncoderExpand encodes continuous values into burst spike trains with configurable spike count and inter-spike interval (ISI).

    This encoder simulates biological bursting by mapping each input value to a burst of spikes.
    The number of spikes and the inter-spike interval are determined based on the input magnitude and configuration parameters.
    The time dimension is expanded by a factor of `max_window`, allowing for temporal burst representation.

    Parameters
    ----------
    max_window : int, optional
        Maximum length of the burst window (i.e., max number of time steps per encoded value). Default is 8.
    n_max : int, optional
        Maximum number of spikes that can be emitted per input value. Default is 4.
    t_max : int, optional
        Maximum inter-spike interval (ISI) in time steps. Default is 2.
    t_min : int, optional
        Minimum inter-spike interval (ISI) in time steps. Default is 0.

    Notes
    -----
    Each input value is transformed into a spike burst where:
    - Spike count is proportional to the input magnitude, up to `n_max`.
    - ISI is inversely proportional to the input magnitude, between `t_min` and `t_max`.
    - The output has a time dimension expanded by `max_window`.
    """

    def __init__(
        self,
        max_window: int = 8,
        n_max: int = 4,
        t_max: int = 2,
        t_min: int = 0,
    ):
        """
        Initialize the Burst Encoder.

        Parameters
        ----------
        max_window : int, optional
            Maximum length of the burst window (i.e., max number of time steps per encoded value). Default is 8.
        n_max : int, optional
            Maximum number of spikes that can be emitted per input value. Default is 4.
        t_max : int, optional
            Maximum inter-spike interval (ISI) in time steps. Default is 2.
        t_min : int, optional
            Minimum inter-spike interval (ISI) in time steps. Default is 0.
        """
        super().__init__()
        self._max_window = max_window
        self._n_max = n_max
        self._t_max = t_max
        self._t_min = t_min

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous input tensor into burst spike trains.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, freqs, time_steps).
            Must be normalized to the range [0, 1].

        Returns
        -------
        torch.Tensor
            Encoded spike train with shape (batch, channels, freqs, time_steps * max_window).
        """
        batch, channels, freqs, time_steps = x.shape
        x_reshaped = x.reshape(batch * channels * freqs, time_steps)

        spike_index = (
            torch.arange(0, self._max_window, 1)
            .expand((batch * channels * freqs, time_steps, self._max_window))
            .to(device=x.device)
        )

        spike_count = torch.ceil(x_reshaped * self._n_max)
        isi = torch.ceil(self._t_max - (x_reshaped * (self._t_max - self._t_min)))
        burst_length = spike_count * (isi + 1)

        spike_train = (spike_index % (isi.unsqueeze(-1) + 1) == 0).float()
        spike_train[spike_index >= burst_length.unsqueeze(-1)] = 0
        spike_train = spike_train.reshape(batch, channels, freqs, time_steps * self._max_window)

        return spike_train

    def decode(self, x: torch.Tensor, decode_params=None) -> torch.Tensor:
        """
        Decode burst spike trains back to continuous values.

        Parameters
        ----------
        x : torch.Tensor
            Spike train tensor of shape (batch, channels, freqs, time_steps),
            where time_steps is expected to be a multiple of max_window.

        Returns
        -------
        torch.Tensor
            Decoded continuous tensor with shape (batch, channels, freqs, time_steps // max_window).
        """
        batch, channels, freqs, time_steps = x.shape
        time_steps = time_steps // self._max_window

        x_reshaped = x.reshape(batch * channels * freqs, time_steps, self._max_window)
        decoded = torch.sum(x_reshaped, dim=-1) / self._n_max
        decoded = decoded.reshape(batch, channels, freqs, time_steps)

        return decoded
