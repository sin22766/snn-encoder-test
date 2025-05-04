import torch

from .base import SpikeEncoder


class PoissonEncoder(SpikeEncoder):
    """
    PoissonEncoder for converting continuous values to spike trains using stochastic encoding.

    This encoder implements a discrete-time approximation of a Poisson process by:
    1. Normalizing inputs to [0,1] range to represent firing probabilities
    2. Downsampling the time dimension by a factor of interval_freq
    3. Generating spikes by sampling from a Bernoulli distribution

    Parameters
    ----------
    interval_freq : int, optional
        Factor by which to downsample the time dimension. Default is 4.
    random_seed : int, optional
        Seed for random number generation to ensure reproducible results.
        Default is None (random behavior).
    normalize : bool, optional
        Whether to normalize the input tensor to range [0,1]. Default is True.
    """

    def __init__(
        self,
        interval_freq: int = 4,
        random_seed: int = None,
    ):
        super().__init__()
        self._interval_freq = interval_freq
        self._random_seed = random_seed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train using a Poisson process.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded.
            Expected shape: (batch, channels, freqs, time_steps)

        Returns
        -------
        torch.Tensor
            Encoded spike train with shape (batch, channels, freqs, time_steps)
            and binary values {0, 1}.

        Notes
        -----
        The encoding downsampling the time dimension by interval_freq and uses each input
        value as the probability parameter for a Bernoulli distribution, sampling
        spikes stochastically.
        """
        batch, channels, freqs, time_steps = x.shape

        padding = -time_steps % self._interval_freq

        padded_x = torch.nn.functional.pad(x, (0, padding), value=float("nan"))

        grouped_x = padded_x.reshape(batch, channels, freqs, -1, self._interval_freq)

        downsampled_x = grouped_x.nanmean(dim=-1).repeat_interleave(
            dim=-1, repeats=self._interval_freq
        )

        downsampled_x = downsampled_x[..., :time_steps]

        # Initialize random generator
        generator = torch.Generator(device=x.device)
        if self._random_seed is not None:
            generator.manual_seed(self._random_seed)

        spike_train = torch.bernoulli(downsampled_x, generator=generator)

        return spike_train

    def decode(self, x):
        """
        Decode a spike train back to continuous values.

        Parameters
        ----------
        x : torch.Tensor
            Input spike train to be decoded.
            Expected shape: (batch, channels, freqs, time_steps)

        Returns
        -------
        torch.Tensor
            Decoded continuous values with shape (batch, channels, freqs, time_steps)
            and values in range [0, 1].

        Notes
        -----
        Decoding is performed by averaging spike counts over each interval_freq
        time steps, which approximates the original firing probability.
        """
        batch, channels, freqs, time_steps = x.shape

        padding = -time_steps % self._interval_freq

        padded_x = torch.nn.functional.pad(x, (0, padding), value=float("nan"))

        grouped_x = padded_x.reshape(batch, channels, freqs, -1, self._interval_freq)

        decoded = grouped_x.nanmean(dim=-1).repeat_interleave(dim=-1, repeats=self._interval_freq)

        decoded = decoded[..., :time_steps]

        return decoded


class PoissonEncoderExpand(SpikeEncoder):
    """
    PoissonEncoderExpand for converting continuous values to spike trains using stochastic encoding.

    This encoder implements a discrete-time approximation of a Poisson process by:
    1. Normalizing inputs to [0,1] range to represent firing probabilities
    2. Upsampling the time dimension by a factor of interval_freq by repeating each time step interval_freq times
    3. Generating spikes by sampling from a Bernoulli distribution

    Parameters
    ----------
    interval_freq : int, optional
        Factor by which to upsampling the time dimension, controlling temporal resolution
        of the spike train. Default is 4.
    random_seed : int, optional
        Seed for random number generation to ensure reproducible results.
        Default is None (random behavior).
    """

    def __init__(
        self,
        interval_freq: int = 4,
        random_seed: int = None,
    ):
        super().__init__()
        self._interval_freq = interval_freq
        self._random_seed = random_seed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train using a Poisson process.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded.
            Expected shape: (batch, channels, freqs, time_steps)
            Must be normalized to the range [0, 1].

        Returns
        -------
        torch.Tensor
            Encoded spike train with shape (batch, channels, freqs, time_steps * interval_freq)
            and binary values {0, 1}.

        Notes
        -----
        The encoding expands the time dimension by interval_freq and uses each input
        value as the probability parameter for a Bernoulli distribution, sampling
        spikes stochastically.
        """
        # Expand time dimension by repeating each time step interval_freq times
        x_repeat = x.repeat_interleave(self._interval_freq, dim=-1)

        # Initialize random generator
        generator = torch.Generator(device=x.device)
        if self._random_seed is not None:
            generator.manual_seed(self._random_seed)

        # Generate spikes via Bernoulli sampling
        # Each value in x_repeat is treated as probability of generating a spike
        spike_train = torch.bernoulli(x_repeat, generator=generator)

        return spike_train

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a spike train back to continuous values.

        Parameters
        ----------
        x : torch.Tensor
            Input spike train to be decoded.
            Expected shape: (batch, channels, freqs, time_steps)
            where time_steps = original_time_steps * interval_freq

        Returns
        -------
        torch.Tensor
            Decoded continuous values with shape (batch, channels, freqs, time_steps // interval_freq)
            and values in range [0, 1].

        Notes
        -----
        Decoding is performed by averaging spike counts over each interval_freq
        time steps, which approximates the original firing probability.
        """
        # Reshape to group each interval_freq time steps
        x_grouped = x.reshape(*x.shape[:-1], -1, self._interval_freq)

        # Average spike count over each group to recover approximate probability
        decoded = x_grouped.mean(dim=-1)

        return decoded
