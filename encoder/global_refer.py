import torch

from encoder.base import SpikeEncoder


class PhaseEncoder(SpikeEncoder):
    """
    PhaseEncoder for converting continuous values to spike trains using phase encoding.

    This encoder transforms continuous values into a binary representation across multiple 
    phase windows based on decreasing powers of 2. This version does not expand the time dimension,
    but downsamples the time dimension by a factor equal to the phase_window parameter.

    The encoded data size may be paded to fit the phase_window size.

    Parameters
    ----------
    phase_window : int, optional
        Number of phase windows to use for encoding. Default is 8.
    normalize : bool, optional
        Whether to normalize the input tensor along channel axis. Default is True.
        
    Attributes
    ----------
    phase_values : torch.Tensor
        Precomputed threshold values as powers of 2: [2^-1, 2^-2, ..., 2^-phase_window]
    """
    def __init__(
        self,
        phase_window: int = 8,
        normalize: bool = True,
    ):
        """
        Initialize the Poisson Encoder.

        Parameters
        ----------
        phase_window : int, optional
            Number of phase windows to use for encoding. Default is 8.
        normalize : bool, optional
            Whether to normalize the input tensor along channel axis. Default is True.
        """
        super().__init__()
        self.phase_window = phase_window
        self.normalize = normalize

        self.phase_values = torch.pow(2.0, -torch.arange(1, phase_window + 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train using phase encoding.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded.
            Expected shape: (batch, channels, freqs, time_steps)
        
        Returns
        -------
        torch.Tensor
            Encoded spike train with shape (batch, channels, freqs, time_steps),
            where the time dimension will be padded to be divisible by phase_window.
        """
        if self.normalize:
            x = self._normalize(x)
        
        batch, channels, freqs, time_steps = x.shape

        padding = -time_steps % self.phase_window

        padded_x = torch.nn.functional.pad(x, (0, padding), value=float("nan"))

        mean_x = padded_x.reshape(batch * channels * freqs, -1, self.phase_window).nanmean(dim=-1)

        spike_train = torch.zeros(*mean_x.shape, self.phase_window, dtype=torch.float32,  device=mean_x.device)

        if self.phase_values.device != x.device:
            self.phase_values = self.phase_values.to(x.device)
        
        phase_values = self.phase_values

        for i in range(self.phase_window):
            mask = mean_x >= phase_values[i]
            spike_train[:, :, i] = mask.float()
            mean_x[mask] -= phase_values[i]
        
        spike_train = spike_train.reshape(batch, channels, freqs, time_steps + padding)
        return spike_train

    def decode(self, x: torch.Tensor) -> torch.Tensor:
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
            Decoded continuous signal with shape (batch, channels, freqs, time_steps)
            with values reconstructed from the spike patterns.
        """
        batch, channels, freqs, time_steps = x.shape

        time_steps = time_steps // self.phase_window

        x_reshape = x.reshape(batch * channels * freqs, time_steps, self.phase_window)

        if self.phase_values.device != x.device:
            self.phase_values = self.phase_values.to(x.device)
        
        phase_values = self.phase_values

        decoded = x_reshape.matmul(phase_values)

        decoded = decoded.reshape(batch, channels, freqs, time_steps)

        decoded = decoded.repeat_interleave(self.phase_window, dim=-1)

        return decoded



class PhaseEncoderExpand(SpikeEncoder):
    """
    PhaseEncoderExpand for converting continuous values to spike trains using phase encoding.
    
    This encoder transforms continuous values into a binary representation across multiple 
    phase windows based on decreasing powers of 2. It expands the time dimension by a factor 
    equal to the phase_window parameter, creating a detailed temporal code.
    
    The encoding creates a binary decomposition where each phase window represents 
    a specific power-of-2 contribution to the original value, similar to binary representation.
    
    Parameters
    ----------
    phase_window : int, optional
        Number of phase windows to use for encoding. Default is 8.
    normalize : bool, optional
        Whether to normalize the input tensor along channel axis. Default is True.
        
    Attributes
    ----------
    phase_values : torch.Tensor
        Precomputed threshold values as powers of 2: [2^-1, 2^-2, ..., 2^-phase_window]
    
    Notes
    -----
    The encoder preserves the magnitude information of inputs while converting them
    to a spike-based format suitable for spiking neural networks. The time dimension
    is expanded by a factor of phase_window in the output.
    """
    def __init__(
        self,
        phase_window: int = 8,
        normalize: bool = True,
    ):
        """
        Initialize the Poisson Encoder.

        Parameters
        ----------
        phase_window : int, optional
            Number of phase windows to use for encoding. Default is 8.
        normalize : bool, optional
            Whether to normalize the input tensor along channel axis. Default is True.
        """
        super().__init__()
        self.phase_window = phase_window
        self.normalize = normalize

        self.phase_values = torch.pow(2.0, -torch.arange(1, phase_window + 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train using phase encoding.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded.
            Expected shape: (batch, channels, freqs, time_steps)
        
        Returns
        -------
        torch.Tensor
            Encoded spike train with shape (batch, channels, freqs, time_steps * phase_window),
            where the time dimension is expanded by the phase_window factor.
        """
        if self.normalize:
            x = self._normalize(x)  # Normalize along the channel axis
        
        batch, channels, freqs, time_steps = x.shape

        x_reshaped = x.reshape(batch * channels * freqs, time_steps).clone()

        if self.phase_values.device != x.device:
            self.phase_values = self.phase_values.to(x.device)
        
        phase_values = self.phase_values

        spike_train = torch.zeros(*x_reshaped.shape, self.phase_window, dtype=torch.float32,  device=x.device)

        for i in range(self.phase_window):
            mask = x_reshaped >= phase_values[i]
            spike_train[:, :, i] = mask.float()
            x_reshaped[mask] -= phase_values[i]

        return spike_train.reshape(batch, channels, freqs, time_steps * self.phase_window)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a spike train back to continuous values.
        
        Parameters
        ----------
        x : torch.Tensor
            Input spike train to be decoded.
            Expected shape: (batch, channels, freqs, time_steps)
            where time_steps = original_time_steps * phase_window
        
        Returns
        -------
        torch.Tensor
            Decoded continuous signal with shape (batch, channels, freqs, time_steps // phase_window)
            with values reconstructed from the spike patterns.
        """
        batch, channels, freqs, time_steps = x.shape

        time_steps = time_steps // self.phase_window

        x_reshape = x.reshape(batch * channels * freqs, time_steps, self.phase_window)

        if self.phase_values.device != x.device:
            self.phase_values = self.phase_values.to(x.device)
        
        phase_values = self.phase_values

        decoded = x_reshape.matmul(phase_values)

        decoded = decoded.reshape(batch, channels, freqs, time_steps)

        return decoded