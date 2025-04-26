import torch
from scipy import signal
from torch.nn.functional import pad

from encoder.base import SpikeEncoder


class BSAEncoder(SpikeEncoder):
    """
    BSA (Ben Spiker Algorithm) Encoder for converting continuous values to spike trains.

    This encoder uses a sliding window approach with a low-pass filter to transform
    continuous signals into spike trains.
    """

    def __init__(
        self,
        win_size: int = 8,
        cutoff: float = 0.2,
        threshold: float = 0.95,
        normalize: bool = True,
    ):
        """
        Initialize the BSA Encoder.

        Parameters
        ----------
        win_size : int, optional
            Window size for the moving average filter. Default is 8.
        cutoff : float, optional
            Cutoff frequency for the low-pass filter. Default is 0.2.
        threshold : float, optional
            Threshold for spike generation. Default is 0.95.
        normalize : bool, optional
            Whether to normalize the input tensor. Default is True.
        """
        super().__init__()
        self.win_size = win_size
        self.cutoff = cutoff
        self.threshold = threshold
        self.normalize = normalize

        self.filter = signal.firwin(win_size, cutoff=cutoff)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train using BSA.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded.
            Expected shape: (Batch, Channels, Freqs, Times)
                
        Returns
        -------
        torch.Tensor
            Encoded spike train with same shape as input.
        """
        if self.normalize:
            x = self._normalize(x)

            # Get dimensions
        batch, channels, freqs, time_steps = x.shape

        # Reshape for processing
        x_reshaped = x.reshape(batch * channels * freqs, time_steps)

        # Create output tensor and working copy
        spike_train = torch.zeros_like(x_reshaped)
        x_clone = x_reshaped.clone()

        # Create tiled filter window
        filter_window = torch.tensor(self.filter, dtype=x.dtype, device=x.device)
        filter_w = filter_window.expand(batch * channels * freqs, self.win_size)

        # Vectorized encoding
        for t in range(time_steps - self.win_size + 1):
            window_slice = x_clone[:, t : t + self.win_size]
            error1 = torch.abs(window_slice - filter_w).sum(dim=1)
            error2 = torch.abs(window_slice).sum(dim=1)

            mask = error1 <= error2 - self.threshold
            spike_train[mask, t] = 1
            x_clone[mask, t : t + self.win_size] -= filter_w[mask]

        # Reshape back to original dimensions
        return spike_train.reshape(batch, channels, freqs, time_steps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a spike train back to continuous values.
        
        Parameters
        ----------
        x : torch.Tensor
            Input spike train to be decoded.
            Expected shape: (Batch, Channels, Freqs, Times)
                
        Returns
        -------
        torch.Tensor
            Decoded continuous signal with same shape as input.
        """
        # Get dimensions
        batch, channels, freqs, time_steps = x.shape

        # Reshape for processing
        spike_reshape = x.reshape(batch * channels * freqs, time_steps)
        
        # Pad signal for convolution
        spike_padded = pad(spike_reshape, (self.win_size - 1, self.win_size - 1))

        # Unfold and prepare for convolution
        unfold_spike = spike_padded.unfold(1, self.win_size, 1)
        filter_window = torch.tensor(self.filter, dtype=x.dtype, device=x.device).reshape(1, 1, self.win_size)

        # Apply convolution
        convolute = torch.sum(unfold_spike * filter_window, dim=2)[:, :time_steps]
        
        # Reshape result
        return convolute.reshape(batch, channels, freqs, time_steps)
