import torch
from encoder.base import SpikeEncoder
from scipy import signal

class BSAEncoder(SpikeEncoder):
    """
    BSA (Ben Spiker Algorithm) Encoder for converting continuous values to spike trains.
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

        Args:
            axis (int, optional): Axis along which to encode the spikes, if not specified, the last axis is used.
            win_size (int, optional): Window size for the moving average filter.
            cutoff (float, optional): Cutoff frequency for the low-pass filter.
            threshold (float, optional): Threshold for spike generation use to modify the baseline on each timestep.
            normalize (bool, optional): Whether to normalize the input tensor. Default is True.
            normalize_axis (int, optional): Axis along which to normalize the tensor. If None, global normalization is applied.
        """
        super().__init__()
        self.win_size = win_size
        self.cutoff = cutoff
        self.threshold = threshold
        self.normalize = normalize

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train using BSA.
        Expect input to be of shape (Batch, Channels, Times) or (Batch, Channels, Freqs, Times).

        Args:
            x (torch.Tensor): Input tensor to be encoded.

        Returns:
            torch.Tensor: Encoded spike train.
        """
        if self.normalize:
            x = self._normalize(x)
        
        

        
