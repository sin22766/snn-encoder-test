import torch
from snntorch import spikegen
from encoder.base import SpikeEncoder


class PoissonEncoder(SpikeEncoder):
    """
    Poisson Encoder for converting continuous values to spike trains.
    """

    def __init__(self, interval_freq: int = 16, random_seed: int = 0, **kwargs):
        """
        Initialize the Poisson Encoder.

        Args:
            interval_freq (int): Sample interval for encoding.
            random_seed (int): Random seed for reproducibility.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.interval_freq = interval_freq
        self.random_seed = random_seed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train using a Poisson process.

        Args:
            x (torch.Tensor): Input tensor to be encoded.

        Returns:
            torch.Tensor: Encoded spike train.
        """
        # Implement the Poisson encoding logic here
        pass

class PoissonEncoderSNNTorch(SpikeEncoder):
    """
    Poisson Encoder that utilize the snnTorch SpikeGen.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Poisson Encoder.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train using a Poisson process.

        Args:
            x (torch.Tensor): Input tensor to be encoded.

        Returns:
            torch.Tensor: Encoded spike train.
        """
        
        # Calculate the absolute value of the input tensor
        x = torch.abs(x)
        # Normalize the input tensor to the range [0, 1]
        if x.max() > 0:
            x = x / x.max()
        
        spike_train = spikegen.rate(x, time_var_input=True)

        return spike_train

