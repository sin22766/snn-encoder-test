import torch

from encoder.base import SpikeEncoder


class PoissonEncoder(SpikeEncoder):
    """
    Poisson Encoder using bernoulli distribution for spike generation.
    
    This encoder implements a discrete-time approximation of a Poisson process by:
    - Treating the normalized input as spike probability per time bin
    - Generating spikes by sampling from a Bernoulli distribution
    
    The input is normalized to the range [0, 1] to represent firing probabilities.
    """

    def __init__(
        self,
        interval_freq: int = 4,
        random_seed: int = None,
        normalize: bool = True,
    ):
        """
        Initialize the Poisson Encoder.
        
        Parameters
        ----------
        interval_freq : int, optional
            Frequency of the interval for encoding, controlling temporal resolution
            of the spike train. Default is 4.
        random_seed : int, optional
            Seed for random number generation to ensure reproducible results.
            Default is None.
        normalize : bool, optional
            Whether to normalize the input tensor to range [0,1]. Default is True.
        """
        super().__init__()
        self.interval_freq = interval_freq
        self.random_seed = random_seed
        self.normalize = normalize

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train using a Poisson process.
        
        Generates spikes by treating input values as probabilities and
        sampling from a Bernoulli distribution for each time bin.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).
            Values represent firing probabilities after normalization.
            
        Returns
        -------
        torch.Tensor
            Encoded spike train with binary values {0, 1}.
        """
        if self.normalize:
            x = self._normalize(x)  # Normalize along the channel axis

        x_repeat = x.repeat_interleave(self.interval_freq, dim=-1)

        generator = torch.Generator(device=x.device)
        if self.random_seed is not None:
            generator.manual_seed(self.random_seed)

        spike_train = torch.bernoulli(x_repeat, generator=generator)

        return spike_train

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
            Decoded continuous values in range [0, 1].
        """
        x_grouped = x.reshape(*x.shape[:-1], -1, self.interval_freq)
        decoded = x_grouped.mean(dim=-1)
        return decoded