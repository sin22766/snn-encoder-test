import torch

from encoder.base import SpikeEncoder


class StepForwardEncoder(SpikeEncoder):
    """
    Step Forward Encoder for converting continuous values to spike trains.
    
    This encoder generates spikes when the input signal crosses a threshold
    relative to an adaptive baseline that changes with each spike.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        normalize: bool = True,
    ):
        """
        Initialize the Step Forward Encoder.
        
        Parameters
        ----------
        threshold : float, optional
            Threshold for spike generation used to modify the baseline on each timestep.
            Default is 0.1.
        normalize : bool, optional
            Whether to normalize the input tensor. Default is True.
        """
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train using Step Forward method.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).
            
        Returns
        -------
        torch.Tensor
            Encoded spike train with values {-1, 0, 1}.
        """
        if self.normalize:
            x = self._normalize(x)  # Normalize along the channel axis

        spike_train = torch.zeros_like(x)

        # Initialize encoding base with the first time step
        encoding_base = x.select(-1, 0).clone() 

        for t in range(1, x.shape[-1]):
            current = x.select(-1, t)

            upper_cross = current >= encoding_base + self.threshold
            lower_cross = current <= encoding_base - self.threshold

            spike_train.select(-1, t)[upper_cross] = 1
            spike_train.select(-1, t)[lower_cross] = -1

            encoding_base[upper_cross] += self.threshold
            encoding_base[lower_cross] -= self.threshold

        return spike_train

    def decode(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
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
        decoded = torch.zeros_like(x)
        encoding_base = base.clone()

        decoded.select(-1, 0).copy_(encoding_base)

        for t in range(1, x.shape[-1]):
            spikes = x.select(-1, t)
            encoding_base += self.threshold * spikes
            decoded.select(-1, t).copy_(encoding_base)

        return decoded
        

class TBREncoder(SpikeEncoder):
    """
    Threshold-Based Representation Encoder for converting continuous values to spike trains.
    
    This encoder generates spikes when the difference between consecutive timesteps
    exceeds a precalculated threshold based on statistical properties of the input.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        normalize: bool = True,
    ):
        """
        Initialize the TBR Encoder.
        
        Parameters
        ----------
        threshold : float, optional
            Threshold multiplier for spike generation. Default is 0.1.
        normalize : bool, optional
            Whether to normalize the input tensor. Default is True.
        """
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into a spike train by comparing with
        previous time-step using a fixed threshold.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).
            
        Returns
        -------
        torch.Tensor
            Encoded spike train with values {-1, 0, 1}.
        """
        if self.normalize:
            x = self._normalize(x)  # Normalize along the channel axis

        padding = torch.zeros_like(x.select(-1, 0).unsqueeze(-1))
        x_diff = x.diff(dim=-1, prepend=padding)
        threshold = torch.mean(x_diff, dim=-1, keepdim=True) + (torch.std(x_diff, dim=-1, keepdim=True) * self.threshold)
        pos_spikes = x_diff > threshold
        neg_spikes = x_diff < -threshold

        spike_train = (pos_spikes.float() - neg_spikes.float())

        return spike_train

    def decode(self, x: torch.Tensor, base: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
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
        decoded = (x.cumsum(dim=-1) * threshold) + base
        return decoded
    
    def get_decode_parameters(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the parameters needed for decoding.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be encoded, with shape (Batch, Channels, Freqs, Times).
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing the encoding base and threshold.
        """
        padding = torch.zeros_like(x.select(-1, 0).unsqueeze(-1))
        x_diff = x.diff(dim=-1, prepend=padding)
        threshold = torch.mean(x_diff, dim=-1, keepdim=True) + (torch.std(x_diff, dim=-1, keepdim=True) * self.threshold)
        encoding_base = x.select(-1, 0).clone().unsqueeze(-1)
        return encoding_base, threshold
