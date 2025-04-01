import torch


class SpikeEncoder:
    """
    Base class for spike encoders.
    """

    def __init__(self, **kwargs):
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a continuous value into spikes.

        Args:
            x (torch.Tensor): Input tensor to be encoded.

        Returns:
            torch.Tensor: Encoded spike train.
        """
        raise NotImplementedError("Subclasses must implement encode method")
