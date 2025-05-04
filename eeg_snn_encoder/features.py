from typing import Optional

import torch


def stft(
    x: torch.Tensor,
    n_fft: int = 256,
    hop_length: int = 32,
    win_length: int = 128,
    window: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply STFT to batched EEG data efficiently by vectorizing the operation.

    Parameters:
    -----------
    x : torch.Tensor
        Input EEG data of shape (batch, channels, time_steps)
    n_fft : int, optional
        FFT size
    hop_length : int, optional
        Hop length between frames
    win_length : int, optional
        Window length
    window : torch.Tensor, optional
        Precomputed window to reuse (must be on same device as eeg_data)

    Returns:
    --------
    torch.Tensor
        STFT result with shape (batch, channels, freq_bins, time_frames)
    """
    batch_size, n_channels, time_steps = x.shape

    # Avoid creating a new window on every call if reused
    if window is None:
        window = torch.hann_window(win_length, device=x.device)

    # Reshape to (batch*channels, time_steps)
    reshaped_data = x.reshape(-1, time_steps)

    # Apply STFT to all channels at once
    with torch.no_grad():
        stft = torch.stft(
            reshaped_data,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )

    # Reshape back to (batch, channels, freq_bins, time_frames)
    stft_output = stft.reshape(batch_size, n_channels, *stft.shape[1:])

    return stft_output


def normalize(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Normalize EEG data using the min-max scaling method to ensure all values are between [0, 1]

    Parameters:
    -----------
    x : torch.Tensor
        Input EEG data of shape (batch, channels, time_steps)

    Returns:
    --------
    torch.Tensor
        Normalized tensor with values in range [0, 1].
    """
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values

    diff = x_max - x_min
    diff[diff == 0] = 1.0

    return (x - x_min) / diff
