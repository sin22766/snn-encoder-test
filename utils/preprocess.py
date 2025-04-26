import torch


def VectorizeSTFT(eeg_data: torch.Tensor, n_fft=256, hop_length=32, win_length=128):
    """
    Apply STFT to batched EEG data using vectorization

    Parameters:
    -----------
    eeg_data: torch.Tensor
        EEG data with shape (batch, channels, time_steps)

    Returns:
    --------
    stft_output: torch.Tensor
        STFT output with shape (batch, channels, frequency_bins, time_frames)
    """
    batch_size, n_channels, time_steps = eeg_data.shape
    window = torch.hann_window(win_length, device=eeg_data.device)

    # Reshape to (batch*channels, time_steps)
    reshaped_data = eeg_data.reshape(-1, time_steps)

    # Apply STFT to all channels at once
    stft = torch.stft(
        reshaped_data,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )

    # Reshape back to (batch, channels, freq_bins, time_frames)
    _, freq_bins, time_frames = stft.shape
    stft_output = stft.reshape(batch_size, n_channels, freq_bins, time_frames)

    return stft_output