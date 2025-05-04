from typing import Tuple, TypedDict

import snntorch as snn
from snntorch import SConv2dLSTM, surrogate
import torch
import torch.nn as nn

__doc__ = """
Module for EEG seizure classification using spiking neural networks.

This module provides models for seizure detection from EEG data using spiking neural network (SNN) architectures.
The models is the PyTorch nn.Module class and is designed to work with EEG data.
"""


class ModelConfig(TypedDict):
    """
    Configuration parameters for the spiking neural network (SNN) seizure classifier.

    Attributes
    ----------
    threshold : float
        Neuron firing threshold.
    slope : float
        Slope parameter for the surrogate gradient used in the fully connected layers.
    beta : float
        Membrane potential decay rate.
    dropout_rate1 : float
        Dropout rate applied after the first fully connected layer.
    dropout_rate2 : float
        Dropout rate applied after the output layer.
    """


class EEGSTFTSpikeClassifier(nn.Module):
    """
    Spiking Neural Network for EEG-based seizure classification using STFT features.

    This model processes multi-channel EEG data and classifies each sequence as seizure or non-seizure.

    Parameters
    ----------
    config : ModelConfig, optional
        Configuration parameters for neuron behavior and dropout.
    input_channels : int, optional
        Number of EEG channels.
    freq_dim : int, optional
        Frequency dimension of the STFT features.
    num_classes : int, optional
        Number of output classes (e.g., 2 for binary classification).

    Attributes
    ----------
    conv_lstm1, conv_lstm2, conv_lstm3 : SConv2dLSTM
        Spiking convolutional LSTM layers for spatial-temporal feature extraction.
    fc1, fc2 : nn.Linear
        Fully connected layers for classification.
    lif1, lif2 : snn.Leaky
        Leaky integrate-and-fire neurons applied after fully connected layers.
    dropout1, dropout2 : nn.Dropout
        Dropout layers applied after each fully connected layer.
    """

    def __init__(
        self,
        config: ModelConfig = {
            "threshold": 0.05,
            "slope": 13.42287274232855,
            "beta": 0.9181805491303656,
            "dropout_rate1": 0.5083664100388336,
            "dropout_rate2": 0.26260898840708335,
        },
        input_channels: int = 22,
        freq_dim: int = 129,
        num_classes: int = 2,
    ):
        super().__init__()

        # Define surrogate gradient functions for backpropagation
        lstm_surrogate = surrogate.straight_through_estimator()
        fc_surrogate = surrogate.fast_sigmoid(slope=config["slope"])

        # Calculate dimensions after max pooling operations
        # After 3 max-pooling layers (each dividing by 2), size becomes: freq_dim → freq_dim/2 → freq_dim/4 → freq_dim/8
        final_freq_dim = freq_dim // 8

        # Spiking Convolutional LSTM layers
        self.conv_lstm1 = SConv2dLSTM(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            max_pool=(2, 1),
            threshold=config["threshold"],
            spike_grad=lstm_surrogate,
        )

        self.conv_lstm2 = SConv2dLSTM(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            max_pool=(2, 1),
            threshold=config["threshold"],
            spike_grad=lstm_surrogate,
        )

        self.conv_lstm3 = SConv2dLSTM(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            max_pool=(2, 1),
            threshold=config["threshold"],
            spike_grad=lstm_surrogate,
        )

        # Fully connected layers with leaky integrate-and-fire neurons
        self.fc1 = nn.Linear(64 * final_freq_dim * 1, 512)
        self.lif1 = snn.Leaky(
            beta=config["beta"], spike_grad=fc_surrogate, threshold=config["threshold"]
        )
        self.dropout1 = nn.Dropout(config["dropout_rate1"])

        # Output layer
        self.fc2 = nn.Linear(512, num_classes)
        self.lif2 = snn.Leaky(
            beta=config["beta"], spike_grad=fc_surrogate, threshold=config["threshold"]
        )
        self.dropout2 = nn.Dropout(config["dropout_rate2"])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, freq, time), where:
            - channels: Number of EEG channels
            - freq: Frequency dimension from STFT
            - time: Number of time steps

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - Tensor of output spikes over all time steps (time, batch_size, num_classes)
            - Tensor of membrane potentials over all time steps (same shape)
        """
        time_steps = x.size(3)

        # Initialize neuron states
        # LSTM state variables (synaptic current and membrane potential)
        synapse1, membrane1 = self.conv_lstm1.init_sconv2dlstm()
        synapse2, membrane2 = self.conv_lstm2.init_sconv2dlstm()
        synapse3, membrane3 = self.conv_lstm3.init_sconv2dlstm()

        # LIF neuron membrane potentials
        membrane_fc1 = self.lif1.init_leaky()
        membrane_fc2 = self.lif2.init_leaky()

        # Containers to store outputs at each time step
        output_spikes = []
        output_potentials = []

        # Process each time step sequentially
        for t in range(time_steps):
            # Extract current time slice (keeping the dimension with unsqueeze)
            x_t = x[:, :, :, t].unsqueeze(-1)

            # Process through spiking convolutional LSTM layers
            spike1, synapse1, membrane1 = self.conv_lstm1(x_t, synapse1, membrane1)
            spike2, synapse2, membrane2 = self.conv_lstm2(spike1, synapse2, membrane2)
            spike3, synapse3, membrane3 = self.conv_lstm3(spike2, synapse3, membrane3)

            # Flatten and process through fully connected layers
            flattened = spike3.flatten(1)
            fc1_input = self.fc1(flattened)
            fc1_input = self.dropout1(fc1_input)
            spike_fc1, membrane_fc1 = self.lif1(fc1_input, membrane_fc1)

            fc2_input = self.fc2(spike_fc1)
            fc2_input = self.dropout2(fc2_input)
            spike_fc2, membrane_fc2 = self.lif2(fc2_input, membrane_fc2)

            # Record outputs
            output_spikes.append(spike_fc2)
            output_potentials.append(membrane_fc2)

        # Stack outputs along time dimension
        return torch.stack(output_spikes), torch.stack(output_potentials)
