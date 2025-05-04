from eeg_snn_encoder import config  # noqa: F401

__docformat__ = "numpy"

__doc__ = """
This module contains the utility for experiment the spike encoders for EEG data preprocessing.
The encoders are designed to convert continuous EEG signals into spike trains suitable for spiking neural networks (SNNs).

The module includes various encoders such as:
- DummyEncoder: A dummy encoder that does not perform any encoding.
- SpikeEncoder: A base encoder for spike encoding.
- PoissonEncoder: An encoder that uses Poisson processes for encoding.
- PoissonEncoderExpand: An expanded version of the Poisson encoder.
- StepForwardEncoder: An encoder that uses a step forward approach for encoding.
- TBREncoder: An encoder that uses temporal binning for encoding.
- BurstEncoder: An encoder that uses burst encoding.
"""
