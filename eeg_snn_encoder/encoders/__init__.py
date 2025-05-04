# Base / Utility
from .base import SpikeEncoder
from .deconvolution import BSAEncoder
from .dummy import DummyEncoder
from .global_temporal import PhaseEncoder, PhaseEncoderExpand
from .latency_isi import BurstEncoder, BurstEncoderExpand
from .rate import PoissonEncoder, PoissonEncoderExpand
from .temporal import StepForwardEncoder, TBREncoder

__all__ = [
    # Base
    "DummyEncoder",
    "SpikeEncoder",
    # Rate
    "PoissonEncoder",
    "PoissonEncoderExpand",
    # Temporal
    "StepForwardEncoder",
    "TBREncoder",
    # Latency-based
    "BurstEncoder",
    "BurstEncoderExpand",
    # Phase
    "PhaseEncoder",
    "PhaseEncoderExpand",
    # Deconv
    "BSAEncoder",
]

__doc__ = """
This module contains various spike encoders for EEG data preprocessing.
The encoders are designed to convert continuous EEG signals into spike trains suitable for spiking neural networks (SNNs).
"""
