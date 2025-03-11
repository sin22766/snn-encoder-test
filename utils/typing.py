from datetime import datetime
from typing import List, TypedDict


class SeizureInfo(TypedDict):
    start_time: int
    end_time: int


class FileInfo(TypedDict):
    name: str
    channels_set_idx: int
    start_time: datetime
    end_time: datetime
    seizures: List[SeizureInfo]


class PatientSummary(TypedDict):
    """
    The PatientSummary dictionary

    Attributes:
        patient_id: The patient ID.
        sampling_rate: The sampling rate.
        channels_list: The list of channels.
        files: The list of files.
    """

    patient_id: str
    sampling_rate: int
    channels_list: List[List[str]]
    files: List[FileInfo]