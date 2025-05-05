from datetime import timedelta
from pathlib import Path
from typing import List

from mne.io import read_raw_edf

from eeg_snn_encoder.parser import PatientSummary, SummaryParser

COMMON_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
]


def all_summary_file_parser(dataset_path: Path) -> List[PatientSummary]:
    """
    Parse multiple structured summary files and extract patient EEG metadata.
    Also fills in missing start and end times for files.

    Parameters
    ----------
    dataset_path : Path
        Paths to the CHB-MIT dataset directory.

    Returns
    -------
    List[PatientSummary]
        A list of dictionaries containing patient ID, sampling rate, channel lists, and file info
        including seizure event annotations.
    """
    filelist = list(dataset_path.rglob("*summary.txt"))
    filelist.sort()
    summaries = []
    for file_path in filelist:
        parser = SummaryParser(file_path)
        summary = parser.parse()
        summaries.append(summary)

    # Fill in missing start and end times for case 24
    for summary in summaries:
        for file in summary["files"]:
            if file["start_time"] is None or file["end_time"] is None:
                # Read the EDF file to get the start and end times
                edf_path = dataset_path / summary["patient_id"] / file["filename"]
                raw = read_raw_edf(edf_path, preload=True, verbose=40)
                start_time = raw.info["meas_date"]
                end_time = start_time + timedelta(seconds=raw.n_times / raw.info["sfreq"])

                file["start_time"] = start_time
                file["end_time"] = end_time

    return summaries


def filter_channels(
    summaries: List[PatientSummary], channels: List[str] = COMMON_CHANNELS
) -> List[PatientSummary]:
    """
    Filter patient summaries to only include files that contain specified channels.

    Parameters
    ----------
    summaries : List[PatientSummary]
        List of patient summaries.
    channels : List[str]
        List of channels to filter by.

    Returns
    -------
    List[PatientSummary]
        Filtered list of patient summaries.
    """
    result = []
    common_channels_set = set(channels)

    for patient in summaries:
        valid_indices = set()
        for i, ch_set in enumerate(patient["channels_list"]):
            if all(ch in ch_set for ch in common_channels_set):
                valid_indices.add(i)

        if not valid_indices:
            continue

        valid_files = [
            file for file in patient["files"] if file["channels_set_idx"] in valid_indices
        ]

        if not valid_files:
            continue

        filtered_patient: PatientSummary = {
            "patient_id": patient["patient_id"],
            "sampling_rate": patient["sampling_rate"],
            "channels_list": [patient["channels_list"][i] for i in valid_indices],
            "files": valid_files,
        }
        result.append(filtered_patient)

    return result


class CHBMITPreprocessor:
    def __init__(self, dataset_path: Path, channels: List[str] = COMMON_CHANNELS):
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        self._dataset_path = dataset_path
        self._summaries: List[PatientSummary] = filter_channels(
            all_summary_file_parser(dataset_path), channels
        )

    def list_files(self) -> List[Path]:
        """
        List all summary files in the dataset directory.

        Returns
        -------
        List[Path]
            A list of paths to summary files.
        """
        return list(self._dataset_path.rglob("*summary.txt"))
