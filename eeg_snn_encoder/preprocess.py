from datetime import timedelta
import math
from pathlib import Path
from typing import List, TypedDict

import h5py
import mne
import numpy as np

from eeg_snn_encoder.parser import PatientSummary

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

EXCLUDE_FILES = [
    "chb12_27.edf",
    "chb12_28.edf",
    "chb12_29.edf",
    "chb13_40.edf",
    "chb16_18.edf",
]

def preprocess_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Preprocess the raw EEG data. Apply bandpass and notch filters.

    Args:
        raw: The raw EEG data.

    Returns:
        raw: The preprocessed raw EEG data.
    """

    # Apply bandpass filter between 0.5 and 80 Hz
    raw.filter(l_freq=0.5, h_freq=80, fir_design="firwin", verbose=False)

    # Apply notch filter at 50 and 60 Hz to remove powerline noise
    raw.notch_filter(freqs=[60], fir_design="firwin", verbose=False)

    return raw

def match_channels(raw: mne.io.Raw, channels: List[str] = COMMON_CHANNELS) -> mne.io.Raw:
    """
    Match the given channels with the channels present in the raw data.

    Args:
        raw: The raw data.
        channels: The channels to match.

    Returns:
        result: The raw data that matches the given channels.
    """

    # Deduplicate T8-P8 channel which has T8-P8-0 and T8-P8-1
    if "T8-P8-0" in raw.ch_names and "T8-P8-1" in raw.ch_names:
        raw.drop_channels("T8-P8-1")
        raw.rename_channels({"T8-P8-0": "T8-P8"})

    # Drop channels that are not in the common set and dummy channels
    raw.drop_channels([ch for ch in raw.ch_names if ch.startswith("--") or ch not in channels])

    # Reorder the channels
    raw.reorder_channels(channels)

    return raw


def filter_by_channels(
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

class FileWindowInfo(TypedDict):
    file: str
    total_time: int
    total_windows: int

class IctalFileWindowInfo(FileWindowInfo):
    start_time: int
    end_time: int

class CHBMITPreprocessor:
    def __init__(self, dataset_info: List[PatientSummary], channels: List[str] = COMMON_CHANNELS, exclude_files: List[str] = EXCLUDE_FILES):
        """
        Initialize the CHBMIT preprocessor.

        Parameters
        ----------
        dataset_info : List[PatientSummary]
            List of patient summaries containing EEG data.
        channels : List[str], optional
            List of channels to filter by, by default COMMON_CHANNELS.
        """
        self._dataset_info = filter_by_channels(dataset_info, channels)
        self._channels = channels
        self._exclude_files = exclude_files
    
    def list_ictal(self, window_size: int = 8, sliding_step: int = 4) -> List[IctalFileWindowInfo]:
        """
        List ictal files in the dataset.

        Parameters
        ----------
        window_size : int, optional
            Size of the window in seconds, by default 8.
        sliding_step : int, optional
            Sliding step of the window in seconds, by default 4.

        Returns
        -------
        List[IctalFileWindowInfo]
            List of ictal files with their metadata.
        """
        ictal_list: List[IctalFileWindowInfo] = []

        for patient in self._dataset_info:
            for file in patient["files"]:
                if file["filename"] in self._exclude_files:
                    continue

                for seizure in file["seizures"]:
                    seizure_time = seizure["end_time"] - seizure["start_time"]
                    total_windows = (seizure_time - window_size) // sliding_step + 1

                    ictal_list.append({
                        "file": file["filename"],
                        "start_time": seizure["start_time"],
                        "end_time": seizure["end_time"],
                        "total_time": seizure_time,
                        "total_windows": total_windows,
                    })

        return ictal_list
    
    def list_interictal(self, ictal_list: List[IctalFileWindowInfo], hour_gap: int = 4, window_size: int = 8, sliding_step: int = 8) -> List[FileWindowInfo]:
        """
        List interictal files in the dataset.

        Parameters
        ----------
        ictal_list : List[IctalFileWindowInfo]
            List of ictal files.
        hour_gap : int, optional
            Minimum gap in hours between seizures to consider a file interictal, by default 4.
        window_size : int, optional
            Size of the window in seconds, by default 8.

        Returns
        -------
        List[FileWindowInfo]
            List of interictal files with their metadata.
        """
        interictal_files: List[FileWindowInfo] = []
        
        for patient in self._dataset_info:
            patient_ictal = [ictal["file"] for ictal in ictal_list if ictal["file"].startswith(patient["patient_id"])]
            patient_file = [file["filename"] for file in patient["files"]]
            
            for i, file in enumerate(patient_file):
                in_range = False
                file_duration = patient["files"][i]["end_time"] - patient["files"][i]["start_time"]

                for ictal in patient_ictal:
                    ictal_idx = patient_file.index(ictal)

                    if i - ictal_idx == 0:
                        in_range = True
                        break
                    elif i - ictal_idx > 0:
                        diff = patient["files"][i]["start_time"] - patient["files"][ictal_idx]["end_time"]
                    else:
                        diff = patient["files"][ictal_idx]["start_time"] - patient["files"][i]["end_time"]
                    if diff.seconds <= hour_gap * 3600:
                        in_range = True
                        break

                if not in_range:
                    interictal_files.append({
                        "file": file,
                        "total_time": file_duration.seconds,
                        "total_windows": (file_duration.seconds - window_size) // sliding_step + 1
                    })
        
        return interictal_files
    

    def save_dataset(
        self,
        dataset_path: Path,
        ictal_list: List[IctalFileWindowInfo],
        interictal_list: List[dict],
        window_size: int = 8,
        sliding_step: int = 4,
        random_seed: int = 20250101,
        sample_size: int = 2509,
    ):
        """
        Merge ictal and interictal data, label them, and save to a single HDF5 file.

        Args:
            dataset_path: Path to the dataset (as str or Path).
            ictal_list: List of ictal files with seizure metadata.
            interictal_list: List of interictal files (far from seizures).
            window_size: Window size in seconds.
            sliding_step: Sliding step in seconds for ictal.
            random_seed: Random seed for reproducibility.
            sample_size: Number of interictal samples to include.
        """
        np.random.seed(random_seed)

        all_data = []
        all_labels = []
        all_manifest = []

        print("=== Processing ictal data ===")
        for ictal in ictal_list:
            print("→", ictal["file"])
            patient_id = ictal["file"].split("_")[0][:5]
            file_path: Path = dataset_path / patient_id / ictal["file"]
            raw = mne.io.read_raw_edf(file_path, preload=True, include=COMMON_CHANNELS, verbose=40)
            raw = preprocess_raw(raw)
            raw = match_channels(raw)

            for i in range(ictal["total_windows"]):
                start_time = ictal["start_time"] + i * sliding_step
                end_time = start_time + window_size

                data = raw.get_data(picks=COMMON_CHANNELS, tmin=start_time, tmax=end_time)
                all_data.append(data)
                all_labels.append(1)
                all_manifest.append({
                    "file": ictal["file"],
                    "start_time": start_time,
                    "end_time": end_time,
                    "label": 1,
                })

        print("=== Processing interictal data ===")
        sample_per_file = math.ceil(sample_size / len(interictal_list))

        for interictal in interictal_list:
            print("→", interictal["file"])
            patient_id = interictal["file"].split("_")[0][:5]
            file_path = dataset_path / patient_id / interictal["file"]
            raw = mne.io.read_raw_edf(file_path.as_posix(), preload=True, include=COMMON_CHANNELS, verbose=40)
            raw = preprocess_raw(raw)
            raw = match_channels(raw)

            total_seconds = int(raw.n_times // raw.info["sfreq"])
            selected_times = []

            while len(selected_times) < sample_per_file:
                start_time = np.random.randint(0, total_seconds - window_size)
                end_time = start_time + window_size

                if all(abs(start_time - t) > window_size for t in selected_times):
                    data = raw.get_data(picks=COMMON_CHANNELS, tmin=start_time, tmax=end_time)

                    if data.shape[1] != window_size * raw.info["sfreq"]:
                        print("⚠ Skipping", interictal["file"], "due to shape mismatch:", data.shape)
                        continue

                    all_data.append(data)
                    all_labels.append(0)
                    all_manifest.append({
                        "file": interictal["file"],
                        "start_time": start_time,
                        "end_time": end_time,
                        "label": 0,
                    })
                    selected_times.append(start_time)

        # Final preparation
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        all_manifest = np.array(all_manifest, dtype="S")

        # Shuffle
        indices = np.random.permutation(len(all_data))
        all_data = all_data[indices]
        all_labels = all_labels[indices]
        all_manifest = all_manifest[indices]

        output_dir = dataset_path / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_dir / "windowed.h5", "w") as f:
            f.create_dataset("data", data=all_data)
            f.create_dataset("labels", data=all_labels)
            f.create_dataset("info", data=all_manifest)
            f.create_dataset("channels", data=np.array(COMMON_CHANNELS, dtype="S"))
        
