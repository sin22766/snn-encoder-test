import math
import os
import h5py
import mne
from typing import List, TypedDict
from utils.constant import COMMON_CHANNELS, EXCLUDE_FILES
from utils.typing import PatientSummary
import numpy as np


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


class IctalInfo(TypedDict):
    file: str
    start_time: int
    end_time: int
    total_time: int
    total_windows: int

def list_ictal(summary_info: List[PatientSummary], window_size: int = 8, sliding_step: int = 4, exclude_files: List[str] = EXCLUDE_FILES) -> List[IctalInfo]:
    """
    Search for the ictal files in the dataset.

    Args:
        dataset_path: The path to the dataset.
        window_size: The size of the window.
        sliding_step: The sliding step of the window.
        exclude_files: The files to exclude.

    Returns:
        ictal_list: The list of ictal files.
    """

    ictal_list: List[IctalInfo] = []

    for patient in summary_info:
        for file in patient["files"]:
            if file["name"] in exclude_files:
                continue

            for seizure in file["seizures"]:
                seizure_time = seizure["end_time"] - seizure["start_time"]

                ictal_list.append({
                    "file": file["name"],
                    "start_time": seizure["start_time"],
                    "end_time": seizure["end_time"],
                    "total_time": seizure_time,
                    "total_windows": (seizure_time - window_size) // sliding_step + 1
                })

    return ictal_list


class InterictalInfo(TypedDict):
    file: str
    total_time: int
    total_windows: int


def list_interictal(summary_info: List[PatientSummary], ictal_list: List[IctalInfo], hour_gap: int = 4, window_size: int = 8, sliding_step: int = 4, exclude_files: List[str] = EXCLUDE_FILES) -> List[InterictalInfo]:
    """
    Search for the interictal files in the dataset.

    Args:
        summary_info: The summary information.
        ictal_list: The list of ictal files.
        window_size: The size of the window.
        sliding_step: The sliding step of the window.
        exclude_files: The files to exclude.

    Returns:
        interictal_list: The list of interictal files.
    """

    interictal_files: List[InterictalInfo] = []

    for patient in summary_info:
        patient_ictal = [ictal["file"] for ictal in ictal_list if ictal["file"].startswith(patient["patient_id"])]
        patient_file = [file["name"] for file in patient["files"]]
        
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


def preprocess_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Preprocess the raw data. Apply bandpass and notch filters.

    Args:
        raw: The raw data.

    Returns:
        raw: The preprocessed raw data.
    """

    # Apply bandpass filter between 0.5 and 80 Hz
    raw.filter(l_freq=0.5, h_freq=80, fir_design="firwin", verbose=False)

    # Apply notch filter at 50 and 60 Hz to remove powerline noise
    raw.notch_filter(freqs=[60], fir_design="firwin", verbose=False)

    return raw


def save_ictal(dataset_path: str, ictal_list: List[IctalInfo], window_size: int = 8, sliding_step: int = 4):
    """
    Prepare the ictal data for the dataset and save to h5 file.

    Args:
        dataset_path: The path to the dataset.
        ictal_list: The list of ictal files.
        window_size: The size of the window in seconds.
        sliding_step: The sliding step of the window in seconds.
    """

    ictals = []
    ictals_manifest = []

    for ictal in ictal_list:
        print("Processing", ictal["file"])
        patient_id = ictal["file"].split("_")[0][:5]
        raw = mne.io.read_raw_edf(f"{dataset_path}/{patient_id}/{ictal['file']}", preload=True, include=COMMON_CHANNELS, verbose=40)
        raw = preprocess_raw(raw)
        raw = match_channels(raw)

        total_windows = ictal["total_windows"]

        for i in range(total_windows):
            start_time = ictal["start_time"] + i * sliding_step
            end_time = start_time + window_size

            data = raw.get_data(picks=COMMON_CHANNELS, tmin=start_time, tmax=end_time)
            ictals.append(data)

            ictals_manifest.append({
                "file": ictal["file"],
                "start_time": start_time,
                "end_time": end_time
            })

    ictals = np.array(ictals)

    os.makedirs(f"{dataset_path}/processed", exist_ok=True)

    with h5py.File(f"{dataset_path}/processed/ictal.h5", "w") as f:
        f.create_dataset("data", data=ictals)
        f.create_dataset("info", data=np.array(ictals_manifest, dtype="S"))
        f.create_dataset("channels", data=np.array(COMMON_CHANNELS, dtype="S"))


def save_interictal(dataset_path: str, interictal_list: List[InterictalInfo], window_size: int = 8, random_seed: int = 20250101, sample_size: int = 2509):
    """
    Prepare the interictal data for the dataset and save to h5 file.

    Args:
        dataset_path: The path to the dataset.
        interictal_list: The list of interictal files.
        window_size: The size of the window in seconds.
        sliding_step: The sliding step of the window in seconds.
    """

    interictals = []
    interictals_manifest = []

    np.random.seed(random_seed)
    sample_per_file = math.ceil(sample_size / len(interictal_list))

    for interictal in interictal_list:
        print("Processing", interictal["file"])
        patient_id = interictal["file"].split("_")[0][:5]
        raw = mne.io.read_raw_edf(f"{dataset_path}/{patient_id}/{interictal['file']}", preload=True, include=COMMON_CHANNELS, verbose=40)
        raw = preprocess_raw(raw)
        raw = match_channels(raw)

        # Seem like some files have different duration
        total_times = raw.n_times // raw.info["sfreq"]

        # Randomly select start time for the interictal data
        # But prevent the start time from being too close to the ictal data (closer than windows size)

        start_times = []

        while len(start_times) < sample_per_file:
            start_time = np.random.randint(0, total_times - window_size)
            end_time = start_time + window_size

            if len(start_times) == 0 or all(abs(start_time - t) > window_size for t in start_times):
                data = raw.get_data(picks=COMMON_CHANNELS, tmin=start_time, tmax=end_time)

                if data.shape[1] != window_size * raw.info["sfreq"]:
                    print("Skipping", interictal["file"], "due to shape mismatch", f"{data.shape}")
                    continue
                interictals.append(data)

                interictals_manifest.append({
                    "file": interictal["file"],
                    "start_time": start_time,
                    "end_time": end_time
                })

                start_times.append(start_time)
    
    interictals = np.array(interictals)

    choosen_idx = np.random.choice(interictals.shape[0], sample_size, replace=False)

    interictals = interictals[choosen_idx]
    interictals_manifest = [interictals_manifest[i] for i in choosen_idx]

    with h5py.File(f"{dataset_path}/processed/interictal.h5", "w") as f:
        f.create_dataset("data", data=interictals)
        f.create_dataset("info", data=np.array(interictals_manifest, dtype="S"))
        f.create_dataset("channels", data=np.array(COMMON_CHANNELS, dtype="S"))


            


