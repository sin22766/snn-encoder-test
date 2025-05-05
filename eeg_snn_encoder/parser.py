from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, TypedDict

from mne.io import read_raw_edf
from ttp import ttp


class SeizureInfo(TypedDict):
    start_time: int
    end_time: int


class FileInfo(TypedDict):
    filename: str
    channels_set_idx: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    seizures: List[SeizureInfo]


class PatientSummary(TypedDict):
    patient_id: str
    sampling_rate: int
    channels_list: List[List[str]]
    files: List[FileInfo]


COMMON_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9",
    "FT9-FT10", "FT10-T8",
]

SUMMARY_FILE_TEMPLATE = """
<group>
Data Sampling Rate: {{ sampling_rate | to_int }} Hz

<group name="channel_infos*">
Channels in EDF Files: {{ _start_ }}
Channels changed: {{ _start_ }}

<group name="channels*" itemize="channel_name">
Channel {{ ignore }}: {{ channel_name }}
</group>

<group name="files*">
File Name: {{ filename }}
File Start Time: {{ start_time }}
File End Time: {{ end_time }}

<group name="seizures*">
Seizure Start Time: {{ start_time | to_int }} seconds
Seizure {{ ignore }} Start Time: {{ start_time | to_int }} seconds
Seizure End Time: {{ end_time | to_int }} seconds
Seizure {{ ignore }} End Time: {{ end_time | to_int }} seconds
</group>

</group>
</group>
</group>
"""


def normalize_time(time_str: str, base_date: datetime) -> datetime:
    """
    Convert a time string (HH:MM:SS) into a datetime object based on a given base date.

    Parameters
    ----------
    time_str : str
        Time in HH:MM:SS format.
    base_date : datetime
        The base date to which the time offset will be added.

    Returns
    -------
    datetime
        A datetime object representing the normalized time.
    """
    h, m, s = map(int, time_str.split(":"))
    return base_date + timedelta(hours=h, minutes=m, seconds=s)


def summary_file_parser(filepath: Path) -> PatientSummary:
    """
    Parse a structured summary file and extract patient EEG metadata including channels,
    files, sampling rate, and seizure events.

    Parameters
    ----------
    filepath : Path
        Path to the structured text summary file.

    Returns
    -------
    PatientSummary
        A dictionary containing patient ID, sampling rate, channel lists, and file info
        including seizure event annotations.
    """
    patient_id = filepath.stem.split("-")[0]
    base_date = datetime(2075, 3, 10)
    parser = ttp(data=filepath.read_text(), template=SUMMARY_FILE_TEMPLATE)
    parser.parse()
    parsed = parser.result(format="raw")[0][0][0]

    data: PatientSummary = {
        "patient_id": patient_id,
        "sampling_rate": parsed["sampling_rate"],
        "channels_list": [info["channels"] for info in parsed["channel_infos"]],
        "files": [],
    }

    for idx, info in enumerate(parsed["channel_infos"]):
        for file in info["files"]:
            start_time = normalize_time(file.get("start_time"), base_date) if "start_time" in file else None
            end_time = normalize_time(file.get("end_time"), base_date) if "end_time" in file else None

            if data["files"] and start_time and end_time:
                last_end = data["files"][-1]["end_time"]
                if start_time < last_end:
                    start_time += timedelta(days=1)
                    end_time += timedelta(days=1)
                    base_date += timedelta(days=1)
                if end_time < last_end:
                    end_time += timedelta(days=1)

            file_info: FileInfo = {
                "filename": file["filename"],
                "channels_set_idx": idx,
                "start_time": start_time,
                "end_time": end_time,
                "seizures": file.get("seizures", []),
            }

            data["files"].append(file_info)

    return data


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
    summaries = [summary_file_parser(filepath) for filepath in filelist]

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
