from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import List, Optional, TypedDict

from mne.io import read_raw_edf


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


class SummaryParser:
    """Parser for patient EEG summary files.

    This class parses text files containing patient EEG data, including sampling rates,
    channel information, file details, and seizure events.

    Parameters
    ----------
    data_path : Path
        Path to the summary file to parse
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.base_date = datetime(2075, 3, 10)
        self.channels_set = 0
        self.latest_time = None

        self.summary = {
            "patient_id": data_path.stem.split("-")[0],
            "sampling_rate": None,
            "channels_list": [],
            "files": [],
        }

        # Compile regex patterns for better performance
        self._patterns = {
            "number": re.compile(r"\d+"),
            "time": re.compile(r"\d+:\d+:\d+"),
            "channel": re.compile(r"Channel (\d+): (.+)"),
            "file": re.compile(r"File Name: (.+)"),
            "seizure_start": re.compile(r"Seizure\s*\d*\s*Start Time:\s*(\d+)"),
            "seizure_end": re.compile(r"Seizure\s*\d*\s*End Time:\s*(\d+)"),
        }

    def parse(self) -> PatientSummary:
        """Parse the entire summary file.

        Returns
        -------
        PatientSummary
            Structured summary data
        """
        raw = self.data_path.read_text()

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            self._parse_line(line)

        return self.summary

    def _parse_line(self, line: str) -> None:
        """Process a single line from the summary file."""
        if line.startswith("Data Sampling Rate"):
            match = self._patterns["number"].search(line)
            if match:
                self.summary["sampling_rate"] = int(match.group())

        elif self._patterns["channel"].match(line):
            # Ensure the channels list has enough entries
            while len(self.summary["channels_list"]) <= self.channels_set:
                self.summary["channels_list"].append([])

            match = self._patterns["channel"].match(line)
            if match:
                self.summary["channels_list"][self.channels_set].append(match.group(2))

        elif line.startswith("Channels changed"):
            self.channels_set += 1

        elif line.startswith("File Name:"):
            match = self._patterns["file"].match(line)
            if match:
                self.summary["files"].append(
                    {
                        "filename": match.group(1),
                        "start_time": None,
                        "end_time": None,
                        "seizures": [],
                    }
                )

        elif line.startswith("File Start Time:"):
            match = self._patterns["time"].search(line)
            if match:
                time = self._normalize_time(match.group())
                self.summary["files"][-1]["start_time"] = time
                self.latest_time = time

        elif line.startswith("File End Time:"):
            match = self._patterns["time"].search(line)
            if match:
                time = self._normalize_time(match.group())
                self.summary["files"][-1]["end_time"] = time
                self.latest_time = time

        elif self._patterns["seizure_start"].match(line):
            match = self._patterns["seizure_start"].match(line)
            if match:
                time = int(match.group(1))
                self.summary["files"][-1]["seizures"].append(
                    {"start_time": time, "end_time": None}
                )

        elif self._patterns["seizure_end"].match(line):
            if not self.summary["files"] or not self.summary["files"][-1]["seizures"]:
                return

            match = self._patterns["seizure_end"].match(line)
            if match:
                time = int(match.group(1))
                self.summary["files"][-1]["seizures"][-1]["end_time"] = time

    def _normalize_time(self, time_str: str) -> datetime:
        """Convert time string to datetime object.

        Parameters
        ----------
        time_str : str
            Time string in HH:MM:SS format

        Returns
        -------
        datetime
            Normalized datetime object
        """
        h, m, s = map(int, time_str.split(":"))
        time = self.base_date + timedelta(hours=h, minutes=m, seconds=s)

        # Handle day transitions
        if self.latest_time and time < self.latest_time:
            time += timedelta(days=1)
            self.base_date += timedelta(days=1)

        return time


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
