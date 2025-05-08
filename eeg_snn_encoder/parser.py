from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import List, Optional, TypedDict


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
                        "channels_set_idx": self.channels_set,
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
