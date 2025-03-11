from datetime import datetime, timedelta
import os
import re
from typing import List

import mne
from utils.constant import COMMON_CHANNELS
from utils.typing import FileInfo, PatientSummary


class SummaryFileParser:
    """Class to handle parsing of patient summary files."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.patient_id = os.path.basename(file_path).split("-")[0]
        self.reference_date = datetime(2075, 3, 10)
        self.previous_time = None
        self.current_channels_set = 0
        self.current_file_index = -1
        
        self.data: PatientSummary = {
            "patient_id": self.patient_id,
            "sampling_rate": None,
            "channels_list": [],
            "files": [],
        }

    def parse(self) -> PatientSummary:
        """Parse the summary file and return structured patient data."""
        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    self._process_line(line)
                    
            return self.data
        except Exception as e:
            raise ValueError(f"Error parsing summary file {self.file_path}: {str(e)}")

    def _process_line(self, line: str) -> None:
        """Process a single line from the summary file."""
        if "Data Sampling Rate" in line:
            self._extract_sampling_rate(line)
        elif re.match(r"Channel \d+:", line):
            self._extract_channel_info(line)
        elif line.startswith("Channels changed"):
            self.current_channels_set += 1
        elif line.startswith("File Name:"):
            self._extract_file_name(line)
        elif line.startswith("File Start Time"):
            self._extract_start_time(line)
        elif line.startswith("File End Time"):
            self._extract_end_time(line)
        elif re.match(r"Seizure (\d+ )?Start Time:", line):
            self._extract_seizure_start(line)
        elif re.match(r"Seizure (\d+ )?End Time:", line):
            self._extract_seizure_end(line)

    def _extract_sampling_rate(self, line: str) -> None:
        """Extract sampling rate from line."""
        match = re.search(r"\d+", line)
        if match:
            self.data["sampling_rate"] = int(match.group())

    def _extract_channel_info(self, line: str) -> None:
        """Extract channel information from line."""
        match = re.match(r"Channel (\d+): (.+)", line)
        if match:
            # Ensure we have a list for the current channel set
            if len(self.data["channels_list"]) <= self.current_channels_set:
                self.data["channels_list"].append([])

            self.data["channels_list"][self.current_channels_set].append(match.group(2))

    def _extract_file_name(self, line: str) -> None:
        """Extract file name from line."""
        match = re.match(r"File Name: (.+)", line)
        if match:
            file_info: FileInfo = {
                "name": match.group(1),
                "channels_set_idx": self.current_channels_set,
                "start_time": None,
                "end_time": None,
                "seizures": [],
            }
            self.data["files"].append(file_info)
            self.current_file_index += 1

    def _extract_start_time(self, line: str) -> None:
        """Extract and normalize file start time."""
        if self.current_file_index < 0:
            return  # No file record created yet
            
        time_str = line.split(":", 1)[1].strip()
        datetime_obj = self._normalize_time(time_str)
        
        self.previous_time = datetime_obj
        self.data["files"][self.current_file_index]["start_time"] = datetime_obj

    def _extract_end_time(self, line: str) -> None:
        """Extract and normalize file end time."""
        if self.current_file_index < 0 or self.previous_time is None:
            return  # No file record created yet or no start time
            
        time_str = line.split(":", 1)[1].strip()
        datetime_obj = self._normalize_time(time_str)
        
        current_file = self.data["files"][self.current_file_index]
        current_file["end_time"] = datetime_obj
        
        self.previous_time = datetime_obj

    def _normalize_time(self, time_str: str) -> datetime:
        """
        Normalize time string to a datetime object, handling edge cases.
        
        Args:
            time_str: Time string in HH:MM:SS format
            
        Returns:
            Normalized datetime object
        """
        # Handle hours >= 24
        hour_parts = time_str.split(":")
        hour = int(hour_parts[0])
        
        # Adjust hours if needed
        adjusted_hour = hour % 24
        day_offset = hour // 24
        
        adjusted_time_str = f"{adjusted_hour:02d}:{hour_parts[1]}:{hour_parts[2]}"
        
        # Create initial datetime
        date_to_use = self.reference_date.date()
        time_obj = datetime.strptime(
            f"{date_to_use} {adjusted_time_str}", "%Y-%m-%d %H:%M:%S"
        )
        
        # Add day offset
        time_obj += timedelta(days=day_offset)
        
        # Ensure chronological order
        if self.previous_time and time_obj < self.previous_time:
            time_obj += timedelta(days=1)
            
        return time_obj

    def _extract_seizure_start(self, line: str) -> None:
        """Extract seizure start time."""
        if self.current_file_index < 0:
            return
            
        match = re.search(r"(\d+) seconds", line)
        if match:
            seizure_start_time = int(match.group(1))
            self.data["files"][self.current_file_index]["seizures"].append(
                {"start_time": seizure_start_time, "end_time": None}
            )

    def _extract_seizure_end(self, line: str) -> None:
        """Extract seizure end time."""
        if self.current_file_index < 0:
            return
            
        current_file = self.data["files"][self.current_file_index]
        if not current_file["seizures"]:
            return  # No seizure record created yet
            
        match = re.search(r"(\d+) seconds", line)
        if match:
            seizure_end_time = int(match.group(1))
            current_file["seizures"][-1]["end_time"] = seizure_end_time


def parse_all_summary_files(dataset_path: str) -> List[PatientSummary]:
    """
    Parse all summary files in the dataset directory.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        List of parsed patient summaries
        
    Raises:
        FileNotFoundError: If dataset path doesn't exist
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
    summary_info: List[PatientSummary] = []
    
    # Find all patient directories
    patient_dirs = [d for d in os.listdir(dataset_path) if d.startswith("chb")]
    
    for patient_dir in patient_dirs:
        patient_path = os.path.join(dataset_path, patient_dir)
        if not os.path.isdir(patient_path):
            continue
            
        summary_file = os.path.join(patient_path, f"{patient_dir}-summary.txt")
        
        if os.path.exists(summary_file):
            try:
                parser = SummaryFileParser(summary_file)
                patient_info = parser.parse()
                summary_info.append(patient_info)
            except Exception as e:
                print(f"Error processing {summary_file}: {e}")

    # Getting the start time and end time of the chb24 patient
    for file in summary_info[23]["files"]:
        raw = mne.io.read_raw_edf(os.path.join(dataset_path, "chb24", file["name"]), preload=True)

        start_time = raw.info["meas_date"]
        end_time = start_time + timedelta(seconds=raw.n_times / raw.info["sfreq"])

        file["start_time"] = start_time
        file["end_time"] = end_time
    
    return summary_info


def filter_common_channels(
    summary_info: List[PatientSummary], channels: List[str] = COMMON_CHANNELS
) -> List[PatientSummary]:
    """
    Filter patient summaries to only include files with common channels.
    
    Args:
        summary_info: List of patient summaries
        channels: List of common channels to filter by
        
    Returns:
        Filtered list of patient summaries
    """
    if not summary_info:
        return []
        
    result = []
    common_channels_set = set(channels)
    
    for patient in summary_info:
        # Create a deep copy of the patient data
        filtered_patient = {
            "patient_id": patient["patient_id"],
            "sampling_rate": patient["sampling_rate"],
            "channels_list": patient["channels_list"].copy(),
            "files": []
        }
        
        # Identify channel sets that contain all common channels
        valid_channel_sets = []
        for i, channel_set in enumerate(patient["channels_list"]):
            if common_channels_set.issubset(set(channel_set)):
                valid_channel_sets.append(i)
        
        # Only include files with valid channel sets
        for file in patient["files"]:
            if file["channels_set_idx"] in valid_channel_sets:
                filtered_patient["files"].append(file.copy())
        
        # Only include patient if they have any valid files
        if filtered_patient["files"]:
            result.append(filtered_patient)
    
    return result
