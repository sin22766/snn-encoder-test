# Common channels that are present in all patients
COMMON_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",  # Left hemisphere
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",  # Left central
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",  # Right central
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",  # Right hemisphere
    "FZ-CZ", "CZ-PZ",  # Midline
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"  # Additional lateral channels
]

EXCLUDE_FILES = [
    "chb12_27.edf",
    "chb12_28.edf",
    "chb12_29.edf",
    "chb13_40.edf",
    "chb16_18.edf",
]
