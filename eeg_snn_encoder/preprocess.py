from pathlib import Path


class CHBMITPreprocessor:
    def __init__(self, dataset_path: Path):
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        self._dataset_path = dataset_path

    def parse_summaries(self):
        """
        Parse the summary files to get the seizure start and end times.
        """
        # for summary_file in self._dataset_path.rglob("*summary.txt"):
        #     content = summary_file.read_text()
