from utils import info, dataset
import os
import pandas as pd


def main():
    print("Parsing all summary files...")
    summary_info = info.parse_all_summary_files("./CHB-MIT")

    print("Filtering common channels...")
    filtered_summary_info = info.filter_common_channels(summary_info)

    print("Listing all ictal files...")
    ictal = dataset.list_ictal(filtered_summary_info)

    print("Listing all interictal files...")
    interictal = dataset.list_interictal(filtered_summary_info, ictal)

    print("Saving interictal files...")
    dataset.save_interictal("./CHB-MIT", interictal)
    

if __name__ == "__main__":
    main()
