import argparse
import pandas as pd
import sys


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("filename", type=str, help="Filename of the CSV.")
    args = parser.parse_args()
    try:
        return pd.read_csv(args.filename, index_col="Index")
    except FileNotFoundError:
        print(f"Error: File {args.filename} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
