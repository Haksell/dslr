import argparse
import pandas as pd
import sys

OPTIMIZER_DEFAULT = "batch"
OPTIMIZER_CHOICES = [OPTIMIZER_DEFAULT, "minibatch", "momentum", "rmsprop", "sgd"]
HOUSE_COLORS = {
    "Gryffindor": "#D62728",
    "Hufflepuff": "#ECB939",
    "Ravenclaw": "#1F77B4",
    "Slytherin": "#2CA02C",
}


def parse_args(description, *, additional_arguments=None, flags=None, optimizer=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("filename", type=str, help="Filename of the CSV.")
    if optimizer:
        parser.add_argument(
            "--optimizer",
            type=str,
            choices=OPTIMIZER_CHOICES,
            default=OPTIMIZER_DEFAULT,
            help="Type of optimizer to use.",
        )
    if additional_arguments:
        for arg in additional_arguments:
            parser.add_argument(arg, type=str)
    if flags:
        for flag in flags:
            parser.add_argument(flag, action="store_true")
    args = parser.parse_args()
    try:
        data = pd.read_csv(args.filename, index_col="Index")
        if additional_arguments or flags:
            return data, args
        else:
            return data
    except FileNotFoundError:
        print(f"Error: File {args.filename} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
