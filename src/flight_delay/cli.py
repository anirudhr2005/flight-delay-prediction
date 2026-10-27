"""Shared command-line arguments for the pipeline scripts."""
import argparse

from .config import DEFAULT_SPLIT, DEFAULT_YEAR, SPLITS


def year_parser(description: str, split: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", default=DEFAULT_YEAR, choices=["2018", "2019", "2020", "2021", "2022"],
                        help=f"which Combined_Flights file to use (default {DEFAULT_YEAR})")
    if split:
        parser.add_argument("--split", default=DEFAULT_SPLIT, choices=SPLITS,
                            help=f"train/test split (default {DEFAULT_SPLIT})")
    return parser
