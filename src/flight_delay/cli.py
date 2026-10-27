"""Shared command-line arguments for the pipeline scripts."""
import argparse

from .config import DEFAULT_YEAR


def year_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--year", default=DEFAULT_YEAR, choices=["2018", "2019", "2020", "2021", "2022"],
                        help=f"which Combined_Flights file to use (default {DEFAULT_YEAR})")
    return parser
