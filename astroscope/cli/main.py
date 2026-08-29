"""CLI entry point for Astroscope computational observatory."""

import argparse
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="astroscope",
        description=(
            "Astroscope\n"
            "Open-source computational astronomical observatory"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Observatory workflow commands",
    )

    # Subcommand: discover
    subparsers.add_parser(
        "discover",
        help="Discover newly available space telescope observations",
    )

    # Subcommand: ingest
    subparsers.add_parser(
        "ingest",
        help="Retrieve raw data products from archive references",
    )

    # Subcommand: process
    subparsers.add_parser(
        "process",
        help="Process and calibrate raw astronomical observations",
    )

    # Subcommand: analyze
    subparsers.add_parser(
        "analyze",
        help="Extract sources, perform photometry, and build light curves",
    )

    # Subcommand: candidates
    subparsers.add_parser(
        "candidates",
        help="Rank and inspect statistically unusual astronomical candidates",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Execute the Astroscope command-line interface."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 0

    if parsed_args.command in ("discover", "ingest", "process", "analyze", "candidates"):
        print(f"Astroscope: Command '{parsed_args.command}' is not implemented yet in Session 1.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
