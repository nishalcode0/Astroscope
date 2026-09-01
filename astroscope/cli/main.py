"""CLI entry point for Astroscope computational observatory."""

import argparse
import sys
from typing import List, Optional

from astroscope.archive.hubble import HubbleAdapter
from astroscope.ingestion.core import ingest_product


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
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Retrieve raw data products from archive references",
    )
    ingest_parser.add_argument(
        "--obs-id",
        required=True,
        help="Mission observation ID to ingest (e.g. n4eya1020)",
    )
    ingest_parser.add_argument(
        "--product-id",
        help="Specific product ID to ingest (e.g. n4eya1020_mos.fits)",
    )
    ingest_parser.add_argument(
        "--mission",
        default="HST",
        help="Mission to query (default: HST)",
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

    if parsed_args.command == "ingest":
        if not parsed_args.product_id:
            print("Error: --product-id is required for ingestion. Single-product ingestion is strictly enforced.")
            return 1

        print(f"Ingesting observation {parsed_args.obs_id}...")

        if parsed_args.mission.upper() != "HST":
            print(f"Error: Mission {parsed_args.mission} is not supported yet.")
            return 1

        adapter = HubbleAdapter()
        try:
            obs = adapter.get_observation_metadata(parsed_args.obs_id)
            if obs.mast_obsid is None:
                print("Error: Observation has no mast_obsid.")
                return 1

            # Ensure mast_obsid is a native int to avoid numpy.int64 type issues
            mast_obsid = int(obs.mast_obsid)
            products = adapter.get_products(mast_obsid)

            target_product = next((p for p in products if p.product_id == parsed_args.product_id), None)

            if not target_product:
                print(f"Error: Product {parsed_args.product_id} not found in observation {parsed_args.obs_id}.")
                return 1

            print(f"Ingesting {target_product.product_id}...")
            path = ingest_product(target_product, obs)
            print(f"Saved to {path}")
            print("Ingestion complete.")
            return 0
        except Exception as e:
            print(f"Ingestion failed: {e}")
            return 1

    if parsed_args.command in ("discover", "process", "analyze", "candidates"):
        print(f"Astroscope: Command '{parsed_args.command}' is not implemented yet in Session 1.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
