"""CLI entry point for Astroscope computational observatory."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import numpy as np

from astroscope.archive.hubble import HubbleAdapter
from astroscope.ingestion.core import ingest_product
from astroscope.processing.algorithms import (
    build_snr_map,
    estimate_background,
    estimate_noise,
)
from astroscope.processing.catalog import (
    measure_aperture_fluxes,
    measure_sources,
)
from astroscope.processing.core import (
    ProcessingError,
    compute_statistics,
    load_science_image,
)
from astroscope.processing.detection import (
    create_detection_mask,
    filter_sources,
    label_sources,
)
from astroscope.processing.export import export_sources_csv


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
    process_parser = subparsers.add_parser(
        "process",
        help="Process and detect astronomical sources in a science image",
    )
    process_parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw FITS file to process",
    )
    process_parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="SNR detection threshold (default: 5.0)",
    )
    process_parser.add_argument(
        "--min-pixels",
        type=int,
        default=3,
        help="Minimum number of connected pixels for a source",
    )
    process_parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for exporting the detected source catalog as CSV",
    )
    process_parser.add_argument(
        "--aperture-radius",
        type=float,
        help="Optional circular aperture radius in pixels",
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
            print(
                "Error: --product-id is required for ingestion. "
                "Single-product ingestion is strictly enforced."
            )
            return 1

        print(f"Ingesting observation {parsed_args.obs_id}...")

        if parsed_args.mission.upper() != "HST":
            print(
                f"Error: Mission {parsed_args.mission} "
                "is not supported yet."
            )
            return 1

        adapter = HubbleAdapter()

        try:
            obs = adapter.get_observation_metadata(parsed_args.obs_id)

            if obs.mast_obsid is None:
                print("Error: Observation has no mast_obsid.")
                return 1

            mast_obsid = int(obs.mast_obsid)
            products = adapter.get_products(mast_obsid)

            target_product = next(
                (
                    product
                    for product in products
                    if product.product_id == parsed_args.product_id
                ),
                None,
            )

            if not target_product:
                print(
                    f"Error: Product {parsed_args.product_id} "
                    f"not found in observation {parsed_args.obs_id}."
                )
                return 1

            print(f"Ingesting {target_product.product_id}...")

            path = ingest_product(target_product, obs)

            print(f"Saved to {path}")
            print("Ingestion complete.")

            return 0

        except Exception as e:
            print(f"Ingestion failed: {e}")
            return 1

    if parsed_args.command == "process":
        input_path = Path(parsed_args.input)

        print(f"Loading science image from {input_path}...")

        try:
            science_image = load_science_image(input_path)
            stats = compute_statistics(science_image)

            print(
                f"Mission: "
                f"{science_image.header.get('TELESCOP', 'Unknown')}"
            )
            print(
                f"Instrument: "
                f"{science_image.header.get('INSTRUME', 'Unknown')}"
            )
            print(
                f"Filter: "
                f"{science_image.header.get('FILTER', 'Unknown')}"
            )

            print("-" * 40)

            print("Image Statistics:")
            print(f"  Shape: {stats.shape}")
            print(f"  Finite pixels: {stats.finite_pixels}")
            print(f"  NaN/Masked pixels: {stats.nan_pixels}")

            if stats.finite_pixels > 0:
                print(f"  Min: {stats.min_val:.4g}")
                print(f"  Max: {stats.max_val:.4g}")
                print(f"  Mean: {stats.mean_val:.4g}")
                print(f"  Median: {stats.median_val:.4g}")
                print(f"  Std Dev: {stats.std_val:.4g}")
            else:
                print("  No finite pixels found in the image.")

            # Run the source-detection pipeline only when real image
            # data is available. This preserves the existing CLI
            # statistics contract used by the test suite.
            if isinstance(science_image.data, np.ndarray):
                image = science_image.data

                background = estimate_background(image)
                noise = estimate_noise(image)

                snr_map = build_snr_map(image)

                detection_mask = create_detection_mask(
                    snr_map,
                    threshold=parsed_args.threshold,
                )

                labels, source_count = label_sources(detection_mask)

                labels, filtered_count = filter_sources(
                    labels,
                    source_count,
                    min_pixels=parsed_args.min_pixels,
                )

                sources = measure_sources(
                    image,
                    snr_map,
                    labels,
                    filtered_count,
                    background=background,
                )

                # Optional aperture photometry.
                if parsed_args.aperture_radius is not None:
                    aperture_fluxes = measure_aperture_fluxes(
                        image,
                        sources,
                        radius=parsed_args.aperture_radius,
                        background=background,
                    )

                    # Source is frozen, so create new Source instances
                    # instead of modifying the existing objects.
                    sources = [
                        replace(
                            source,
                            aperture_flux=aperture_fluxes[source.source_id],
                        )
                        for source in sources
                    ]

                print("-" * 40)

                print("Processing:")
                print(f"  Background: {background:.6f}")
                print(f"  Noise: {noise:.6f}")
                print(
                    f"  Detection threshold: "
                    f"{parsed_args.threshold:.2f}"
                )
                print(
                    f"  Minimum source pixels: "
                    f"{parsed_args.min_pixels}"
                )
                print(f"  Detected sources: {len(sources)}")

                if parsed_args.aperture_radius is not None:
                    print(
                        f"  Aperture radius: "
                        f"{parsed_args.aperture_radius:.2f} pixels"
                    )

                print("-" * 40)

                print("Source Catalog:")

                for source in sources:
                    line = (
                        f"Source {source.source_id:2d}: "
                        f"x={source.x_centroid:7.2f}, "
                        f"y={source.y_centroid:7.2f}, "
                        f"pixels={source.pixel_count:4d}, "
                        f"flux={source.background_subtracted_flux:10.2f}, "
                        f"peak={source.background_subtracted_peak:8.2f}, "
                        f"SNR={source.peak_snr:6.2f}"
                    )

                    if parsed_args.aperture_radius is not None:
                        line += (
                            f", aperture_flux="
                            f"{source.aperture_flux:10.2f}"
                        )

                    print(line)

                if parsed_args.output is not None:
                    export_sources_csv(
                        sources,
                        parsed_args.output,
                    )

                    print("-" * 40)
                    print(
                        f"Catalog exported to {parsed_args.output}"
                    )

            return 0

        except ProcessingError as e:
            print(f"Error: {e}")
            return 1

        except ValueError as e:
            print(f"Error: {e}")
            return 1

    if parsed_args.command in ("discover", "analyze", "candidates"):
        print(
            f"Astroscope: Command '{parsed_args.command}' "
            "is not implemented yet in Session 1."
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())