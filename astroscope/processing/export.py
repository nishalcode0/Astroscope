from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from astroscope.processing.models import Source


def export_sources_csv(
    sources: Iterable[Source],
    output_path: Path,
) -> None:
    """
    Export measured astronomical sources to a CSV catalog.

    Parameters
    ----------
    sources : Iterable[Source]
        Collection of measured astronomical sources.

    output_path : Path
        Destination path for the CSV catalog.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_id",
        "x_centroid",
        "y_centroid",
        "pixel_count",
        "peak_signal",
        "total_signal",
        "peak_snr",
        "background_subtracted_peak",
        "background_subtracted_flux",
        "aperture_flux",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for source in sources:
            writer.writerow(
                {
                    "source_id": source.source_id,
                    "x_centroid": source.x_centroid,
                    "y_centroid": source.y_centroid,
                    "pixel_count": source.pixel_count,
                    "peak_signal": source.peak_signal,
                    "total_signal": source.total_signal,
                    "peak_snr": source.peak_snr,
                    "background_subtracted_peak": (
                        source.background_subtracted_peak
                    ),
                    "background_subtracted_flux": (
                        source.background_subtracted_flux
                    ),
                    "aperture_flux": source.aperture_flux,
                }
            )