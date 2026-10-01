"""Clean raw NYC Yellow Taxi Parquet files and save them as CSV, following test_notebook.ipynb."""

from collections.abc import Iterable
from pathlib import Path

import click
import pandas as pd

from .config import DEFAULT_MONTHS, PROCESSED_DIR, RAW_DIR, build_dataset_filename

# 3 = No charge, 4 = Dispute, 5 = Unknown, 6 = Voided trip
EXCLUDED_PAYMENT_TYPES = [3, 4, 5, 6]
DROPPED_COLUMNS = ["passenger_count", "RatecodeID", "store_and_fwd_flag", "payment_type"]
DATETIME_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]

MIN_DURATION_MIN = 1
MAX_DURATION_MIN = 180
MAX_TRIP_DISTANCE = 150
MAX_FARE_AMOUNT = 600


def add_duration(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the trip duration in minutes, calculated from pickup and dropoff times."""
    duration = frame["tpep_dropoff_datetime"] - frame["tpep_pickup_datetime"]
    return frame.assign(duration_min=duration.dt.total_seconds() / 60)


def clean_trips(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid trips and columns that are not needed downstream."""
    cleaned = frame[~frame["payment_type"].isin(EXCLUDED_PAYMENT_TYPES)]
    cleaned = cleaned.drop(columns=DROPPED_COLUMNS)

    cleaned = add_duration(cleaned)
    cleaned = cleaned[cleaned["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN)]
    cleaned = cleaned.drop(columns=DATETIME_COLUMNS)
    cleaned["duration_min"] = cleaned["duration_min"].round(2)

    cleaned = cleaned[cleaned["trip_distance"].between(0, MAX_TRIP_DISTANCE)]
    cleaned = cleaned[cleaned["fare_amount"].between(0, MAX_FARE_AMOUNT)]
    # Comparisons with NaN are False, so this also drops trips without an Airport_fee.
    cleaned = cleaned[cleaned["Airport_fee"] >= 0]
    cleaned = cleaned[cleaned["total_amount"] >= 0]
    cleaned = cleaned.drop(columns="total_amount")
    return cleaned.reset_index(drop=True)


def transform_month(
    month: str,
    raw_dir: Path = RAW_DIR,
    output_dir: Path = PROCESSED_DIR,
) -> Path:
    """Read one month of raw Parquet data, clean it and write it to the processed directory as CSV."""
    input_path = raw_dir / build_dataset_filename(month)
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Download it first with: python -m src.download --months {month}"
        )

    raw = pd.read_parquet(input_path)
    cleaned = clean_trips(raw)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name.replace(".parquet", "_clean.csv")
    cleaned.to_csv(output_path, index=False)

    print(f"{month}: kept {len(cleaned):,} of {len(raw):,} rows -> {output_path}")
    return output_path


def transform_months(
    months: Iterable[str],
    raw_dir: Path = RAW_DIR,
    output_dir: Path = PROCESSED_DIR,
) -> list[Path]:
    """Clean all requested months and return the written file paths."""
    return [transform_month(month, raw_dir, output_dir) for month in months]


@click.command()
@click.option(
    "--months",
    multiple=True,
    help="Month to transform as YYYY-MM. Repeat the option, for example --months 2025-01 --months 2025-02.",
)
def main(months: tuple[str, ...]) -> None:
    transform_months(months or DEFAULT_MONTHS)


if __name__ == "__main__":
    main()
