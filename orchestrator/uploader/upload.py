import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from googleapiclient.errors import HttpError

from .storage import DriveStorage


def _generate_metadata(
    exp_path: Path,
    experiment_type: str,
    timestamp: datetime,
    notes: str,
) -> dict:
    """Return the metadata dict for an experiment.

    Args:
        exp_path: Path to the experiment directory.
        experiment_type: One of "interaction" or "illumination".
        timestamp: UTC timestamp of the upload.
        notes: Optional researcher notes.

    Returns:
        A dict suitable for serialisation as metadata.json.
    """
    return {
        "experiment_name": exp_path.name,
        "date": timestamp.strftime("%Y-%m-%d"),
        "timestamp": timestamp.isoformat(),
        "type": experiment_type,
        "notes": notes,
    }


def _drive_prefix(timestamp: datetime, experiment_type: str) -> str:
    """Return the Drive folder prefix for an experiment, e.g. fls-experiments/2025-04-09/10-30-00_interaction."""
    return (
        f"fls-experiments"
        f"/{timestamp.strftime('%Y-%m-%d')}"
        f"/{timestamp.strftime('%H-%M-%S')}_{experiment_type}"
    )


def cli() -> None:
    """Entry point for the fls-upload command."""
    parser = argparse.ArgumentParser(description="Upload an FLS experiment directory to Google Drive.")
    parser.add_argument("--experiment", required=True, help="Path to the experiment directory.")
    parser.add_argument("--type", dest="experiment_type", required=True, choices=["interaction", "illumination"], help="Experiment type.")
    parser.add_argument("--notes", default="", help="Optional researcher notes.")
    parser.add_argument("--datetime", help="Optional datetime string in Y-m-d_H-M-S format. If not provided, current time is used.")
    args = parser.parse_args()

    exp_path = Path(args.experiment)
    if not exp_path.is_dir():
        sys.exit(f"Error: {exp_path} is not a directory.")

    if args.datetime:
        try:
            timestamp = datetime.strptime(args.datetime, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)
        except ValueError:
            sys.exit(f"Error: --datetime '{args.datetime}' is not in the required format Y-m-d_H-M-S")
    else:
        timestamp = datetime.now(timezone.utc)
        
    prefix = _drive_prefix(timestamp, args.experiment_type)

    try:
        storage = DriveStorage.from_env()
    except RuntimeError as e:
        sys.exit(f"Error: {e}")

    print(f"Uploading to {prefix}/")

    try:
        metadata = _generate_metadata(exp_path, args.experiment_type, timestamp, args.notes)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as tmp:
            json.dump(metadata, tmp, indent=2)
            tmp.flush()
            storage.upload_file(Path(tmp.name), f"{prefix}/metadata.json")
        print("  metadata.json ✓")

        files = [f for f in exp_path.rglob("*") if f.is_file()]
        if not files:
            print("  (no files found in experiment directory)", file=sys.stderr)
            return

        for file_path in files:
            relative = file_path.relative_to(exp_path)
            storage.upload_file(file_path, f"{prefix}/{relative}")
            print(f"\r  {relative} ✓\033[K")

    except HttpError as e:
        sys.exit(f"Error: Drive API request failed ({e.status_code} {e.reason})")

    print(f"\nDone. {len(files) + 1} file(s) stored at: {prefix}")


if __name__ == "__main__":
    cli()
