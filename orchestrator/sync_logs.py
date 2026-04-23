import os
import re
import shutil
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone
import argparse

from uploader.storage import DriveStorage
from uploader.upload import _generate_metadata, _drive_prefix

_drive_folder_cache = {}

def file_exists_in_drive(storage: DriveStorage, drive_path: str) -> bool:
    """Check if a file exists in Google Drive at the given path, using a folder cache."""
    parts = Path(drive_path).parts
    folder_parts, filename = parts[:-1], parts[-1]
    
    parent_id = storage._root_folder_id
    for part in folder_parts:
        # Use private method to quickly traverse/create folders if missing
        parent_id = storage._get_or_create_folder(part, parent_id)
        
    if parent_id not in _drive_folder_cache:
        # Fetch all files in this folder once and cache their names
        query = f"'{parent_id}' in parents and trashed=false"
        existing_names = set()
        page_token = None
        while True:
            results = storage._service.files().list(
                q=query, 
                fields="nextPageToken, files(name)",
                pageToken=page_token
            ).execute()
            
            for f in results.get("files", []):
                existing_names.add(f.get("name"))
                
            page_token = results.get("nextPageToken")
            if not page_token:
                break
                
        _drive_folder_cache[parent_id] = existing_names
        
    return filename in _drive_folder_cache[parent_id]

def add_to_drive_cache(storage: DriveStorage, drive_path: str):
    """Mark a file as successfully uploaded in the cache."""
    parts = Path(drive_path).parts
    folder_parts, filename = parts[:-1], parts[-1]
    
    parent_id = storage._root_folder_id
    for part in folder_parts:
        parent_id = storage._get_or_create_folder(part, parent_id)
        
    if parent_id in _drive_folder_cache:
        _drive_folder_cache[parent_id].add(filename)


def main():
    parser = argparse.ArgumentParser(description="Group log files and upload missing ones to Drive.")
    parser.add_argument("--type", dest="experiment_type", required=True, choices=["interaction", "illumination"], help="Experiment type.")
    parser.add_argument("--notes", default="", help="Optional researcher notes.")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing log files.")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Directory {logs_dir} does not exist. Run from the orchestrator directory.")
        return

    # Regex to match the timestamp format YYYY-MM-DD_HH-MM-SS
    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")

    # 1. Group loose files in logs/ by their timestamp
    for item in logs_dir.iterdir():
        if item.is_file():
            match = ts_pattern.search(item.name)
            if match:
                timestamp_str = match.group(1)
                
                # Check if a directory with this timestamp already exists
                # This ensures we group into `<mission>_<timestamp>` folders if they exist
                target_dir = None
                for d in logs_dir.iterdir():
                    if d.is_dir() and timestamp_str in d.name:
                        target_dir = d
                        break
                
                # If no matching directory exists, create one with the exact timestamp
                if not target_dir:
                    target_dir = logs_dir / timestamp_str
                    target_dir.mkdir(exist_ok=True)
                
                dest = target_dir / item.name
                shutil.move(str(item), str(dest))
                print(f"Moved root file {item.name} to directory {target_dir.name}/")

    # 2. Check each directory in logs/ and upload missing files to drive
    try:
        storage = DriveStorage.from_env()
    except RuntimeError as e:
        print(f"Error initializing Google Drive storage: {e}")
        return

    for item in logs_dir.iterdir():
        if item.is_dir():
            match = ts_pattern.search(item.name)
            if not match:
                continue
            
            timestamp_str = match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            prefix = _drive_prefix(timestamp, args.experiment_type)
            print(f"\nProcessing experiment directory: {item.name}")

            # Check and upload metadata.json first
            metadata_drive_path = f"{prefix}/metadata.json"
            if not file_exists_in_drive(storage, metadata_drive_path):
                print("  Uploading missing metadata.json...")
                metadata = _generate_metadata(item, args.experiment_type, timestamp, args.notes)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                    json.dump(metadata, tmp, indent=2)
                    tmp.flush()
                    tmp_name = tmp.name
                storage.upload_file(Path(tmp_name), metadata_drive_path)
                add_to_drive_cache(storage, metadata_drive_path)
                os.remove(tmp_name)
            else:
                print("  Already exists: metadata.json")
            
            # Check and upload all files within the directory
            files = [f for f in item.rglob("*") if f.is_file()]
            for file_path in files:
                relative = file_path.relative_to(item)
                drive_path = f"{prefix}/{relative}"
                
                if not file_exists_in_drive(storage, drive_path):
                    print(f"  Uploading missing {relative}...")
                    storage.upload_file(file_path, drive_path)
                    add_to_drive_cache(storage, drive_path)
                else:
                    print(f"  Already exists: {relative}")

if __name__ == "__main__":
    main()
