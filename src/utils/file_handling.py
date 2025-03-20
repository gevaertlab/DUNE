# src/utils/file_handling.py
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import json
from datetime import datetime


class FileHandler:
    def __init__(self, base_dir: Path):
        """
        Initialize the file handler.

        Args:
            base_dir (Path): Base directory for all pipeline operations
        """
        self.base_dir = Path(base_dir)
        # Create logs directory only if needed
        (self.base_dir / "logs").mkdir(parents=True, exist_ok=True)

    def create_directory(self, dir_path: Path) -> Path:
        """
        Create a directory if it doesn't exist.

        Args:
            dir_path (str): Path to create

        Returns:
            Path: Created directory path
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def verify_file_integrity(self, file_path: Path) -> bool:
        """
        Verify the integrity of a file based on its type.

        Args:
            file_path (Path): Path to the file to verify

        Returns:
            bool: True if file is valid, False otherwise
        """
        if not file_path.exists():
            return False
        if file_path.stat().st_size == 0:
            return False

        # Specific checks for NIfTI files
        if file_path.suffix in ['.nii', '.gz'] or file_path.suffixes == ['.nii', '.gz']:
            try:
                import nibabel as nib
                img = nib.load(str(file_path))
                img.header  # Verify header is readable
                return True
            except Exception:
                return False

        return True

    def cleanup_temp_files(self, temp_dir: Path) -> None:
        """
        Safely remove temporary files and directories.

        Args:
            temp_dir (Path): Directory containing temporary files
        """
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def save_metadata(self, file_path: Path, metadata: Dict) -> None:
        """
        Save metadata for a file.

        Args:
            file_path (Path): Path to the file to save metadata for
            metadata (Dict): Metadata to save
        """
        metadata['timestamp'] = datetime.now().isoformat()

        # Save as a JSON file with the same name
        metadata_file = file_path.with_suffix('.json')

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


    def save_processing_metadata(self, output_dir: Path, metadata: Dict) -> None:
        """
        Save processing metadata for a case (backward compatibility method).

        Args:
            output_dir (Path): Directory where metadata will be stored
            metadata (Dict): Metadata to save
        """
        # Create a metadata filename in the output directory
        metadata_file = output_dir / "processing_metadata.json"

        # Add timestamp
        metadata['timestamp'] = datetime.now().isoformat()

        # Save to file
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
