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
        # Créer uniquement le répertoire des logs au niveau racine
        (self.base_dir / "logs").mkdir(parents=True, exist_ok=True)

    def create_case_directory(self, case_id: str) -> Path:
        """
        Create and return a standardized directory structure for a case.

        Args:
            case_id (str): Unique identifier for the case

        Returns:
            Path: Path to the created case directory
        """
        case_dir = self.base_dir / case_id
        subdirs = ["dicom", "nifti", "preprocessed", "features"]

        for subdir in subdirs:
            (case_dir / subdir).mkdir(parents=True, exist_ok=True)

        return case_dir

    def get_case_files(self, case_dir: Path, pattern: str = "*.nii.gz") -> List[Path]:
        """
        Retrieve all files matching a pattern in a case directory.

        Args:
            case_dir (Path): Directory to search in
            pattern (str, optional): Glob pattern to match. Defaults to "*.nii.gz"

        Returns:
            List[Path]: List of matching file paths
        """
        return list(case_dir.glob(pattern))

    def save_processing_metadata(self, case_dir: Path, metadata: Dict) -> None:
        """
        Save processing metadata for a case.

        Args:
            case_dir (Path): Case directory where metadata will be stored
            metadata (Dict): Metadata to save
        """
        metadata_file = case_dir / "processing_metadata.json"
        metadata['timestamp'] = datetime.now().isoformat()

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def cleanup_temp_files(self, temp_dir: Path) -> None:
        """
        Safely remove temporary files and directories.

        Args:
            temp_dir (Path): Directory containing temporary files
        """
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

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
        if file_path.suffix in ['.nii', '.gz']:
            try:
                import nibabel as nib
                img = nib.load(str(file_path))
                img.header  # Verify header is readable
                return True
            except Exception:
                return False

        return True

    def get_processing_state(self, case_dir: Path) -> Dict:
        """
        Retrieve the current processing state of a case.

        Args:
            case_dir (Path): Case directory to check

        Returns:
            Dict: Dictionary containing processing state information
        """
        state_file = case_dir / "processing_state.json"
        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
        return {
            "completed_steps": [],
            "start_time": datetime.now().isoformat(),
            "status": "initialized"
        }

    def update_processing_state(self, case_dir: Path, step: str, status: str = "completed") -> None:
        """
        Update the processing state of a case.

        Args:
            case_dir (Path): Case directory
            step (str): Step that was completed
            status (str, optional): Status of the step. Defaults to "completed"
        """
        state = self.get_processing_state(case_dir)
        state["completed_steps"].append({
            "step": step,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
        state["last_update"] = datetime.now().isoformat()

        with open(case_dir / "processing_state.json", 'w') as f:
            json.dump(state, f, indent=2)
