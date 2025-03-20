# src/pipeline/dicom.py
from pathlib import Path
from typing import Optional, Dict
import dicom2nifti
import logging
import pydicom
from src.utils.logger import PipelineLogger
from src.utils.file_handling import FileHandler


class DicomConverter:
    def __init__(self, logger: PipelineLogger, file_handler: FileHandler):
        self.logger = logger
        self.file_handler = file_handler
        logging.getLogger('dicom2nifti').setLevel(logging.WARNING)

    def _verify_dicom_files(self, dicom_files: list) -> bool:
        try:
            test_file = dicom_files[0]
            ds = pydicom.dcmread(test_file, force=True)
            self.logger.logger.info(
                f"Successfully read DICOM file: {test_file}")
            self.logger.logger.info(
                f"DICOM Series Description: {ds.get('SeriesDescription', 'N/A')}")
            return True
        except Exception as e:
            self.logger.logger.error(f"Error reading DICOM file: {str(e)}")
            return False

    def convert_sequence(self, input_dicom_folder: Path, output_dir: Path) -> Optional[Path]:
        sequence_id = input_dicom_folder.name
        self.logger.log_step_start("DICOM conversion", sequence_id)

        try:
            if not input_dicom_folder.exists():
                raise FileNotFoundError(
                    f"DICOM folder not found: {input_dicom_folder}")

            dicom_files = sorted(list(input_dicom_folder.glob("*.dcm")))
            self.logger.logger.info(
                f"Found {len(dicom_files)} .dcm files in {input_dicom_folder}")

            if not dicom_files:
                raise FileNotFoundError(
                    f"No DICOM files found in {input_dicom_folder}")

            if not self._verify_dicom_files(dicom_files):
                raise ValueError("Invalid DICOM files")

            # Créer les répertoires temporaires et de sortie
            temp_dir = output_dir / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Effectuer la conversion
            self.logger.logger.info(
                f"Starting conversion for {len(dicom_files)} files...")
            try:
                dicom2nifti.convert_directory(
                    str(input_dicom_folder),
                    str(temp_dir),
                    compression=True,
                    reorient=True
                )
            except Exception as conv_error:
                self.logger.logger.error(
                    f"Conversion error details: {str(conv_error)}")
                raise

            nifti_files = list(temp_dir.glob("*.nii.gz"))
            if not nifti_files:
                self.logger.logger.error("Conversion produced no output files")
                raise FileNotFoundError(
                    "No NIfTI file was created during conversion")

            self.logger.logger.info(
                f"Found {len(nifti_files)} converted NIfTI files")

            # Création du chemin de sortie correct en utilisant le répertoire 'nifti'
            final_path = output_dir / "nifti" / f"{sequence_id}.nii.gz"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            nifti_files[0].rename(final_path)

            # Vérifier l'intégrité du fichier
            if not self.file_handler.verify_file_integrity(final_path):
                raise ValueError("Created NIfTI file is corrupted")

            self.logger.log_step_complete("DICOM conversion", sequence_id)
            return final_path

        except Exception as e:
            self.logger.log_error("DICOM conversion", sequence_id, e)
            self.logger.logger.error(
                f"Detailed error in convert_sequence: {str(e)}")
            return None

        finally:
            if 'temp_dir' in locals() and temp_dir.exists():
                self.file_handler.cleanup_temp_files(temp_dir)

    def process_case(self, case_path: Path, output_base: Path) -> Dict[str, Optional[Path]]:
        case_id = case_path.name
        self.logger.log_step_start("Case processing", case_id)

        results = {}
        try:
            # Vérifier que le répertoire du cas existe
            if not case_path.exists():
                raise FileNotFoundError(
                    f"Case directory not found: {case_path}")

            # Lister tous les fichiers DICOM
            dicom_files = list(case_path.glob("**/*.dcm"))
            self.logger.logger.info(
                f"Found {len(dicom_files)} DICOM files in {case_path}")

            if not dicom_files:
                raise FileNotFoundError(f"No DICOM files found in {case_path}")

            # Utiliser directement output_base comme répertoire de sortie
            self.logger.logger.info(
                f"Processing case directory as sequence: {case_path}")
            nifti_file = self.convert_sequence(case_path, output_base)
            if nifti_file:
                results[case_path.name] = nifti_file

            if not any(results.values()):
                raise Exception(
                    f"No sequences were successfully converted in {case_path}")

            self.logger.log_step_complete("Case processing", case_id)
            return results

        except Exception as e:
            self.logger.log_error("Case processing", case_id, e)
            return results
