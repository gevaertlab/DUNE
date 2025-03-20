# src/pipeline/preprocessing.py
from pathlib import Path
import subprocess
from typing import Optional, Dict
import shlex
from src.utils.logger import PipelineLogger
from src.utils.file_handling import FileHandler


class Preprocessor:
    def __init__(self, logger: PipelineLogger, file_handler: FileHandler, config: Dict):
        """
        Initialize the preprocessor with necessary dependencies and configurations.

        Args:
            logger (PipelineLogger): Logger instance for tracking operations
            file_handler (FileHandler): File handler instance for managing files
            config (Dict): Configuration containing paths and parameters
            
        Raises:
            FileNotFoundError: If required scripts or templates are not found
        """
        self.logger = logger
        self.file_handler = file_handler

        # Get paths from config
        scripts_base = Path(config['preprocessing']['scripts_dir'])
        self.scripts = {
            'brain_extract': scripts_base / 'brain_extract.sh',
            'bias_correction': scripts_base / 'bias_correction.sh',
            'standardization': scripts_base / 'brain_standardization.sh'
        }

        # Verify scripts existence
        self._verify_dependencies()

    def _verify_dependencies(self) -> None:
        """
        Verify that all required preprocessing scripts exist.
        
        Raises:
            FileNotFoundError: If any required script is missing
        """
        for script_name, script_path in self.scripts.items():
            if not script_path.exists():
                self.logger.logger.error(f"Script not found: {script_path}")
                raise FileNotFoundError(
                    f"Required script not found: {script_path}")
            elif not script_path.is_file():
                self.logger.logger.error(
                    f"Path exists but is not a file: {script_path}")
                raise FileNotFoundError(
                    f"Required script is not a file: {script_path}")
            else:
                self.logger.logger.debug(f"Found script: {script_path}")

    def _run_processing_step(self,
                             step_name: str,
                             script_path: Path,
                             input_file: Path,
                             output_file: Path,
                             is_standardization: bool = False) -> bool:
        """
        Execute a preprocessing step through a bash script.

        Args:
            step_name (str): Name of the processing step
            script_path (Path): Path to the bash script
            input_file (Path): Input NIfTI file
            output_file (Path): Output file path
            is_standardization (bool): Whether this is the standardization step

        Returns:
            bool: True if processing was successful, False otherwise
        """
        self.logger.log_step_start(step_name, input_file.stem)

        try:
            # Prepare command
            if is_standardization:
                cmd = [
                    "bash", str(script_path),
                    "-t", str(input_file),
                    "-u", str(input_file),
                    "-o", str(output_file)
                ]
            else:
                cmd = [
                    "bash", str(script_path),
                    "-i", str(input_file),
                    "-o", str(output_file)
                ]

            # Log the command being executed
            self.logger.logger.debug(f"Executing command: {' '.join(cmd)}")

            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            if result.stderr:
                self.logger.logger.warning(
                    f"Warning in {step_name}: {result.stderr}")

            self.logger.log_step_complete(step_name, input_file.stem)
            return True

        except subprocess.CalledProcessError as e:
            self.logger.log_error(
                step_name,
                input_file.stem,
                f"Return code: {e.returncode}\nError: {e.stderr}"
            )
            return False


    def process_file(self, input_file: Path, output_dir: Path, skip_brain_extraction: bool = False) -> Optional[Path]:
        """
        Apply all preprocessing steps to a NIfTI file.

        Args:
            input_file (Path): Input NIfTI file to process
            output_dir (Path): Directory for output files
            skip_brain_extraction (bool): If True, skip brain extraction step

        Returns:
            Optional[Path]: Path to the processed file if successful, None otherwise
        """
        sequence_name = input_file.stem
        self.logger.log_step_start("Preprocessing", sequence_name)

        try:
            # Create working directories
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = output_dir / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Define intermediate files
            brain_output = temp_dir / f"{sequence_name}_brain.nii.gz"
            bias_output = temp_dir / f"{sequence_name}_bias.nii.gz"

            # 1. Brain extraction (skip if requested)
            if skip_brain_extraction:
                self.logger.logger.info("Skipping brain extraction as requested")
                # Copy input file to brain_output since we're skipping that step
                import shutil
                shutil.copy(input_file, brain_output)
            else:
                if not self._run_processing_step(
                    "Brain Extraction",
                    self.scripts['brain_extract'],
                    input_file,
                    brain_output
                ):
                    raise Exception("Brain extraction failed")

            # 2. Bias field correction
            if not self._run_processing_step(
                "Bias Correction",
                self.scripts['bias_correction'],
                brain_output,
                bias_output
            ):
                raise Exception("Bias correction failed")

            # 3. Standardization
            if not self._run_processing_step(
                "Standardization",
                self.scripts['standardization'],
                bias_output,
                temp_dir,
                is_standardization=True
            ):
                raise Exception("Standardization failed")

            # Check and rename final file
            norm_file = temp_dir / "normT1.nii.gz"
            if not norm_file.exists():
                raise FileNotFoundError("Normalized file not found")

            final_path = output_dir / f"{sequence_name}_norm.nii.gz"
            norm_file.rename(final_path)

            # Verify final file
            if not self.file_handler.verify_file_integrity(final_path):
                raise ValueError("Processed file is corrupted")

            # Save processing metadata
            metadata = {
                "input_file": str(input_file),
                "output_file": str(final_path),
                "processing_steps": [
                    "bias_correction",
                    "standardization"
                ]
            }

            # Ajouter l'étape de brain extraction seulement si elle a été effectuée
            if not skip_brain_extraction:
                metadata["processing_steps"].insert(0, "brain_extraction")

            # Sauvegardons les métadonnées dans le répertoire de sortie
            self.file_handler.save_processing_metadata(
                output_dir,
                metadata
            )

            self.logger.log_step_complete("Preprocessing", sequence_name)
            return final_path

        except Exception as e:
            self.logger.log_error("Preprocessing", sequence_name, e)
            return None

        finally:
            # Cleanup
            if 'temp_dir' in locals() and temp_dir.exists():
                self.file_handler.cleanup_temp_files(temp_dir)
