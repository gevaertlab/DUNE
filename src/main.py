# src/main.py
import click
from pathlib import Path
import yaml
import logging 
from typing import Optional
from src.pipeline.dicom import DicomConverter
from src.pipeline.preprocessing import Preprocessor
from src.pipeline.feature_extraction import FeatureExtractor
from src.utils.logger import PipelineLogger
from src.utils.file_handling import FileHandler

# src/main.py
class Pipeline:

    def __init__(self, base_dir: Path, config_path: Optional[Path] = None, enable_logs: bool = True):
        """
        Initialize the pipeline with all its components.

        Args:
            base_dir (Path): Base directory for all outputs
            config_path (Optional[Path]): Path to configuration file
            enable_logs (bool): Whether to create log files
        """
        self.base_dir = base_dir
        self.config = self._load_config(config_path)

        # Initialize utilities with log setting
        logs_dir = base_dir / "logs" if enable_logs else None
        self.logger = PipelineLogger(logs_dir if logs_dir else Path("/tmp"),
                                    enable_logs=enable_logs)
        self.file_handler = FileHandler(base_dir)

        # Initialize pipeline components
        self.dicom_converter = DicomConverter(self.logger, self.file_handler)
        self.preprocessor = Preprocessor(
            self.logger, self.file_handler, self.config)
        self.feature_extractor = FeatureExtractor(
            self.logger, self.file_handler, self.config)


    def _load_config(self, config_path: Optional[Path]) -> dict:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path (Optional[Path]): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            'preprocessing': {
                'scripts_dir': Path('scripts/preprocessing').absolute(),
                'templates_dir': Path('data/templates').absolute(),
                'skip_brain_extraction': False,
                'keep_preprocessed': False
            },
            'model': {
                'weights_path': Path('data/models/best_model.pt').absolute()
            },
            'output': {
                'enable_logs': True
            }
        }

        if config_path and config_path.exists():
            try:
                import yaml
                with open(config_path) as f:
                    user_config = yaml.safe_load(f)

                # Convert string paths to Path objects
                if 'preprocessing' in user_config:
                    if 'scripts_dir' in user_config['preprocessing']:
                        user_config['preprocessing']['scripts_dir'] = Path(
                            user_config['preprocessing']['scripts_dir']).absolute()
                    if 'templates_dir' in user_config['preprocessing']:
                        user_config['preprocessing']['templates_dir'] = Path(
                            user_config['preprocessing']['templates_dir']).absolute()

                if 'model' in user_config and 'weights_path' in user_config['model']:
                    user_config['model']['weights_path'] = Path(
                        user_config['model']['weights_path']).absolute()

                # Merge avec les valeurs par défaut
                merged_config = default_config.copy()
                for section in user_config:
                    if section in merged_config:
                        merged_config[section].update(user_config[section])
                    else:
                        merged_config[section] = user_config[section]

                return merged_config
            except Exception as e:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.logger.error(
                        f"Error loading config file: {str(e)}")
                return default_config

        return default_config
        

    def process_case(self, input_path: Path, skip_brain_extraction: bool = False, keep_preprocessed: bool = False) -> bool:
        """
        Process a single case through the pipeline.

        Args:
            input_path (Path): Path to the case directory containing DICOM files or to a NIfTI file
            skip_brain_extraction (bool): If True, skip the brain extraction step
            keep_preprocessed (bool): If True, keep the preprocessed NIfTI files in the output

        Returns:
            bool: True if processing was successful, False otherwise
        """
        # Get case ID from input path
        if input_path.suffix in ['.nii', '.gz'] or input_path.suffixes == ['.nii', '.gz']:
            # Pour les fichiers NIfTI, utiliser le nom complet (sans l'extension)
            if len(input_path.suffixes) > 1 and input_path.suffixes[-2] == '.nii' and input_path.suffixes[-1] == '.gz':
                # Cas de .nii.gz
                case_id = input_path.name.replace('.nii.gz', '')
            else:
                # Cas de .nii
                case_id = input_path.name.replace('.nii', '')
            is_single_file = True
        else:
            # Directory with DICOM files or multiple NIfTI files
            case_id = input_path.name
            is_single_file = False

        self.logger.log_step_start("Pipeline processing", case_id)

        try:
            # Simplify output structure based on input type
            if is_single_file:
                # Single file input: create simple output structure
                # Just create the base directory if it doesn't exist
                self.base_dir.mkdir(parents=True, exist_ok=True)
                features_dir = self.base_dir
                if keep_preprocessed:
                    preprocessed_dir = self.base_dir
                else:
                    preprocessed_dir = None
            else:
                # Directory input: create case-specific structure
                case_dir = self.base_dir / case_id
                features_dir = case_dir / "features"
                features_dir.mkdir(parents=True, exist_ok=True)

                if keep_preprocessed:
                    preprocessed_dir = case_dir / "preprocessed"
                    preprocessed_dir.mkdir(parents=True, exist_ok=True)
                else:
                    preprocessed_dir = None

            # Handle different input types
            if is_single_file:
                # Input is a single NIfTI file
                # We don't copy the input file anymore
                nifti_files = {case_id: input_path}
            else:
                # Check if input is a directory with NIfTI files
                nifti_glob = list(input_path.glob('*.nii.gz')) + \
                    list(input_path.glob('*.nii'))
                if nifti_glob:
                    # Input is a directory with NIfTI files
                    nifti_files = {}
                    for nifti_path in nifti_glob:
                        # Utiliser le nom complet du fichier sans extension
                        if nifti_path.suffixes and nifti_path.suffixes[-1] == '.gz' and len(nifti_path.suffixes) > 1:
                            seq_name = nifti_path.name.replace('.nii.gz', '')
                        else:
                            seq_name = nifti_path.name.replace('.nii', '')
                        nifti_files[seq_name] = nifti_path
                else:
                    # Input is a DICOM directory, convert to NIfTI
                    nifti_files = self.dicom_converter.process_case(
                        input_path, self.base_dir / "temp")
                    # We will clean up temp files at the end

            if not nifti_files:
                raise Exception(
                    "No valid input files found or DICOM conversion failed")

            # 2. Preprocessing
            preprocessed_files = {}
            for seq_name, nifti_path in nifti_files.items():
                if nifti_path:
                    # Create a temporary directory for preprocessing
                    temp_preprocessed_dir = self.base_dir / "temp_preprocessed"
                    temp_preprocessed_dir.mkdir(parents=True, exist_ok=True)

                    processed = self.preprocessor.process_file(
                        nifti_path,
                        temp_preprocessed_dir,
                        skip_brain_extraction=skip_brain_extraction
                    )

                    if processed:
                        # If we need to keep preprocessed files, copy to output dir
                        if keep_preprocessed and preprocessed_dir:
                            import shutil
                            # Create a properly named output file
                            dest_file = preprocessed_dir / \
                                f"{seq_name}_preprocessed.nii.gz"
                            shutil.copy(processed, dest_file)
                            preprocessed_files[seq_name] = dest_file
                        else:
                            preprocessed_files[seq_name] = processed

            if not preprocessed_files:
                raise Exception("Preprocessing failed for all sequences")

            # 3. Feature extraction - save features directly to output directory
            for seq_name, preprocessed_file in preprocessed_files.items():
                features_file = self.feature_extractor.extract_features(
                    {seq_name: preprocessed_file},
                    features_dir / f"{seq_name}_features.csv"
                )
                if not features_file:
                    self.logger.log_error(
                        "Feature extraction", seq_name, "Failed to extract features")

            # We skip the feature grouping step as it's not needed anymore

            self.logger.log_step_complete("Pipeline processing", case_id)

            # Clean up temporary files
            temp_dir = self.base_dir / "temp"
            if temp_dir.exists():
                self.file_handler.cleanup_temp_files(temp_dir)

            temp_preprocessed_dir = self.base_dir / "temp_preprocessed"
            if temp_preprocessed_dir.exists():
                self.file_handler.cleanup_temp_files(temp_preprocessed_dir)

            return True

        except Exception as e:
            self.logger.log_error("Pipeline processing", case_id, e)

            # Clean up temp files even on error
            for temp_dir in [self.base_dir / "temp", self.base_dir / "temp_preprocessed"]:
                if temp_dir.exists():
                    self.file_handler.cleanup_temp_files(temp_dir)

            return False
        
        
@click.group()
def cli():
    """Neuroimaging pipeline for feature extraction from MRI sequences."""
    pass


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--skip-brain-extraction', '-s', is_flag=True, help='Skip brain extraction step (use if input is already brain-extracted)')
@click.option('--keep-preprocessed', '-p', is_flag=True, help='Keep preprocessed NIfTI files in the output')
@click.option('--no-logs', is_flag=True, help='Disable writing log files (still shows console output)')
def process(input_path: str, output_path: str, config: Optional[str] = None,
            verbose: bool = False, skip_brain_extraction: bool = False,
            keep_preprocessed: bool = False, no_logs: bool = False):
    """
    Process a case or directory of cases through the pipeline.

    INPUT_PATH: Path to directory containing DICOM files, NIfTI files, or to a single NIfTI file (.nii.gz)
    OUTPUT_PATH: Path where results will be stored
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    config_path = Path(config) if config else None

    # Set up logging level
    log_level = logging.DEBUG if verbose else logging.INFO

    # Initialize pipeline with log settings
    pipeline = Pipeline(output_path, config_path, enable_logs=not no_logs)

    # Process single file or directory
    if input_path.is_file():
        # Input is a single file (probably a NIfTI file)
        if input_path.suffix in ['.nii', '.gz'] or input_path.suffixes == ['.nii', '.gz']:
            pipeline.process_case(input_path,
                                  skip_brain_extraction=skip_brain_extraction,
                                  keep_preprocessed=keep_preprocessed)
        else:
            click.echo(
                f"Error: Input file {input_path} is not a NIfTI file (.nii or .nii.gz)")
    elif input_path.is_dir():
        # Process directory as a single case
        pipeline.process_case(input_path,
                              skip_brain_extraction=skip_brain_extraction,
                              keep_preprocessed=keep_preprocessed)
    else:
        click.echo(f"Error: Input path {input_path} does not exist")

@cli.command()
@click.argument('output_path', type=click.Path())
def init(output_path: str):
    """
    Initialize a new workspace with the required directory structure.

    OUTPUT_PATH: Path where the workspace will be created
    """
    output_path = Path(output_path)

    # Créer uniquement le répertoire des logs
    (output_path / "logs").mkdir(parents=True, exist_ok=True)

    click.echo(f"Workspace initialized at {output_path}")
    click.echo("Created directory structure:")
    click.echo(f"  {output_path}/")
    click.echo(f"  └── logs/")
    click.echo(
        "\nNote: Case-specific directories will be created automatically during processing.")

if __name__ == "__main__":
    cli()
