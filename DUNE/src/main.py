# src/main.py
import click
from pathlib import Path
import yaml
from typing import Optional
from src.pipeline.dicom import DicomConverter
from src.pipeline.preprocessing import Preprocessor
from src.pipeline.feature_extraction import FeatureExtractor
from src.pipeline.feature_grouping import FeatureGrouper
from src.utils.logger import PipelineLogger
from src.utils.file_handling import FileHandler


# src/main.py
class Pipeline:
    def __init__(self, base_dir: Path, config_path: Optional[Path] = None):
        """
        Initialize the pipeline with all its components.

        Args:
            base_dir (Path): Base directory for all outputs
            config_path (Optional[Path]): Path to configuration file
        """
        self.base_dir = base_dir
        self.config = self._load_config(config_path)

        # Initialize utilities
        self.logger = PipelineLogger(base_dir / "logs")
        self.file_handler = FileHandler(base_dir)

        # Initialize pipeline components
        self.dicom_converter = DicomConverter(self.logger, self.file_handler)
        self.preprocessor = Preprocessor(
            self.logger, self.file_handler, self.config)
        self.feature_extractor = FeatureExtractor(
            self.logger, self.file_handler, self.config)
        self.feature_grouper = FeatureGrouper(self.logger, self.file_handler)


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
                'templates_dir': Path('data/templates').absolute()
            },
            'model': {
                'weights_path': Path('data/models/best_model.pt').absolute()
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
                self.logger.logger.error(f"Error loading config file: {str(e)}")
                return default_config

        return default_config


    def process_case(self, input_path: Path) -> bool:
        """
        Process a single case through the complete pipeline.

        Args:
            input_path (Path): Path to the case directory containing DICOM files

        Returns:
            bool: True if processing was successful, False otherwise
        """
        case_id = input_path.name
        self.logger.log_step_start("Pipeline processing", case_id)

        try:
            # Create case directory structure
            case_dir = self.file_handler.create_case_directory(case_id)

            # 1. DICOM to NIfTI conversion
            nifti_files = self.dicom_converter.process_case(
                input_path, case_dir)
            if not nifti_files:
                raise Exception("DICOM conversion failed")

            # 2. Preprocessing - utiliser les sous-répertoires du case_dir
            preprocessed_files = {}
            for seq_name, nifti_path in nifti_files.items():
                if nifti_path:
                    processed = self.preprocessor.process_file(
                        nifti_path,
                        case_dir / "preprocessed"
                    )
                    if processed:
                        preprocessed_files[seq_name] = processed

            if not preprocessed_files:
                raise Exception("Preprocessing failed for all sequences")

            # 3. Feature extraction - utiliser les sous-répertoires du case_dir
            features_file = self.feature_extractor.extract_features(
                preprocessed_files,
                case_dir / "features" / f"{case_id}_features.csv"
            )
            if not features_file:
                raise Exception("Feature extraction failed")

            # 4. Feature grouping - sauvegarder dans le répertoire du cas
            grouped_file = self.feature_grouper.group_features(
                features_file,
                case_dir / "features" / f"{case_id}_grouped.csv"
            )
            if not grouped_file:
                raise Exception("Feature grouping failed")

            self.logger.log_step_complete("Pipeline processing", case_id)
            return True

        except Exception as e:
            self.logger.log_error("Pipeline processing", case_id, e)
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
def process(input_path: str, output_path: str, config: Optional[str] = None, verbose: bool = False):
    """
    Process a case or directory of cases through the pipeline.

    INPUT_PATH: Path to directory containing DICOM files
    OUTPUT_PATH: Path where results will be stored
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    config_path = Path(config) if config else None

    # Initialize pipeline
    pipeline = Pipeline(output_path, config_path)

    # Process single case or directory
    if input_path.is_dir():
        if next(input_path.glob('*.dcm'), None):
            # Single case
            pipeline.process_case(input_path)
        else:
            # Directory of cases
            for case_dir in input_path.iterdir():
                if case_dir.is_dir() and next(case_dir.glob('*.dcm'), None):
                    pipeline.process_case(case_dir)


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
