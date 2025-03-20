# src/pipeline/feature_grouping.py
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
from src.utils.logger import PipelineLogger
from src.utils.file_handling import FileHandler

class FeatureGrouper:
    """
    Handles the grouping of sequence-level features into case-level features.
    Combines features from multiple MRI sequences into a single feature vector per case.
    """

    def __init__(self, logger: PipelineLogger, file_handler: FileHandler):
        """
        Initialize the feature grouper.

        Args:
            logger (PipelineLogger): Logger instance for tracking operations
            file_handler (FileHandler): File handler instance for managing files
        """
        self.logger = logger
        self.file_handler = file_handler

    def _extract_case_id(self, sequence_id: str) -> str:
        """
        Extract case ID from sequence identifier.

        Args:
            sequence_id (str): Identifier in format 'case_id__sequence_name'

        Returns:
            str: Extracted case ID
        """
        return sequence_id.split('__')[0]

    def _validate_input_data(self, df: pd.DataFrame) -> bool:
        """
        Validate input DataFrame structure and content.

        Args:
            df (pd.DataFrame): Input DataFrame to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required columns
            if 'eid__sequence' not in df.columns:
                self.logger.logger.error(
                    "Missing required column 'eid__sequence'")
                return False

            # Check for empty DataFrame
            if df.empty:
                self.logger.logger.error("Empty input DataFrame")
                return False

            # Check for valid sequence identifiers
            if not all('__' in str(idx) for idx in df['eid__sequence']):
                self.logger.logger.error("Invalid sequence identifier format")
                return False

            return True

        except Exception as e:
            self.logger.logger.error(f"Validation error: {str(e)}")
            return False

    def group_features(self, input_file: Path, output_file: Path) -> Optional[Path]:
        """
        Group features from multiple sequences into case-level features.

        Args:
            input_file (Path): Path to input CSV file with sequence-level features
            output_file (Path): Path where grouped features will be saved

        Returns:
            Optional[Path]: Path to the grouped features file if successful, None otherwise
        """
        self.logger.log_step_start("Feature grouping", input_file.stem)

        try:
            # Read input data
            df = pd.read_csv(input_file)

            if not self._validate_input_data(df):
                raise ValueError("Invalid input data format")

            # Extract case IDs
            df['case_id'] = df['eid__sequence'].apply(self._extract_case_id)

            # Get feature columns (exclude eid__sequence and case_id)
            feature_columns = [col for col in df.columns
                               if col not in ['eid__sequence', 'case_id']]

            # Initialize results structure
            results = []

            # Process each case
            for case in df['case_id'].unique():
                self.logger.log_step_start("Case processing", case)

                # Get sequences for this case
                case_df = df[df['case_id'] == case]

                # Initialize case dictionary
                case_dict = {'eid': case}

                # Process each sequence
                for idx, row in case_df.iterrows():
                    sequence_num = idx - case_df.index[0] + 1
                    prefix = f"seq{sequence_num}_"

                    # Add features for this sequence
                    for col in feature_columns:
                        case_dict[f"{prefix}{col}"] = row[col]

                results.append(case_dict)
                self.logger.log_step_complete("Case processing", case)

            # Create result DataFrame
            result_df = pd.DataFrame(results)

            # Organize columns
            base_cols = ['eid']
            num_sequences = len(df[df['case_id'] == df['case_id'].iloc[0]])

            for seq in range(1, num_sequences + 1):
                seq_cols = [f"seq{seq}_{col}" for col in feature_columns]
                base_cols.extend(seq_cols)

            # Reorder columns and save
            result_df = result_df[base_cols]
            output_file.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_file, index=False)

            # Save metadata
            metadata = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "cases_processed": len(result_df),
                "sequences_per_case": num_sequences,
                "feature_dimensions": {
                    "original": len(feature_columns),
                    "grouped": len(result_df.columns) - 1  # excluding 'eid'
                }
            }
            self.file_handler.save_processing_metadata(
                output_file.parent,
                metadata
            )

            self.logger.log_step_complete("Feature grouping", input_file.stem)
            return output_file

        except Exception as e:
            self.logger.log_error("Feature grouping", input_file.stem, e)
            return None

    def get_grouping_summary(self, output_file: Path) -> Dict:
        """
        Generate a summary of the grouping operation.

        Args:
            output_file (Path): Path to the grouped features file

        Returns:
            Dict: Summary statistics and information
        """
        try:
            df = pd.read_csv(output_file)
            return {
                "total_cases": len(df),
                "features_per_sequence": len([col for col in df.columns
                                              if col.startswith("seq1_")]),
                "total_features": len(df.columns) - 1,  # excluding 'eid'
                "file_path": str(output_file)
            }
        except Exception as e:
            self.logger.log_error("Summary generation", output_file.stem, e)
            return {}
