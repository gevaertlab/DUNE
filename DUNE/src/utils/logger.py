# src/utils/logger.py
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class PipelineLogger:
    """
    A centralized logging system for the neuroimaging pipeline.
    Handles both console and file logging with standardized formatting.
    """

    def __init__(self, output_dir: Path, name: str = "neuro_pipeline"):
        """
        Initialize the pipeline logger.

        Args:
            output_dir (Path): Directory where log files will be stored
            name (str, optional): Name identifier for the logger. Defaults to "neuro_pipeline"
        """
        self.output_dir = output_dir
        self.name = name
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Configure the logger with standardized formatting for both console and file output.

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        # Create formatters and handlers if they don't exist
        if not logger.handlers:
            # Logging format
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler
            self.output_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.output_dir / \
                f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def log_step_start(self, step_name: str, case_id: str) -> None:
        """
        Log the start of a pipeline step.

        Args:
            step_name (str): Name of the pipeline step
            case_id (str): Identifier of the case being processed
        """
        self.logger.info(f"Starting {step_name} for case {case_id}")

    def log_step_complete(self, step_name: str, case_id: str) -> None:
        """
        Log the successful completion of a pipeline step.

        Args:
            step_name (str): Name of the pipeline step
            case_id (str): Identifier of the case being processed
        """
        self.logger.info(f"Completed {step_name} for case {case_id}")

    def log_error(self, step_name: str, case_id: str, error: Exception) -> None:
        """
        Log an error that occurred during processing.

        Args:
            step_name (str): Name of the pipeline step where the error occurred
            case_id (str): Identifier of the case being processed
            error (Exception): The error that occurred
        """
        self.logger.error(
            f"Error in {step_name} for case {case_id}: {str(error)}")
