# src/utils/logger.py
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


# Codes ANSI pour les couleurs
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


class ColoredFormatter(logging.Formatter):
    """
    Formatter personnalisé qui ajoute des couleurs aux logs dans le terminal
    """

    def format(self, record):
        # Format de base
        log_format = "%(asctime)s | %(levelname)s | %(message)s"

        # Ajouter des couleurs selon le niveau
        if record.levelno >= logging.ERROR:
            prefix = Colors.RED
        elif record.levelno >= logging.WARNING:
            prefix = Colors.YELLOW
        elif record.levelno >= logging.INFO:
            prefix = Colors.BLUE
        else:
            prefix = Colors.RESET

        formatter = logging.Formatter(prefix + log_format + Colors.RESET,
                                      datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class PipelineLogger:
    """
    A centralized logging system for the neuroimaging pipeline.
    Handles both console and file logging with standardized formatting.
    """

    def __init__(self, output_dir: Path, name: str = "neuro_pipeline", enable_logs: bool = True):
        """
        Initialize the pipeline logger.

        Args:
            output_dir (Path): Directory where log files will be stored
            name (str, optional): Name identifier for the logger. Defaults to "neuro_pipeline"
            enable_logs (bool, optional): Whether to create log files. Defaults to True
        """
        self.output_dir = output_dir
        self.name = name
        self.enable_logs = enable_logs
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
            # Console handler with colors
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(ColoredFormatter())
            logger.addHandler(console_handler)

            # File handler (optional)
            if self.enable_logs:
                # Regular formatting for file (no colors)
                file_formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )

                self.output_dir.mkdir(parents=True, exist_ok=True)
                log_file = self.output_dir / \
                    f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

        return logger

    def log_step_start(self, step_name: str, case_id: str) -> None:
        """
        Log the start of a pipeline step.

        Args:
            step_name (str): Name of the pipeline step
            case_id (str): Identifier of the case being processed
        """
        message = f"Starting {step_name} for case {case_id}"
        self.logger.info(message)

    def log_step_complete(self, step_name: str, case_id: str) -> None:
        """
        Log the successful completion of a pipeline step.

        Args:
            step_name (str): Name of the pipeline step
            case_id (str): Identifier of the case being processed
        """
        message = f"Completed {step_name} for case {case_id}"
        # Utiliser des caractères spéciaux pour indiquer la réussite
        formatted_message = f"{Colors.GREEN}✓{Colors.RESET} {message}"
        self.logger.info(formatted_message)

    def log_error(self, step_name: str, case_id: str, error: Exception) -> None:
        """
        Log an error that occurred during processing.

        Args:
            step_name (str): Name of the pipeline step where the error occurred
            case_id (str): Identifier of the case being processed
            error (Exception): The error that occurred
        """
        message = f"Error in {step_name} for case {case_id}: {str(error)}"
        # Utiliser des caractères spéciaux pour indiquer l'erreur
        formatted_message = f"{Colors.RED}✗{Colors.RESET} {message}"
        self.logger.error(formatted_message)
