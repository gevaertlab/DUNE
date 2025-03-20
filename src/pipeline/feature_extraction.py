# src/pipeline/feature_extraction.py
from pathlib import Path
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import pandas as pd
import torchio as tio
from src.utils.logger import PipelineLogger
from src.utils.file_handling import FileHandler


class BrainAE(nn.Module):
    def __init__(self, in_channels=1, init_features=4, num_blocks=6):
        super(BrainAE, self).__init__()

        self.num_blocks = num_blocks
        self.skip_connections = False

        # ENCODER BLOCKS
        feature_list = [init_features*(2**n) for n in range(num_blocks)]
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2, 0)
        for n in feature_list:
            enc_block = self._block(in_channels, n)
            self.encoder.append(enc_block)
            in_channels = n

        # BOTTLENECK
        bn_features = 2*feature_list[-1]
        self.bottleneck = self._block(feature_list[-1], bn_features)

        # DECODER BLOCKS
        feature_list = feature_list[::-1]
        self.decoder = nn.ModuleList()
        self.transposers = nn.ModuleList()

        for n in feature_list:
            upconv = nn.ConvTranspose3d(bn_features, n, 2, 2, 0)
            dec_block = self._block(n, n)
            self.transposers.append(upconv)
            self.decoder.append(dec_block)
            bn_features = n

        # FINAL CONVOLUTION
        self.last_conv = nn.Conv3d(feature_list[-1], 1, 1, 1, 0)

    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, 3, 1, 1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, 3, 1, 1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )

    def forward(self, x):
        # ENCODING
        encodings = []
        for k in range(self.num_blocks):
            enc = self.encoder[k](x)
            x = self.pool(enc)
            encodings.append(enc)

        # BOTTLENECK
        bottleneck = self.bottleneck(x)

        # DECODING
        encodings.reverse()
        dec = bottleneck
        for k in range(self.num_blocks):
            dec = self.transposers[k](dec, output_size=encodings[k].shape)
            dec = self.decoder[k](dec)

        dec = self.last_conv(dec)
        return torch.sigmoid(dec), bottleneck, None


def load_and_preprocess_nifti(file_path):
    """Load and preprocess a NIfTI file for feature extraction."""
    img = nib.load(file_path)
    rescaler = tio.RescaleIntensity(out_min_max=(0, 1))
    img_data = rescaler(img).get_fdata()
    img_data = np.array(img_data, dtype=np.float32)

    # Add channel and batch dimensions, reorder for PyTorch
    img_data = np.expand_dims(img_data, axis=0)  # channel dim
    img_data = img_data.transpose((0, 3, 2, 1))  # reorder for PyTorch
    img_data = np.expand_dims(img_data, axis=0)  # batch dim

    return torch.tensor(img_data)


def extract_features(model, input_path, output_path):
    """Extract features from NIfTI files and save to CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    input_path = Path(input_path)
    features_dict = {}

    # Handle both single file and directory inputs
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("*.nii.gz"))

    print(f"Processing {len(files)} files...")

    with torch.no_grad():
        for file in files:
            try:
                print(f"Processing {file.name}...")
                img_tensor = load_and_preprocess_nifti(file)
                img_tensor = img_tensor.to(device)

                _, bottleneck, _ = model(img_tensor)
                features = bottleneck.cpu().numpy().reshape(-1)

                features_dict[file.stem] = features

            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")

    # Create and save features DataFrame
    features_df = pd.DataFrame.from_dict(features_dict, orient='index')
    features_df.to_csv(output_path, index_label="eid__sequence")
    print(f"Features saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from brain MRI images')
    parser.add_argument('input', type=str,
                        help='Input NIfTI file or directory')
    parser.add_argument('output', type=str, help='Output CSV file path')
    parser.add_argument('--model', type=str,
                        default='best_model.pt', help='Path to model weights')
    args = parser.parse_args()

    # Initialize model and load weights
    model = BrainAE()
    try:
        state_dict = torch.load(args.model, map_location='cpu')
        if 'module' in next(iter(state_dict.keys())):
            # Remove 'module.' prefix if model was saved with DataParallel
            state_dict = {k.replace('module.', ''): v for k,
                          v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Loaded model from {args.model}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Extract features
    extract_features(model, args.input, args.output)


if __name__ == "__main__":
    main()


class FeatureExtractor:
    """
    Handles the extraction of features from preprocessed brain MRI images
    using a pre-trained autoencoder model.
    """

    def __init__(self, logger: PipelineLogger, file_handler: FileHandler, config: Dict):
        """
        Initialize the feature extractor with model and configuration.

        Args:
            logger (PipelineLogger): Logger instance for tracking operations
            file_handler (FileHandler): File handler instance for managing files
            config (Dict): Configuration containing model paths and parameters

        Raises:
            FileNotFoundError: If model weights file is not found
            RuntimeError: If model loading fails
        """
        self.logger = logger
        self.file_handler = file_handler
        self.config = config

        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize and load model
        self.model = self._initialize_model()

    def _initialize_model(self) -> nn.Module:
        """
        Initialize and load the pre-trained model.

        Returns:
            nn.Module: Loaded model

        Raises:
            FileNotFoundError: If weights file doesn't exist
            RuntimeError: If model loading fails
        """
        model_path = Path(self.config['model']['weights_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        try:
            model = BrainAE()
            state_dict = torch.load(model_path, map_location=self.device)

            # Handle DataParallel wrapped state dict
            if 'module.' in next(iter(state_dict.keys())):
                state_dict = {k.replace('module.', '')                              : v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _preprocess_image(self, file_path: Path) -> Optional[torch.Tensor]:
        """
        Load and preprocess a NIfTI image for feature extraction.

        Args:
            file_path (Path): Path to the NIfTI file

        Returns:
            Optional[torch.Tensor]: Preprocessed image tensor or None if failed
        """
        try:
            # Load and normalize image
            img = nib.load(str(file_path))
            rescaler = tio.RescaleIntensity(out_min_max=(0, 1))
            img_data = rescaler(img).get_fdata()
            img_data = np.array(img_data, dtype=np.float32)

            # Add dimensions and reorder for PyTorch
            img_data = np.expand_dims(img_data, axis=0)  # channel dim
            img_data = img_data.transpose((0, 3, 2, 1))  # reorder
            img_data = np.expand_dims(img_data, axis=0)  # batch dim

            return torch.tensor(img_data).to(self.device)

        except Exception as e:
            self.logger.log_error("Image preprocessing", file_path.stem, e)
            return None

    def extract_features(self, input_files: Dict[str, Path], output_file: Path) -> Optional[Path]:
        """
        Extract features from a set of preprocessed NIfTI files.

        Args:
            input_files (Dict[str, Path]): Dictionary mapping sequence names to file paths
            output_file (Path): Path where features will be saved

        Returns:
            Optional[Path]: Path to the features CSV file if successful, None otherwise
        """
        self.logger.log_step_start("Feature extraction", "multiple sequences")

        try:
            features_dict = {}

            with torch.no_grad():
                for seq_name, file_path in input_files.items():
                    self.logger.log_step_start("Feature extraction", seq_name)

                    # Preprocess image
                    img_tensor = self._preprocess_image(file_path)
                    if img_tensor is None:
                        continue

                    # Extract features
                    _, bottleneck, _ = self.model(img_tensor)
                    features = bottleneck.cpu().numpy().reshape(-1)

                    # Store features
                    features_dict[file_path.stem] = features

                    self.logger.log_step_complete(
                        "Feature extraction", seq_name)

            if not features_dict:
                raise ValueError("No features were successfully extracted")

            # Create and save features DataFrame
            features_df = pd.DataFrame.from_dict(features_dict, orient='index')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(output_file, index_label="eid__sequence")

            # Save metadata
            metadata = {
                "input_files": [str(p) for p in input_files.values()],
                "output_file": str(output_file),
                "feature_dimensions": features_df.shape,
                "model_config": {
                    "architecture": "BrainAE",
                    "device": str(self.device)
                }
            }
            self.file_handler.save_processing_metadata(
                output_file.parent,
                metadata
            )

            self.logger.log_step_complete(
                "Feature extraction", "all sequences")
            return output_file

        except Exception as e:
            self.logger.log_error("Feature extraction", "all sequences", e)
            return None
