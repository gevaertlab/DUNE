# DUNE: Deep feature extraction by UNet-based Neuroimaging-oriented autoEncoder

A versatile neuroimaging encoder that captures brain complexity across multiple diseases: cancer, dementia and schizophrenia.

## Overview

DUNE (Deep feature extraction by UNet-based Neuroimaging-oriented autoEncoder) is a neuroimaging-oriented deep learning model designed to extract deep features from multisequence brain MRIs, enabling their processing by basic machine learning algorithms. This project provides an end-to-end solution from DICOM conversion to low-dimensional feature extraction that captures clinically relevant patterns across several neurological conditions including cancer, dementia, and schizophrenia.

## Pipeline Architecture

The pipeline consists of the following sequential stages:

1. **DICOM to NIfTI Conversion**: Transforms medical DICOM images into the NIfTI format for analysis (skipped if NIfTI files are provided directly)
2. **Preprocessing**: Prepares images through brain extraction (optional if already done), bias field correction, and spatial standardization
3. **Feature Extraction**: Uses a UNet-based autoencoder (without skip connections) to extract low-dimensional embeddings

Each stage can be run independently based on your specific needs.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.7+
- ANTs (Advanced Normalization Tools)
- DICOM2NIFTI
- TorchIO
- Nibabel
- Click

### ANTs Installation

DUNE requires ANTs (Advanced Normalization Tools) for brain image preprocessing. ANTs is not installable via pip and requires separate installation:

- Installation instructions are available on the [ANTs GitHub repository](https://github.com/ANTsX/ANTs)
- Ensure the ANTs binaries are in your PATH before running DUNE

### Setup

```bash
# Clone the repository
git clone https://github.com/gevaertlab/DUNE.git
cd DUNE

# Install dependencies
pip install -r requirements.txt

# Ensure ANTs binaries are in your PATH
# For example:
export ANTSPATH=/path/to/ANTs/bin/
export PATH=${ANTSPATH}:$PATH
```

## Usage

### Initialize Workspace

```bash
python -m src.main init /path/to/workspace
```

### Process a Case

```bash
# Process a DICOM folder
python -m src.main process /path/to/dicom/folder /path/to/output

# Process a NIfTI file with brain extraction
python -m src.main process /path/to/image.nii.gz /path/to/output

# Process a NIfTI file skipping brain extraction (if already performed)
python -m src.main process /path/to/image.nii.gz /path/to/output --skip-brain-extraction

# Keep preprocessed files in the output
python -m src.main process /path/to/image.nii.gz /path/to/output --keep-preprocessed

# Process without creating log files
python -m src.main process /path/to/image.nii.gz /path/to/output --no-logs

# Process a folder containing multiple NIfTI files
python -m src.main process /path/to/nifti/folder /path/to/output

# Only perform feature extraction (skip DICOM conversion and preprocessing)
python -m src.main process /path/to/preprocessed.nii.gz /path/to/output --features-only
```

### Output Structure

The pipeline produces a simplified output structure:

1. **For a single file input (e.g., my_image.nii.gz)**:
   ```
   output_dir/
   ├── my_image_features.csv              # Extracted features
   ├── my_image_preprocessed.nii.gz       # (Optional) Preprocessed image if --keep-preprocessed is used
   └── logs/                              # Processing logs (unless --no-logs is used)
   ```

2. **For a directory input (e.g., case_dir/)**:
   ```
   output_dir/
   ├── case_dir/
   │   ├── features/
   │   │   ├── sequence1_features.csv     # Features for sequence 1
   │   │   └── sequence2_features.csv     # Features for sequence 2
   │   └── preprocessed/                  # (Optional) Only if --keep-preprocessed is used
   │       ├── sequence1_preprocessed.nii.gz
   │       └── sequence2_preprocessed.nii.gz
   └── logs/                              # Processing logs (unless --no-logs is used)
   ```

### Feature Files Structure

Each generated feature file is a CSV containing 3,072 features extracted from the bottleneck layer of the DUNE autoencoder. The files are structured as follows:

- The first column contains the sequence identifier
- The remaining columns (3,072) contain the extracted features

Example of a feature file structure:

| sequence_id        | feature_0 | feature_1 | feature_2 | ... | feature_3071 |
| ------------------ | --------- | --------- | --------- | --- | ------------ |
| patient001-T1_POST | 0.0821    | -0.1427   | 0.2984    | ... | 0.0193       |

These low-dimensional embeddings capture essential brain structure information that can be used for downstream machine learning tasks like classification, regression, or clustering.

### Configuration

The pipeline can be customized through a YAML configuration file:

```yaml
# Paths to resources
paths:
  scripts:
    preprocessing: "scripts/preprocessing"
  templates: "data/templates"
  models: "data/models"

# Preprocessing options
preprocessing:
  parameters:
    brain_extraction:
      threshold: 0.5
    bias_correction:
      iterations: 4
  skip_brain_extraction: false
  keep_preprocessed: false

# Model options
model:
  weights_file: "U-AE.pt"
  input_size: [256, 256, 256]

# Output options
output:
  enable_logs: true

# Pipeline control options
pipeline:
  features_only: false  # If true, only performs feature extraction
```

## Alternative Model Architectures

This repository also contains alternative autoencoder architectures that were evaluated in our paper:

1. **U-AE** (default): UNet without skip connections - produces the most informative embeddings
2. **UNET**: UNet with skip connections - best for image reconstruction
3. **U-VAE**: Variational UNet without skip connections
4. **VAE**: Fully connected variational autoencoder

To use an alternative architecture, specify it in your configuration file:

```yaml
model:
  architecture: "UNET"  # Options: "U-AE", "UNET", "U-VAE", "VAE"
  weights_path: "data/models/UNET.pt"
```
Or when running the feature extraction directly:

```bash
python src/pipeline/feature_extraction.py input.nii.gz output.csv --architecture UNET
```

## Command Line Options

| Option                          | Description                                       |
| ------------------------------- | ------------------------------------------------- |
| `--config`, `-c`                | Path to configuration file                        |
| `--verbose`, `-v`               | Enable verbose output                             |
| `--skip-brain-extraction`, `-s` | Skip brain extraction step                        |
| `--keep-preprocessed`, `-p`     | Keep preprocessed NIfTI files in output           |
| `--no-logs`                     | Disable writing log files                         |
| `--features-only`, `-f`         | Only perform feature extraction                   |

## Pipeline Flexibility

DUNE offers flexibility to handle different workflow scenarios:

1. **Complete Pipeline**: Convert DICOM to NIfTI, preprocess images, and extract features
2. **Partial Processing**: Start with NIfTI files and apply preprocessing and feature extraction
3. **Minimal Preprocessing**: Skip brain extraction if your images are already skull-stripped
4. **Feature Extraction Only**: If you already have fully preprocessed NIfTI files, you can extract features directly

This adaptability allows DUNE to fit into existing neuroimaging workflows and accommodate different data preparation stages.

## Project Structure

```
DUNE
├── data
│   ├── models           # Pre-trained model weights
│   └── templates        # Registration templates (MNI, OASIS)
├── scripts
│   └── preprocessing    # ANTs-based preprocessing scripts
├── src
│   ├── pipeline         # Core pipeline components
│   │   ├── dicom.py     # DICOM conversion and NIfTI handling
│   │   ├── preprocessing.py # Image preprocessing
│   │   └── feature_extraction.py # Feature extraction with BrainAE model
│   ├── utils            # Utility modules
│   │   ├── file_handling.py # File operations
│   │   └── logger.py    # Logging functionality
│   └── main.py          # CLI interface and pipeline orchestration
└── config.yaml          # Default configuration
```

## The DUNE Model

DUNE's feature extraction is powered by a UNet-based autoencoder architecture without skip connections (U-AE):

- **Encoder**: Compresses the 3D brain volume through multiple convolutional blocks
- **Bottleneck**: Creates a compact latent representation of the brain structure (low-dimensional embeddings)
- **Decoder**: Reconstructs the original image from the latent representation

This unsupervised approach learns to capture both obvious and subtle imaging features, creating a numerical "fingerprint" of each scan that preserves important structural information while dramatically reducing dimensionality.

## Citation

If you use DUNE in your research, please consider citing:

```
@article{barba2025dune,
  author = {Barba T, Bagley BA, Steyaert S, Carrillo-Perez F, Sadée C, Iv M, Gevaert O},
  title = {DUNE: a versatile neuroimaging encoder captures brain complexity across three major diseases: cancer, dementia and schizophrenia},
  year = {2025},
  url = {https://www.medrxiv.org/content/10.1101/2025.02.24.25322787v1.full}
}
```

## License
Copyright 2025 - Prof Olivier Gevaert

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.