# DUNE: Deep feature extraction by UNet-based Neuroimaging-oriented autoEncoder

A versatile neuroimaging encoder that captures brain complexity across multiple diseases: cancer, dementia and schizophrenia.

## Overview

DUNE (Deep feature extraction by UNet-based Neuroimaging-oriented autoEncoder) is a neuroimaging-oriented deep learning model designed to extract deep features from multisequence brain MRIs, enabling their processing by basic machine learning algorithms. This project provides an end-to-end solution from DICOM conversion to low-dimensional feature extraction that captures clinically relevant patterns across several neurological conditions including cancer, dementia, and schizophrenia.

## Pipeline Architecture

The pipeline consists of the following sequential stages:

1. **DICOM to NIfTI Conversion**: Transforms medical DICOM images into the NIfTI format for analysis
2. **Preprocessing**: Prepares images through brain extraction, bias field correction, and spatial standardization
3. **Feature Extraction**: Uses a UNet-based autoencoder (without skip connections) to extract low-dimensional embeddings
4. **Feature Grouping**: Combines features from multiple MRI sequences into unified case-level representations
5. **Clinical Inference**: Utilizes extracted embeddings with simple machine learning models to predict clinical parameters

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.7+
- ANTs (Advanced Normalization Tools)
- DICOM2NIFTI
- TorchIO
- Nibabel
- Click

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
python -m src.main process /path/to/dicom/folder /path/to/output --config config.yaml
```

### Configuration

The pipeline can be customized through a YAML configuration file:

```yaml
paths:
  scripts:
    preprocessing: "scripts/preprocessing"
  templates: "data/templates"
  models: "data/models"

preprocessing:
  parameters:
    brain_extraction:
      threshold: 0.5
    bias_correction:
      iterations: 4

model:
  weights_file: "best_model.pt"
  input_size: [256, 256, 256]
```

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
│   │   ├── dicom.py     # DICOM conversion
│   │   ├── preprocessing.py # Image preprocessing
│   │   ├── feature_extraction.py # Feature extraction with BrainAE model
│   │   └── feature_grouping.py # Feature grouping
│   ├── utils            # Utility modules
│   │   ├── file_handling.py # File operations
│   │   └── logger.py    # Logging functionality
│   └── main.py          # CLI interface and pipeline orchestration
└── config.yaml          # Default configuration
```

## Datasets Used

DUNE was developed and validated using the following datasets:

- **UK Biobank (UKB)**: Healthy volunteers
- **UPENN**: University of Pennsylvania glioblastoma dataset
- **UCSF**: UCSF preoperative diffuse glioma MRI dataset
- **TCGA**: TCGA-LGG and TCGA-GBM datasets 
- **ADNI**: Alzheimer's Disease Neuroimaging Initiative dataset
- **SchizConnect**: COBRE and MCIC schizophrenia datasets

## The DUNE Model

DUNE's feature extraction is powered by a UNet-based autoencoder architecture without skip connections (U-AE):

- **Encoder**: Compresses the 3D brain volume through multiple convolutional blocks
- **Bottleneck**: Creates a compact latent representation of the brain structure (low-dimensional embeddings)
- **Decoder**: Reconstructs the original image from the latent representation

This unsupervised approach learns to capture both obvious and subtle imaging features, creating a numerical "fingerprint" of each scan that preserves important structural information while dramatically reducing dimensionality. By removing skip connections from the traditional UNet architecture, DUNE forces all information through the bottleneck, producing more informative embeddings despite lower reconstruction quality.

The model was trained on 3,814 MRI scans including both healthy volunteers from UK Biobank and patients with gliomas to ensure it can effectively extract features from both normal and pathological brain structures.

## Dependencies

- `nibabel`: For reading and writing NIfTI files
- `pydicom`: For DICOM file operations
- `torch`: Deep learning framework
- `torchio`: Medical image preprocessing and augmentation
- `dicom2nifti`: DICOM to NIfTI conversion
- `pandas`: Data manipulation
- `ANTs`: Advanced image registration and normalization
- `scikit-learn`: For machine learning models using the extracted features


## Citation

If you use DUNE in your research, please consider citing:

```
@article{barba2025dune,
  author = {Barba, Thomas and Bagley, Bryce A. and Steyaert, Sandra and Carrillo-Perez, Francisco and Sadée, Christoph and Iv, Michael and Gevaert, Olivier},
  title = {DUNE: a versatile neuroimaging encoder captures brain complexity across three major diseases: cancer, dementia and schizophrenia},
  year = {2025},
  url = {https://www.medrxiv.org/content/10.1101/2025.02.24.25322787v1.full}
}
```
