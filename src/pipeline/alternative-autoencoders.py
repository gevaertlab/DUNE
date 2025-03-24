"""
Alternative Autoencoder Architectures for Brain MRI Feature Extraction

This module contains implementations of alternative autoencoder architectures
that were evaluated in the DUNE paper (Deep feature extraction by UNet-based
Neuroimaging-oriented autoEncoder).

Three architectures are implemented:
1. UNET - UNet autoencoder with skip connections
2. U_VAE - Variational UNet autoencoder without skip connections
3. VAE - Fully connected variational autoencoder

These models are provided for reference and comparison with the primary
model (U-AE, implemented as BrainAE in feature_extraction.py) used in
the production pipeline.

Usage:
    These models can be used as drop-in replacements for the main BrainAE model.
    However, they may require additional parameters or different handling of
    the loss function during training.

Reference:
    See the DUNE paper for a comprehensive comparison of these architectures
    and their performance on various clinical prediction tasks.
"""

import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import torchio as tio
from typing import List, Any, Tuple, Optional
from abc import abstractmethod


class SelfAttention(nn.Module):
    """Self-attention module for 3D feature maps"""
    
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()

        query = self.query_conv(x).view(batch_size, -1, depth * height * width)
        key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        value = self.value_conv(x).view(batch_size, -1, depth * height * width)

        attention_scores = torch.matmul(query, key.permute(0, 2, 1))
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.view(batch_size, channels, depth, height, width)

        out = self.gamma * attention_output + x
        return out


class UNET(nn.Module):
    """
    UNet 3D Autoencoder with skip connections.
    This architecture produces the best image reconstruction but
    may generate less informative embeddings compared to the U-AE model.
    """
    
    def __init__(self, in_channels=1, init_features=4, num_blocks=6, dropout=0.1, attention=False):
        super(UNET, self).__init__()

        self.num_mod = in_channels
        self.num_blocks = num_blocks
        self._type = "UNET"
        self.skip_connections = True

        # Encoder blocks
        feature_list = [init_features*(2**n) for n in range(num_blocks)]
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2, 0)
        for n in feature_list:
            enc_block = self._block(in_channels, n, attention, dropout)
            self.encoder.append(enc_block)
            in_channels = n

        # Bottleneck
        bn_features = 2*feature_list[-1]
        self.bottleneck = self._block(feature_list[-1], bn_features, attention, dropout)

        # Decoder blocks
        feature_list = feature_list[::-1]
        self.decoder = nn.ModuleList()
        self.transposers = nn.ModuleList()
        
        # With skip connections, the decoder input has double the channels 
        for n in feature_list:
            upconv = nn.ConvTranspose3d(bn_features, n, 2, 2, 0)
            dec_block = self._block(n * 2, n, attention, dropout)  # x2 for skip connections
            self.transposers.append(upconv)
            self.decoder.append(dec_block)
            bn_features = n

        # Final convolution
        self.last_conv = nn.Conv3d(feature_list[-1], self.num_mod, 1, 1, 0)

    def forward(self, x):
        # Encoding
        encodings = []
        for k in range(self.num_blocks):
            enc = self.encoder[k](x)
            x = self.pool(enc)
            encodings.append(enc)

        # Bottleneck
        bottleneck = self.bottleneck(x)

        # Decoding with skip connections
        encodings.reverse()
        dec = bottleneck
        for k in range(self.num_blocks):
            dec = self.transposers[k](dec, output_size=encodings[k].shape)
            dec = torch.cat((dec, encodings[k]), dim=1)  # Skip connection
            dec = self.decoder[k](dec)

        dec = self.last_conv(dec)
        return torch.sigmoid(dec), bottleneck, None
    
    def process_nifti(self, nifti_path):
        """Process a NIfTI file and extract features"""
        img = nib.load(nifti_path)
        rescaler = tio.RescaleIntensity(out_min_max=(0, 1))
        img_data = rescaler(img).get_fdata()
        img_data = np.array(img_data, dtype=np.float32)

        # Add channel and batch dimensions, reorder for PyTorch
        img_data = np.expand_dims(img_data, axis=0)  # channel dim
        img_data = img_data.transpose((0, 3, 2, 1))  # reorder for PyTorch
        img_data = np.expand_dims(img_data, axis=0)  # batch dim

        img_tensor = torch.tensor(img_data)
        reconstructed, embedding, _ = self.forward(img_tensor)

        return reconstructed, embedding

    @staticmethod
    def _block(in_channels, features, attn, dropout):
        block = nn.Sequential(
            nn.Conv3d(in_channels, features, 3, 1, 1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, 3, 1, 1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )
        
        if attn:
            attention = SelfAttention(features)
            block = nn.Sequential(block, attention)
            
        return block


class U_VAE(nn.Module):
    """
    Variational UNet Autoencoder without skip connections.
    This architecture adds a variational component to the U-AE model, 
    which may better regularize the latent space.
    """
    
    def __init__(self, in_channels=1, init_features=4, num_blocks=6, 
                 input_dim=(182, 218, 160), hidden_size=2048, dropout=0.1, attention=False):
        super(U_VAE, self).__init__()

        self.num_mod = in_channels
        self.num_blocks = num_blocks
        self._type = "U_VAE"
        self.hidden_size = hidden_size
        self.attention = attention

        # Encoder blocks
        feature_list = [init_features*(2**n) for n in range(num_blocks)]
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2, 0)
        for n in feature_list:
            enc_block = self._block(in_channels, n, attn=self.attention, dropout=dropout)
            self.encoder.append(enc_block)
            in_channels = n

        # Calculate the shape after encoding for proper flattening
        enc_shape, size = self._get_encoded_shape(input_dim, feature_list)
        
        # Variational components
        self.mu = nn.Linear(size, self.hidden_size)
        self.sigma = nn.Linear(size, self.hidden_size)
        self.decoder_input = nn.Linear(self.hidden_size, size)
        self.enc_shape = enc_shape

        # Decoder blocks
        feature_list = feature_list[::-1]
        self.decoder = nn.ModuleList()
        self.transposers = nn.ModuleList()
        for n in feature_list:
            upconv = nn.ConvTranspose3d(enc_shape[0], n, 2, 2, 0)
            dec_block = self._block(n, n, attn=self.attention, dropout=dropout)
            self.transposers.append(upconv)
            self.decoder.append(dec_block)
            enc_shape[0] = n

        # Final convolution
        self.last_conv = nn.Conv3d(feature_list[-1], self.num_mod, 1, 1, 0)

    def reparameterize(self, mu, sigma):
        """Reparameterization trick for variational autoencoder"""
        eps = torch.randn_like(sigma)
        return eps * sigma + mu

    def encode(self, x):
        """Encode input to latent representation"""
        encodings = []
        for k in range(self.num_blocks):
            enc = self.encoder[k](x)
            encodings.append(enc)
            x = self.pool(enc)
        
        # Flatten and get variational parameters
        x_flat = torch.flatten(x, start_dim=1)
        mu = self.mu(x_flat)
        sigma = self.sigma(x_flat)
        sigma = torch.exp(0.5 * sigma)  # Convert to standard deviation
        
        return x, encodings, mu, sigma

    def decode(self, z, encodings):
        """Decode latent representation back to image space"""
        # Reshape from latent vector to 3D feature map
        x = self.decoder_input(z)
        x = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2], self.enc_shape[3])
        
        # Decode through transpose convolutions
        for k in range(self.num_blocks):
            x = self.transposers[k](x, output_size=encodings[k].shape)
            x = self.decoder[k](x)
            
        return self.last_conv(x)

    def forward(self, x):
        # Encoding and reparameterization
        enc, encodings, mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        
        # Decoding
        encodings.reverse()
        output = self.decode(z, encodings)
        
        return torch.sigmoid(output), z, (mu, sigma)

    def process_nifti(self, nifti_path):
        """Process a NIfTI file and extract features"""
        img = nib.load(nifti_path)
        rescaler = tio.RescaleIntensity(out_min_max=(0, 1))
        img_data = rescaler(img).get_fdata()
        img_data = np.array(img_data, dtype=np.float32)

        # Add channel and batch dimensions, reorder for PyTorch
        img_data = np.expand_dims(img_data, axis=0)  # channel dim
        img_data = img_data.transpose((0, 3, 2, 1))  # reorder for PyTorch
        img_data = np.expand_dims(img_data, axis=0)  # batch dim

        img_tensor = torch.tensor(img_data)
        reconstructed, embedding, _ = self.forward(img_tensor)

        return reconstructed, embedding

    @staticmethod
    def _block(in_channels, features, attn, dropout):
        block = nn.Sequential(
            nn.Conv3d(in_channels, features, 3, 1, 1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, 3, 1, 1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )
        
        if attn:
            attention = SelfAttention(features)
            block = nn.Sequential(block, attention)
            
        return block

    @staticmethod
    def _get_encoded_shape(input_dim, feature_list):
        """Calculate the shape of the encoded representation"""
        def block_shaping(dim):
            k, s, p = 3, 1, 1
            dim = int(((dim + 2*p - (k-1) - 1) / s) + 1)  # conv1
            dim = int(((dim + 2*p - (k-1) - 1) / s) + 1)  # conv2
            k, s, p = 2, 2, 0
            dim = int(((dim + 2*p - (k-1) - 1) / s) + 1)  # maxpool
            return dim

        W, H, D = input_dim
        for _ in feature_list:
            D = block_shaping(D)
            H = block_shaping(H)
            W = block_shaping(W)

        C = feature_list[-1]
        return [C, D, H, W], int(C*D*H*W)


class VAE(nn.Module):
    """
    Fully connected Variational Autoencoder.
    This architecture uses a more traditional VAE approach with
    convolutional encoding and decoding paths.
    """
    
    def __init__(self, in_channels=1, latent_dim=2048, input_dim=(182, 218, 160), 
                 hidden_dims=None, dropout=0.1):
        super(VAE, self).__init__()

        self._type = "VAE"
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        if hidden_dims is None:
            hidden_dims = [4, 16, 32, 64, 128]
        
        self.hidden_dims = hidden_dims.copy()
        
        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU(),
                    nn.Dropout3d(dropout)
                )
            )
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*modules)
        
        # Calculate encoded shape for proper flattening
        self.encoded_shapes, self.flatten_shape = self._get_encoded_shape(input_dim, hidden_dims)
        
        # Variational components
        self.fc_mu = nn.Linear(self.flatten_shape, latent_dim)
        self.fc_var = nn.Linear(self.flatten_shape, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_shape)
        
        # Build Decoder
        hidden_dims.reverse()
        self.decoder_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.decoder_layers.append(
                nn.ConvTranspose3d(
                    hidden_dims[i], hidden_dims[i + 1],
                    kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            self.norm_layers.append(
                nn.Sequential(
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                    nn.Dropout3d(dropout)
                )
            )
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(
                hidden_dims[-1], hidden_dims[-1],
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm3d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dims[-1], out_channels=self.in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, input_tensor):
        """Encode the input to get mean and log variance of the latent distribution"""
        result = self.encoder(input_tensor)
        result = torch.flatten(result, start_dim=1)
        
        # Return mean and log variance
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        """Decode the latent representation back to image space"""
        result = self.decoder_input(z)
        
        # Reshape to 3D feature map
        rev_shapes = self.encoded_shapes.copy()
        rev_shapes.reverse()
        result = result.view(-1, *rev_shapes[0])
        
        # Decode through transpose convolutions
        for i, decoder_layer in enumerate(self.decoder_layers):
            result = decoder_layer(result)
            result = self.norm_layers[i](result)
            
        result = self.final_layer(result)
        return result

    def forward(self, input_tensor):
        mu, log_var = self.encode(input_tensor)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        
        # Convert log_var to standard deviation for consistency with other models
        sigma = torch.exp(0.5 * log_var)
        
        return reconstructed, z, (mu, sigma)

    def process_nifti(self, nifti_path):
        """Process a NIfTI file and extract features"""
        img = nib.load(nifti_path)
        rescaler = tio.RescaleIntensity(out_min_max=(0, 1))
        img_data = rescaler(img).get_fdata()
        img_data = np.array(img_data, dtype=np.float32)

        # Add channel and batch dimensions, reorder for PyTorch
        img_data = np.expand_dims(img_data, axis=0)  # channel dim
        img_data = img_data.transpose((0, 3, 2, 1))  # reorder for PyTorch
        img_data = np.expand_dims(img_data, axis=0)  # batch dim

        img_tensor = torch.tensor(img_data)
        reconstructed, embedding, _ = self.forward(img_tensor)

        return reconstructed, embedding

    @staticmethod
    def _get_encoded_shape(input_dim, feature_list):
        """Calculate the shape after encoding for proper flattening"""
        def block_shaping(dim):
            k, s, p = 3, 2, 1
            dim = int(((dim + 2*p - (k-1) - 1) / s) + 1)
            return dim

        W, H, D = input_dim
        shapes = []
        for C in feature_list:
            D = block_shaping(D)
            H = block_shaping(H)
            W = block_shaping(W)
            shapes.append([C, D, H, W])
        
        final_size = int(feature_list[-1] * D * H * W)

        return shapes, final_size


# Function to create model based on architecture name
def create_autoencoder(architecture, **kwargs):
    """
    Factory function to create an autoencoder model based on architecture name
    
    Args:
        architecture: String specifying the architecture ("UNET", "U_VAE", "VAE", or "U-AE")
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        An initialized autoencoder model
    """
    if architecture.upper() == "UNET":
        return UNET(**kwargs)
    elif architecture.upper() == "U_VAE":
        return U_VAE(**kwargs)
    elif architecture.upper() == "VAE":
        return VAE(**kwargs)
    elif architecture.upper() == "U-AE":
        # Import the primary model
        from feature_extraction import BrainAE
        return BrainAE(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. " 
                         "Choose from 'UNET', 'U_VAE', 'VAE', or 'U-AE'.")
