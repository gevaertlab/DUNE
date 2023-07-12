import torch
import torch.nn as nn
from models.attention import SelfAttention
import nibabel as nib
import torchio as tio


class BaseAE(nn.Module):
    def __init__(self, in_channels, init_features, num_blocks, type_ae, dropout, attention=False):
        super(BaseAE, self).__init__()

        self.num_mod = in_channels
        self.num_blocks = num_blocks
        self._type = type_ae
        self.skip_connections = True if type_ae.lower() == "unet" else False

        # ENCODER BLOCKS
        feature_list = [init_features*(2**n) for n in range(num_blocks)]
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2, 0)
        for n in feature_list:
            enc_block = BaseAE._block(in_channels, n, attention, dropout)
            self.encoder.append(enc_block)
            in_channels = n

        # BOTTLENECK EXTRACTION
        bn_features = 2*feature_list[-1]
        self.bottleneck = BaseAE._block(
            feature_list[-1], bn_features, attention, dropout)

        # DECODER BLOCKS
        feature_list = feature_list[::-1]
        self.decoder = nn.ModuleList()
        self.transposers = nn.ModuleList()
        enlarger = 2 if self.skip_connections else 1
        for n in feature_list:
            upconv = nn.ConvTranspose3d(bn_features, n, 2, 2, 0)
            dec_block = BaseAE._block(n * enlarger, n, attention, dropout)
            self.transposers.append(upconv)
            self.decoder.append(dec_block)
            bn_features = n

        # FINAL CONVOLUTION
        self.last_conv = nn.Conv3d(feature_list[-1], self.num_mod, 1, 1, 0)

    def forward(self, x):
        # ENCODING
        encodings = []
        for k in range(self.num_blocks):
            enc = self.encoder[k](x)
            x = self.pool(enc)
            encodings.append(enc)

        # REPRESENTATION EXTRACTION
        bottleneck = self.bottleneck(x)

        # DECODING
        encodings.reverse()
        dec = bottleneck
        for k in range(self.num_blocks):
            dec = self.transposers[k](dec, output_size=encodings[k].shape)
            if self.skip_connections:
                dec = torch.cat((dec, encodings[k]), dim=1)
            dec = self.decoder[k](dec)

        dec = self.last_conv(dec)

        return torch.sigmoid(dec), bottleneck, None

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
            attention = SelfAttention(features)  # Add attention mechanism
            block = nn.Sequential(block, attention)

        return block


class GuidedAE(nn.Module):
    def __init__(self, in_channels, init_features, num_blocks, dropout, template, device, attention=False):
        super(GuidedAE, self).__init__()

        self.num_mod = in_channels
        self.num_blocks = num_blocks
        self._type = "Gu-AE"

        # ENCODER BLOCKS
        feature_list = [init_features*(2**n) for n in range(num_blocks)]
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2, 0)
        for n in feature_list:
            enc_block = BaseAE._block(in_channels, n, attention, dropout)
            self.encoder.append(enc_block)
            in_channels = n

        # BOTTLENECK EXTRACTION
        bn_features = 2*feature_list[-1]
        self.bottleneck = BaseAE._block(
            feature_list[-1], bn_features, attention, dropout)

        # DECODER BLOCKS
        feature_list = feature_list[::-1]
        self.decoder = nn.ModuleList()
        self.transposers = nn.ModuleList()
        for n in feature_list:
            upconv = nn.ConvTranspose3d(bn_features, n, 2, 2, 0)
            dec_block = BaseAE._block(n * 2, n, attention, dropout)
            self.transposers.append(upconv)
            self.decoder.append(dec_block)
            bn_features = n

        self.template = self.import_template(template).to(device)
        self.last_conv = nn.Conv3d(feature_list[-1], self.num_mod, 1, 1, 0)

    def import_template(self, template):

        rescale = tio.RescaleIntensity(out_min_max=(0, 1))

        template = nib.load(template)
        template = rescale(template).get_fdata()
        template = torch.Tensor(template.transpose((2, 1, 0)))
        template = template.unsqueeze(0)

        return template

    def encode(self, x, template=False, bs=1):

        encodings = []
        if template:
            x = self.template.repeat(repeats=(bs, 1, 1, 1, 1))

        for k in range(self.num_blocks):
            enc = self.encoder[k](x)

            encodings.append(enc)
            x = self.pool(enc)

        return encodings if template else x

    def decode(self, x, encodings):

        encodings.reverse()
        for k in range(self.num_blocks):
            x = self.transposers[k](x, output_size=encodings[k].shape)
            x = torch.cat((x, encodings[k]), dim=1)
            x = self.decoder[k](x)

        return x

    def forward(self, x):

        encodings = self.encode(self.template, template=True, bs=x.size(0))
        x = self.encode(x)

        bottleneck = self.bottleneck(x)
        x = self.decode(bottleneck, encodings)
        x = self.last_conv(x)

        return torch.sigmoid(x), bottleneck, None



# class NaseAE(nn.Module):
#     def __init__(self, in_channels, init_features, num_blocks, type_ae, dropout, attention=False, hidden_size="auto"):
#         super(BaseAE, self).__init__()

#         self.num_mod = in_channels
#         self.num_blocks = num_blocks
#         self._type = type_ae
#         self.skip_connections = True if type_ae.lower() == "unet" else False
#         self.attention = attention
#         self.hidden_size = hidden_size

#         # ENCODER BLOCKS
#         feature_list = [init_features*(2**n) for n in range(num_blocks)]
#         self.encoder = nn.ModuleList()
#         self.pool = nn.MaxPool3d(2,2,0)
#         for n in feature_list:
#             enc_block = BaseAE._block(in_channels, n, self.attention, dropout)
#             self.encoder.append(enc_block)
#             in_channels = n

#         # BOTTLENECK EXTRACTION
#         bn_features= 2*feature_list[-1]
#         self.bottleneck = BaseAE._block(feature_list[-1], bn_features, self.attention, dropout)

#         # DECODER BLOCKS
#         feature_list = feature_list[::-1]
#         self.decoder = nn.ModuleList()
#         self.transposers = nn.ModuleList()
#         enlarger = 2 if self.skip_connections else 1
#         for n in feature_list:
#             upconv = nn.ConvTranspose3d(bn_features, n, 2, 2, 0)
#             dec_block = BaseAE._block(n * enlarger, n, self.attention, dropout)
#             self.transposers.append(upconv)
#             self.decoder.append(dec_block)
#             bn_features = n
            
#         # FINAL CONVOLUTION
#         self.last_conv = nn.Conv3d(feature_list[-1], self.num_mod, 1, 1, 0)


#     def forward(self, x):
#         ## ENCODING
#         encodings = []
#         for k in range(self.num_blocks):
#             enc = self.encoder[k](x)
#             x = self.pool(enc)
#             encodings.append(enc)
            
#         ## REPRESENTATION EXTRACTION
#         bottleneck = self.bottleneck(x)

#         ## DECODING
#         encodings.reverse()
#         dec = bottleneck
#         for k in range(self.num_blocks): 
#             dec = self.transposers[k](dec, output_size= encodings[k].shape)
#             if self.skip_connections:
#                 dec = torch.cat((dec, encodings[k]), dim=1)
#             dec = self.decoder[k](dec)

#         dec = self.last_conv(dec)

#         return torch.sigmoid(dec), bottleneck, None

#     @staticmethod
#     def _block(in_channels, features, attn, dropout):
#         block = nn.Sequential(
#             nn.Conv3d(in_channels, features, 3, 1, 1, bias=False),
#             nn.BatchNorm3d(features),
#             nn.ReLU(inplace=True),

#             nn.Conv3d(features, features, 3, 1, 1, bias=False),
#             nn.BatchNorm3d(features),
#             nn.ReLU(inplace=True),
#             nn.Dropout3d(dropout)
#         )
        
#         if attn:
#             attention = SelfAttention(features)  # Add attention mechanism
#             block = nn.Sequential(block, attention)

#         return block