import torch
import torch.nn as nn
from models.attention import SelfAttention

class BaseAE(nn.Module):
    def __init__(self, in_channels, init_features, num_blocks, type_ae, attention=False, hidden_size="auto"):
        super(BaseAE, self).__init__()

        self.num_mod = in_channels
        self.num_blocks = num_blocks
        self._type = type_ae
        self.skip_connections = True if type_ae.lower() == "unet" else False
        self.attention = attention
        self.hidden_size = hidden_size

        # ENCODER BLOCKS
        feature_list = [init_features*(2**n) for n in range(num_blocks)]
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2,2,0)
        for n in feature_list:
            enc_block = BaseAE._block(in_channels, n, self.attention)
            self.encoder.append(enc_block)
            in_channels = n

        # BOTTLENECK EXTRACTION
        bn_features= 2*feature_list[-1]
        self.bottleneck = BaseAE._block(feature_list[-1], bn_features, self.attention)

        # DECODER BLOCKS
        feature_list = feature_list[::-1]
        self.decoder = nn.ModuleList()
        self.transposers = nn.ModuleList()
        enlarger = 2 if self.skip_connections else 1
        for n in feature_list:
            upconv = nn.ConvTranspose3d(bn_features, n, 2, 2, 0)
            dec_block = BaseAE._block(n * enlarger, n, self.attention)
            self.transposers.append(upconv)
            self.decoder.append(dec_block)
            bn_features = n
            
        # FINAL CONVOLUTION
        self.last_conv = nn.Conv3d(feature_list[-1], self.num_mod, 1, 1, 0)


    def forward(self, x):
        ## ENCODING
        encodings = []
        for k in range(self.num_blocks):
            enc = self.encoder[k](x)
            x = self.pool(enc)
            encodings.append(enc)
            
        ## REPRESENTATION EXTRACTION
        bottleneck = self.bottleneck(x)

        ## DECODING
        encodings.reverse()
        dec = bottleneck
        for k in range(self.num_blocks): 
            dec = self.transposers[k](dec, output_size= encodings[k].shape)
            if self.skip_connections:
                dec = torch.cat((dec, encodings[k]), dim=1)
            dec = self.decoder[k](dec)

        dec = self.last_conv(dec)

        return torch.sigmoid(dec), bottleneck, None

    @staticmethod
    def _block(in_channels, features, attn):
        block = nn.Sequential(
            nn.Conv3d(in_channels, features, 3, 1, 1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),

            nn.Conv3d(features, features, 3, 1, 1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )
        
        if attn:
            attention = SelfAttention(features)  # Add attention mechanism
            block = nn.Sequential(block, attention)            

        return block
