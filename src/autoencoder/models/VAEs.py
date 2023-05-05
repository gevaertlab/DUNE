from abc import abstractmethod
from torch import nn
from torch import tensor as Tensor
import torch
from typing import List, Any



### U-shaped VAE (no skip connections)
class U_VAE(nn.Module):
    def __init__(self, in_channels, init_features, num_blocks, input_dim, hidden_size=2048):
        super(U_VAE, self).__init__()

        self.num_mod = in_channels
        self.num_blocks = num_blocks
        self._type = "U_VAE"
        self.hidden_size = hidden_size

        # ENCODER BLOCKS
        feature_list = [init_features*(2**n) for n in range(num_blocks)]
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2, 0)
        for n in feature_list:
            enc_block = U_VAE._block(in_channels, n)
            self.enc_blocks.append(enc_block)
            in_channels = n

        # BOTTLENECK EXTRACTION
        enc_shape, size = U_VAE._get_encoded_shape(input_dim, feature_list)
        self.mu = nn.Linear(size, self.hidden_size)
        self.sigma = nn.Linear(size, self.hidden_size)
        self.decoder_input = nn.Linear(self.hidden_size, size)

        # DECODER BLOCKS
        feature_list = feature_list[::-1]
        self.dec_blocks = nn.ModuleList()
        self.transposers = nn.ModuleList()
        for n in feature_list:
            upconv = nn.ConvTranspose3d(enc_shape[0], n, 2, 2, 0)
            dec_block = U_VAE._block(n, n)
            self.transposers.append(upconv)
            self.dec_blocks.append(dec_block)
            enc_shape[0] = n

        # FINAL CONVOLUTION
        self.last_conv = nn.Conv3d(feature_list[-1], self.num_mod, 1, 1, 0)

    def reparametrize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return eps * sigma + mu

    def encode(self, x):
        encodings = []

        for k in range(self.num_blocks):
            enc = self.enc_blocks[k](x)
            encodings.append(enc)
            x = self.pool(enc)

        return x, encodings

    def decode(self, x, encodings):
        for k in range(self.num_blocks):
            x = self.transposers[k](x, output_size=encodings[k].shape)
            x = self.dec_blocks[k](x)

        return self.last_conv(x)

    def forward(self, x):
        # ENCODING
        enc, encodings = self.encode(x)

        # REPRESENTATION EXTRACTION
        mu = self.mu(enc.view(enc.size(0), -1))
        sigma = self.sigma(enc.view(enc.size(0), -1))
        sigma =  torch.exp(0.5 * sigma)
        bottleneck = self.reparametrize(mu, sigma)

        # DECODING
        encodings.reverse()
        dec = self.decoder_input(bottleneck)
        dec = dec.view(enc.shape)
        dec = self.decode(dec, encodings)

        return torch.sigmoid(dec), bottleneck, (mu, sigma)

    @staticmethod
    def _block(in_channels, features):
        block = nn.Sequential(
            nn.Conv3d(in_channels, features, 3, 1, 1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),

            nn.Conv3d(features, features, 3, 1, 1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )
        return block

    @staticmethod
    def _get_encoded_shape(input_dim, feature_list):

        def block_shaping(dim):
            k, s, p = 3, 1, 1
            dim = int(((dim + 2*p - (k-1) - 1) / s) + 1)  # block conv1
            dim = int(((dim + 2*p - (k-1) - 1) / s) + 1)  # block conv2

            k, s, p = 2, 2, 0
            dim = int(((dim + 2*p - (k-1) - 1) / s) + 1)  # maxpool
            return dim

        D, H, W = input_dim
        for C in feature_list:
            D = block_shaping(D)
            H = block_shaping(H)
            W = block_shaping(W)

        C = feature_list[-1]

        return [C, D, H, W], int(C*D*H*W)




#### Terry's VAE
class VAEBackbone(nn.Module):

    def __init__(self) -> None:
        super(VAEBackbone, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> dict:
        pass


class VAE3D(VAEBackbone):

    def __init__(self,
                 min_dims,
                 in_channels: int = 3,
                 latent_dim: int = 2048,
                 hidden_dims: List = None,
                 example_input_shape=None,
                 **kwargs):  # -> None
        super(VAE3D, self).__init__()

        self._type = "VAE"
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        if hidden_dims is None:
            hidden_dims = [4, 16, 32, 64, 128]
        # don't modify hidden_dims
        self.hidden_dims = hidden_dims.copy()
        hidden_dims_variable = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims_variable:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels=in_channels,
                              out_channels=h_dim,  
                              kernel_size=3,
                              stride=2,
                              padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        self.encoded_shapes, flatten_shape = VAE3D._get_encoded_shape(min_dims, hidden_dims)

        self.fc_mu = nn.Linear(flatten_shape, latent_dim)
        self.fc_var = nn.Linear(flatten_shape, latent_dim)

        # Build Decoder
        hidden_dims_variable.reverse()

        self.decoder_input = nn.Linear(latent_dim, flatten_shape) 
        self.decoder, self.decode_norm = nn.ModuleList(), nn.ModuleList()
        for i, _ in enumerate(hidden_dims_variable[:-1]):
            self.decoder.append(
                    nn.ConvTranspose3d(hidden_dims_variable[i], 
                                       hidden_dims_variable[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1))
            self.decode_norm.append(
                nn.Sequential(
                    nn.BatchNorm3d(hidden_dims_variable[i + 1]),
                    nn.LeakyReLU())
            )


        # self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(hidden_dims_variable[-1],
                               hidden_dims_variable[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm3d(hidden_dims_variable[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_dims_variable[-1], out_channels=self.in_channels,
                      kernel_size=3, padding=1),  
            nn.Sigmoid()) 


    def encode(self, input: Tensor):  # -> List[torch.Tensor@encode]
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x L x W x H]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor):  # -> Tensor
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C=1 x L x W x H]
        """
        result = self.decoder_input(z)
        rev_shapes = self.encoded_shapes.copy()
        rev_shapes.reverse()
    
        result = result.view(-1, *rev_shapes[0])
        for i, decoder_block in enumerate(self.decoder):
            result = decoder_block(result, output_size=rev_shapes[i+1][1:])
            result = self.decode_norm[i](result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, sigma: Tensor):  # -> Tensor
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        eps = torch.randn_like(sigma)
        return eps * sigma + mu

    def forward(self, input: Tensor, **kwargs):  # -> List[Tensor]
        mu, sigma = self.encode(input)
        sigma = torch.exp(0.5 * sigma)

        z = self.reparameterize(mu, sigma)
        reconst = self.decode(z)
        return reconst, z, (mu, sigma)


    def sample(self,
               num_samples: int,
               current_device: int,
               **kwargs):  # -> Tensor
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim, device=current_device)

        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs):  # -> Tensor
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    @staticmethod
    def _get_encoded_shape(input_dim, feature_list):

        def block_shaping(dim):
            k, s, p = 3, 2, 1
            dim = int(((dim + 2*p - (k-1) - 1) / s) + 1)  # block conv1

            return dim

        D, H, W = input_dim
        shapes = []
        for C in feature_list:
            D = block_shaping(D)
            H = block_shaping(H)
            W = block_shaping(W)
            shapes.append([C, D,H,W])
        
        final_size = int(C*D*H*W)

        return shapes, final_size
