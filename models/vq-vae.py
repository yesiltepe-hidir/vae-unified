from turtle import forward
import torch
from torch import embedding, nn
from torch.nn import functional as F
from base import BaseVAE

class VectorQuantizer(nn.Module):
    '''
    Reference: [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    '''
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        # Code book: K embeddings, each D dimensional
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        '''
        Args:
            latents [Tensor]: Encoder output z_e of shape [B x D x H x W] 
        '''
        latents = latents.permute(0, 2, 3, 1).contiguous() # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        latents_flattened = latents.view(-1, self.D) # [BWH x D]

        # Compute L2 distance between latentes and embedding weight
        dist = torch.sum(latents_flattened ** 2, dim=1, keepdim=True) +\
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 *  torch.matmul(latents_flattened, self.embedding.weight.t()) # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1) # [BHW x 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight) # [BHW x D]
        quantized_latents.view(latents_shape) # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue nack to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss # [B x D x H x W]

class ResidualLayer(nn.Module):
    '''
    Residual Layer.
    '''
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input):
        return input + self.resblock(input)

class VQVAE(BaseVAE):
    '''
    Vector-Quantized VAE

    Args:
        in_channels    [int]:   number of input channels
        embedding_dim  [int]:   dimension of each embedding e_i, D
        num_embeddings [int]:   number of embeddings, K
        hidden_dims    [List]:  hidden dimensions
        beta           [float]: hyperparameter for commitment loss
        img_size       [int]:   spatial dimension of input image
    '''
    def __init__(self, in_channels, 
                       embedding_dim, 
                       num_embeddings, 
                       hidden_dims=None, 
                       beta=0.25, 
                       img_size=64,
                       **kwargs):
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]
        
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
                           nn.LeakyReLU()))
            in_channels = h_dim
        
        modules.append(nn.Sequential(nn.Conv2d(in_channels, in_channels,
                                               kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU()))
        
        # Add Residual Blocks
        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(nn.Sequential(nn.Conv2d(in_channels, embedding_dim,
                                     kernel_size=1, stride=1)),
                                     nn.LeakyReLU())

        self.encoder = nn.Sequential(*modules)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)

        # Build Decoder
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(embedding_dim, hidden_dims[-1],
                                               kernel_size=3, stride=1, padding=1),
                                               nn.LeakyReLU()))

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        modules.append(nn.LeakyReLU())

        # Reverse the hidden dimensions
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 
                                         kernel_size=4, stride=2, padding=1),
                                         nn.LeakyReLU()))
        
        modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], out_channels=3, 
                                         kernel_size=4, stride=2, padding=1),
                                         nn.Tanh()))

        self.decoder = nn.Sequential(*modules)
    
    def encode(self, input):
        '''
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        Args:
            input [Tensor]: Input tensor of shape [B x C x H x W]
        
        Returns: 
            result [Tensor]: List of latent codes
        '''        
        result = self.encoder(input)
        return [result]

    def decode(self, z):
        '''
        Maps the given latent codes onto 
        the image space.

        Args:
            z [Tensor]: latent code of shape [B x D x H x W]
        
        Returns:
            result [Tesnor]: batch of images of shape [B x C x H x W]
        '''
        result = self.decoder(z)
        return result
    
    def forward(self, input, **kwargs):
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]
    
    def loss_function(self, *args, **kwargs):
        recons  =  args[0]
        input   =  args[1]
        vq_loss =  args[2]

        recon_loss = F.mse_loss(recons, input)
        loss = recon_loss + vq_loss
        return {'loss': loss,
                'reconstruction_loss': recon_loss,
                'vq_loss': vq_loss}
    
    def sample(self, num_samples, current_device, **kwargs):
        raise Warning('VQ-VAE sapler is not implemented')
    
    def generate(self, x):
        '''
        Given an input image x, returns the 
        reconstructed image.

        Args: 
            x [Tensor]: batch of input images of shape [B x C x H x W]
        
        Returns [Tensor]: batch of reconstructed image of shape [B x C x H x W]
        '''
        return self.forward(x)[0]
        