from turtle import forward
import torch
from base import BaseVAE
from torch import nn
from torch.nn import functional as F

class ConditionalVAE(BaseVAE):
    def __init__(self,
                 in_channels,
                 num_classes,
                 latent_dim,
                 hidden_dims=None,
                 img_size=64,
                 **kwargs):
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        # Conditional Embedding | Representations to be conditioned on
        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        # For the extra label channel
        in_channels += 1

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                                nn.Conv2d(in_channels, out_channels=h_dim,
                                          kernel_size= 3, stride= 2, padding  = 1),
                                nn.BatchNorm2d(h_dim),
                                nn.LeakyReLU()))
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Normal distribution parameters
        # Note that we could use fixed variance as well
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        
        
        # Build Decoder
        modules = []
        
        # Conditioning happens here
        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                                           kernel_size=3, stride = 2, padding=1, output_padding=1),
                                         nn.BatchNorm2d(hidden_dims[i + 1]),
                                         nn.LeakyReLU()))
        
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                                                            kernel_size=3, stride=2, padding=1, output_padding=1),
                                        nn.BatchNorm2d(hidden_dims[-1]),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                                kernel_size= 3, padding= 1),
                                        nn.Tanh())

    def encode(self, input):
        '''
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        Args:
            input [Tensor]: Input image of shape [B x C x H x W]
        
        Returns:
            z [Tensor]: Latent code
        '''
        result = self.encoder(input)
        # Flatten the feature map
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu, log_var):
        '''
        Apply reparameterization trick:
               << mu + std * eps >> 
        '''
        # Obtain std
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input, **kwargs):
        y = kwargs['labels'].float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)

        # Concatenate embedded input with embedded classes
        x = torch.cat([embedded_input, embedded_class], dim=1)

        # Get parameters
        mu, log_var = self.encode(x)
        
        # Get latent variable
        z = self.reparameterize(mu, log_var)
        
        # Concatenate z with labels
        z = torch.cat([z, y], dim=1)

        return [self.decode(z), input, mu, log_var]
    
    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        mu, log_var = args[2], args[3]

        # Reconstruction Loss
        recons_loss = F.mse_loss(recons, input)

        # KL Divergence
        kl_weight = kwargs['M_N']
        kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        # Formulate loss
        loss = recons_loss + kl_weight * kl

        return {'loss': loss, 'reconstruction_loss': recons_loss, 'KL': kl}
    
    def sample(self, num_samples, current_device, **kwargs):
        y = kwargs['labels'].float()
        z = torch.randn(num_samples, self.latent_dim)
        
        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples
    
    def generate(self, x, **kwargs):
        '''
        Given an image x returns the reconstructed image
        '''
        return self.forward(x, **kwargs)[0]
