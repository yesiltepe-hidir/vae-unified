import torch
from torch import nn
from torch.nn import functional as F
from base import BaseVAE

class WAE_MMD(BaseVAE):
    '''
    
    '''
    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims=None,
                 reg_weight=100,
                 kernel_type='imq',
                 latent_var=2.0,
                 **kwargs):
        super(WAE_MMD, self).__init__()

        # Initialize hyperparameters
        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var

        # Define network
        modules = []
        if hidden_dims is None:
            hiddden_dims = [32, 64, 128, 256, 512]
        
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                                nn.Conv2d(in_channels, out_channels=h_dim,
                                           kernel_size=3, stride=2, padding=1),
                                nn.BatchNorm2d(h_dim),
                                nn.LeakyReLU()))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

        # Reverse hidden dimensions
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                                    kernel_size=3, stride=2, padding=1, output_padding=1),
                                nn.BatchNorm2d(hidden_dims[i + 1]),
                                nn.LeakyReLU()))

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                                               kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels = 3,
                                      kernel_size = 3, padding = 1),
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
        
        # Flatten the result into a vector of shape [B x L]
        result = torch.flatten(result, start_dim=1)

        # Split the results into mu and var components
        # of the latent Gaussian Distribution
        z = self.fc_z(result)
        return z
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def forward(self, input, **kwargs):
        z = self.encode(input)
        return [self.decode(z), input, z]
    
    def loss_function(self, *args, **kwargs):
        # Get reconstruction, input and latent variable 
        recons = args[0]
        input  = args[1]
        z = args[2]

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr

        # Reconstruction Loss
        recons_loss = F.mse_loss(recons, input)

        # Maximum Mean Discrepancy (MMD) Loss
        mmd_loss = self.compute_mmd(z, reg_weight)

        loss = recons_loss + mmd_loss
        return {'loss': loss, 'reconstruction_loss': recons_loss, 'mmd_loss': mmd_loss}

    def compute_kernel(self, x1, x2):
        # Convert the rensors into row and column vectors
        D = x1.size(0)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result
    
    def compute_rbf(self, x1, x2, eps = 1e-7):
        """
        Computes the RBF Kernel between x1 and x2.
        
        Args:
            x1 (Tensor)
            x2 (Tensor)
            eps (float)
        """

        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self, x1, x2, eps = 1e-7):
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by
                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z, reg_weight):
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = reg_weight * prior_z__kernel.mean() + \
              reg_weight * z__kernel.mean() - \
              2 * reg_weight * priorz_z__kernel.mean()
        return mmd

    def sample(self, num_samples, current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
