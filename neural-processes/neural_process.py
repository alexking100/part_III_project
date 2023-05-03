import torch
from models import (
    Net,
    Encoder,
    LupiEncoder,
    RepresentationAggregator,
    MuSigmaEncoder,
    Decoder,
    TransposedConvNet,
    TemporalConvNet,
)
from torch import nn
from torch.distributions import Normal
from utils import img_mask_to_np_input


class NeuralProcessConv(nn.Module):
    """
    Implements Neural Process with Convolutional Neural Networks
    in the Encoder and Decoder.

    Parameters
    ----------
    t_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer in encoder and decoder.

    grid_size: int
        Dimension of grid-like input (one side).
    """

    def __init__(self, dimensions):
        super(NeuralProcessConv, self).__init__()
        self.t_dim = dimensions["t_dim"]
        self.y_dim = dimensions["y_dim"]
        self.r_dim = dimensions["r_dim"]
        self.z_dim = dimensions["z_dim"]
        self.h_dim = dimensions["h_dim"]
        self.grid_size = dimensions["grid_size"]

        # Initialize networks
        self.xy_to_r = Net(dimensions)
        self.r_to_mu_sigma = MuSigmaEncoder(dimensions)
        # self.xz_to_y = TransposedConvNet(dimensions)
        self.xz_to_y = Decoder(dimensions)

    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, t_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.t_dim)
        y_grid = y.contiguous().view(
            batch_size * num_points, self.grid_size, self.grid_size
        )
        y_grid = y_grid.unsqueeze(1)
        # Insert the x values into the y_grid as a second channel
        # if there is a nicer way to do this, please find it
        x_grid = torch.zeros_like(y_grid)
        for i in range(batch_size * num_points):
            x_grid[i, :, :, :] = torch.full((1, 50, 50), x_flat[i, 0])
        xy_grid = torch.cat((y_grid, x_grid), dim=1)

        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, xy_grid)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return parameters of distribution from which we sample latent variable z
        return self.r_to_mu_sigma(r)

    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, t_dim). Note that x_context is a
            subset of x_target.

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, t_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        batch_size, num_context, t_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        if self.training:
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from encoded distribution using reparameterization trick
            # ensures z_sample is differentiable with respect to the mu and sigma
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            """
            QUESTION:
            Is it obvious that we should sample z from q_target not q_context?
            When we test, we sample z from q_context because we don't have a target dataset
            """
            z_sample = q_target.rsample()
            # Get parameters of output distribution
            # Decode maps target x and z_sample to a prediction
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred


class NeuralProcess(nn.Module):
    """
    Implements Neural Process for functions of arbitrary dimensions.

    Parameters
    ----------
    t_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """

    def __init__(self, dimensions):
        super(NeuralProcess, self).__init__()
        self.t_dim = dimensions["t_dim"]
        self.y_dim = dimensions["y_dim"]
        self.r_dim = dimensions["r_dim"]
        self.z_dim = dimensions["z_dim"]
        self.h_dim = dimensions["h_dim"]
        self.pi_dim = dimensions["pi_dim"]

        # Initialize networks
        self.xy_to_r = Encoder(dimensions)
        self.r_to_mu_sigma = MuSigmaEncoder(dimensions)
        self.xz_to_y = Decoder(dimensions)
        self.lupi_to_r = LupiEncoder(dimensions)
        self.aggregate_lupi = RepresentationAggregator(dimensions)
        # self.tcn_aggregate = TemporalConvNet(dimensions)

    def aggregate(self, r_i, tcn=False):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        if tcn == False:
            return torch.mean(r_i, dim=1)

        elif tcn == True:
            pass

    def xy_to_mu_sigma(self, x, y, pi):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, t_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.t_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)

        # Encode pi into a representation r_pi then aggregate with r
        # Only valid for target data during training
        if pi is not None:
            pi = pi.view(batch_size * num_points, self.pi_dim)
            r_pi_flat = self.lupi_to_r(x_flat, pi)
            r_pi = r_pi_flat.view(batch_size, num_points, self.r_dim)
            r_pi = self.aggregate(r_pi)
            r = self.aggregate_lupi(r_pi, r)

        # Return parameters of distribution from which we sample latent variable z
        return self.r_to_mu_sigma(r)

    def forward(
        self,
        x_context,
        y_context,
        x_target,
        y_target=None,
        pi_context=None,
        pi_target=None,
    ):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, t_dim). Note that x_context is a
            subset of x_target.

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, t_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

        pi_context : torch.Tensor or None
            Shape (batch_size, num_target, pi_dim). Only used during training.

        pi_target : torch.Tensor or None
            Shape (batch_size, num_target, pi_dim). Only used during training.

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        batch_size, num_context, t_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        if self.training:
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target, pi_target)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context, None)
            # Sample from encoded distribution using reparameterization trick
            # ensures z_sample is differentiable with respect to the mu and sigma
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_target.rsample()
            # Get parameters of output distribution
            # Decode maps target x and z_sample to a prediction
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_mu_sigma(
                x_context, y_context, pi=None
            )
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred


class NeuralProcessImg(nn.Module):
    """
    Wraps regular Neural Process for image processing.

    Parameters
    ----------
    img_size : tuple of ints
        E.g. (1, 28, 28) or (3, 32, 32)

    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """

    def __init__(self, img_size, r_dim, z_dim, h_dim):
        super(NeuralProcessImg, self).__init__()
        self.img_size = img_size
        self.num_channels, self.height, self.width = img_size
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.neural_process = NeuralProcess(
            t_dim=2, y_dim=self.num_channels, r_dim=r_dim, z_dim=z_dim, h_dim=h_dim
        )

    def forward(self, img, context_mask, target_mask):
        """
        Given an image and masks of context and target points, returns a
        distribution over pixel intensities at the target points.

        Parameters
        ----------
        img : torch.Tensor
            Shape (batch_size, channels, height, width)

        context_mask : torch.ByteTensor
            Shape (batch_size, height, width). Binary mask indicating
            the pixels to be used as context.

        target_mask : torch.ByteTensor
            Shape (batch_size, height, width). Binary mask indicating
            the pixels to be used as target.
        """
        x_context, y_context = img_mask_to_np_input(img, context_mask)
        x_target, y_target = img_mask_to_np_input(img, target_mask)
        return self.neural_process(x_context, y_context, x_target, y_target)
