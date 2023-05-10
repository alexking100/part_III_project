import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import (
    Sequential,
    Linear,
    Conv2d,
    ConvTranspose2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
)
import numpy as np


class Net(nn.Module):
    """
    Convolutional Neural Network

    Applies a filter to reduce dimensionality
    and provide a better representation of data

    Layers
    ------

    Conv2d: defines filter to pass over grid
    BatchNorm2d: normalises
    MaxPool2d: aggregates the convolved grid points and takes the maximum

    """

    def __init__(self, dimensions):
        super(Net, self).__init__()

        self.h_dim = dimensions["h_dim"]
        self.r_dim = dimensions["r_dim"]
        self.grid_size = dimensions["grid_size"]
        self.input_channels = 2
        self.out_channels = 2

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(
                self.input_channels,
                dimensions["num_channels"],
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            BatchNorm2d(dimensions["num_channels"]),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1),
            # Defining another 2D convolution layer
            Conv2d(
                dimensions["num_channels"],
                self.out_channels,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            BatchNorm2d(self.out_channels),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1),
        )

        # Calculate the initial dimension of the linear layer after convolution layers
        self.n_conv = self.dim_out_conv(
            dim_in=self.grid_size, dim_k=5, padding=0, stride=1
        )
        self.n_conv = self.dim_out_pool(
            dim_in=self.n_conv, dim_k=3, padding=0, stride=2
        )
        self.n_conv = self.dim_out_conv(
            dim_in=self.n_conv, dim_k=5, padding=0, stride=1
        )
        self.n_conv = self.dim_out_pool(
            dim_in=self.n_conv, dim_k=3, padding=0, stride=2
        )
        self.n_conv = self.n_conv**2
        self.n_conv *= self.out_channels

        # for time to be appended
        self.n_conv += 1

        self.linear_layers = Linear(self.n_conv, self.r_dim)

    def dim_out_conv(self, dim_in: int, dim_k: int, padding: int, stride: int) -> int:
        out = (dim_in - dim_k + 2 * padding) / stride + 1
        # make sure we have an integer as expected
        assert out.is_integer()
        return int(out)

    def dim_out_pool(
        self, dim_in: int, dim_k: int, padding: int, stride: int, dilation: int = 1
    ) -> int:
        out = (dim_in + 2 * padding - dilation * (dim_k - 1)) / stride
        # Round down to nearest integer (as some pixels on edge
        # are not pooled if kernel does not fit perfectly)
        out = int(out)
        return out

    # Defining the forward pass
    def forward(self, x, xy):
        """
        x : torch.Tensor
            Shape (batch_size * num_points, t_dim)

        xy (input) : torch.Tensor
            Shape (batch_size * num_points, num_channels=2, grid_size, grid_size)
        """
        # y contains two channels, one is y value one is x value
        xy = self.cnn_layers(xy)

        # flatten the output of the ConvNet
        xy = xy.view(xy.size(0), -1)

        # combine x and xy when xy is flattened
        xy = torch.cat((x, xy), dim=1)
        r = self.linear_layers(xy)

        return r


class TransposedConvNet(nn.Module):
    def __init__(self, dimensions):
        super(TransposedConvNet, self).__init__()
        self.t_dim = dimensions["t_dim"]
        self.z_dim = dimensions["z_dim"]
        self.grid_size = dimensions["grid_size"]
        self.y_dim = dimensions["y_dim"]
        self.num_channels = dimensions["num_channels"]
        assert self.y_dim == self.grid_size * self.grid_size

        # We need to hard code the output to be 144 to allow
        # the ConvTranspose2d to reach 50*50 grid
        self.linear_layer = Linear(self.z_dim + self.t_dim, 144)

        self.hidden_to_mu = Sequential(
            # First transposed convolutional layer
            # Kernel, Stride and Padding are defined to achieve correct grid dimension
            ConvTranspose2d(
                in_channels=1,
                out_channels=self.num_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            BatchNorm2d(num_features=self.num_channels),
            ReLU(inplace=True),
            # Second transposed convolutional layer
            ConvTranspose2d(
                self.num_channels, out_channels=1, kernel_size=4, stride=2, padding=0
            ),
            BatchNorm2d(num_features=1),
        )

        self.hidden_to_sigma = Sequential(
            # First transposed convolutional layer
            # K, S and P are defined to achieve correct grid dimension
            ConvTranspose2d(
                in_channels=1,
                out_channels=self.num_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            BatchNorm2d(num_features=self.num_channels),
            ReLU(inplace=True),
            # Second transposed convolutional layer
            ConvTranspose2d(
                self.num_channels, out_channels=1, kernel_size=4, stride=2, padding=0
            ),
            BatchNorm2d(num_features=1),
        )

    def forward(self, x, z):
        """
        Take latent sampled vector (z) and target time (x)
        to predictive mean and variance over the grid space
        using a Transposed Convolutional Neural Network.

        x : torch.Tensor
            Shape (batch_size, num_points, t_dim)

        z : torch.Tensor
            Shape (batch_size, num_points, z_dim)

        input_pairs : torch.Tensor
            Shape (batch_size * num_points, t_dim + z_dim)

        z_upsampled : torch.Tensor
            Shape (batch_size * num_points, num_channels=1, 12, 12)

        hidden_grid : torch.Tensor
            Shape (batch_size * num_points, num_channels=self.num_channels, 24, 24)

        mean_grid, sigma_grid : torch.Tensor
            Shape (batch_size * num_points, num_channels=1, 50, 50)
        """

        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.t_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=1)

        z_upsampled = self.linear_layer(input_pairs)

        # z_upsampled outputs a layer of size 144 so we
        # can put this into a 12*12 grid, then the
        # 12*12 grid is convolved into 50*50
        z_upsampled = z_upsampled.view(batch_size * num_points, 12, 12)

        # Make room for num_channels
        z_upsampled = z_upsampled.unsqueeze(1)

        # Train two different convolutions to get the mean, sigma
        # output shape (batch_size, num_channels=1, grid_size, grid_size)
        # Apply sigmoid to make them positive
        mean_grid = torch.sigmoid(self.hidden_to_mu(z_upsampled))
        sigma_grid = torch.sigmoid(self.hidden_to_sigma(z_upsampled))

        # Shape (batch_size, grid_size, grid_size)
        mean_grid = mean_grid.squeeze(1)
        sigma_grid = sigma_grid.squeeze(1)

        # Shape (batch_size, y_dim)
        mean_grid = mean_grid.reshape(batch_size, num_points, self.y_dim)
        sigma_grid = sigma_grid.reshape(batch_size, num_points, self.y_dim)

        return mean_grid, sigma_grid


class LupiEncoder(nn.Module):
    """Maps lupi vector to a representation l_i.

    Parameters
    ----------
    t_dim : int
        Dimension of x values.

    lupi_dim : int
        Dimension of lupi values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation l (same as r).
    """

    def __init__(self, dimensions):
        super(LupiEncoder, self).__init__()
        self.t_dim = dimensions["t_dim"]
        self.pi_dim = dimensions["pi_dim"]
        self.h_dim = dimensions["h_dim"]
        self.r_dim = dimensions["r_dim"]

        layers = [
            nn.Linear(self.pi_dim + self.t_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.r_dim),
        ]
        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, pi):
        """
        x : torch.Tensor
            Shape (batch_size, t_dim)

        lupi : torch.Tensor
            Shape (batch_size, pi_dim)
        """
        input_pairs = torch.cat((x, pi), dim=1)

        return self.input_to_hidden(input_pairs)


class RepresentationAggregator(nn.Module):
    def __init__(self, dimensions):
        super(RepresentationAggregator, self).__init__()
        self.r_dim = dimensions["r_dim"]
        self.pi_dim = dimensions["pi_dim"]
        self.h_dim = dimensions["h_dim"]

        layers = [
            nn.Linear(self.r_dim + self.r_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.r_dim),
        ]
        self.hidden_to_representation = nn.Sequential(*layers)

    def forward(self, r_pi, r):
        return r + self.hidden_to_representation(torch.cat((r_pi, r), dim=1))


class TemporalConvNet(nn.Module):
    def __init__(self, dimensions, context=True):
        super(TemporalConvNet, self).__init__()
        num_channels = dimensions["num_channels"]
        kernel_size = dimensions["tcn_kernel_size"]

        self.tcn_layer = nn.Sequential(
            nn.Conv1d(
                dimensions["r_dim"],
                num_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                num_channels, num_channels, kernel_size, padding=(kernel_size - 1) // 2
            ),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                num_channels, num_channels, kernel_size, padding=(kernel_size - 1) // 2
            ),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                num_channels, num_channels, kernel_size, padding=(kernel_size - 1) // 2
            ),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                num_channels,
                dimensions["r_dim"],
                kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
        )
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.tcn_layer(x)
        # y = self.dropout(y)
        return y[:, :, -1]


class Encoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    t_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """

    def __init__(self, dimensions):
        super(Encoder, self).__init__()

        self.t_dim = dimensions["t_dim"]
        self.y_dim = dimensions["y_dim"]
        self.h_dim = dimensions["h_dim"]
        self.r_dim = dimensions["r_dim"]

        layers = [
            nn.Linear(self.t_dim + self.y_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.r_dim),
        ]
        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, t_dim)

        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        input_pairs = torch.cat((x, y), dim=1)

        # # debugging
        # print(list(self.input_to_hidden.parameters()))

        return self.input_to_hidden(input_pairs)


class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, dimensions):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = dimensions["r_dim"]
        self.z_dim = dimensions["z_dim"]

        self.r_to_hidden = nn.Linear(self.r_dim, self.r_dim)
        self.hidden_to_mu = nn.Linear(self.r_dim, self.z_dim)
        self.hidden_to_sigma = nn.Linear(self.r_dim, self.z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        sigma = torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class Decoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    t_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.
    """

    def __init__(self, dimensions):
        super(Decoder, self).__init__()

        self.t_dim = dimensions["t_dim"]
        self.z_dim = dimensions["z_dim"]
        self.h_dim = dimensions["h_dim"]
        self.y_dim = dimensions["y_dim"]

        layers = [
            nn.Linear(self.t_dim + self.z_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
        ]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(self.h_dim, self.y_dim)
        self.hidden_to_sigma = nn.Linear(self.h_dim, self.y_dim)

    def forward(self, x, z):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, t_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.t_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        sigma = F.softplus(pre_sigma)

        return mu, sigma
