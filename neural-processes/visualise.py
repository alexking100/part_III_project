import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib as mpl

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 12,
        "figure.figsize": [8, 6],
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "legend.fontsize": "medium",
        "axes.labelsize": "medium",
        "xtick.labelsize": "medium",
        "ytick.labelsize": "medium",
    }
)

"""
standardize fonts and figures 
"""


class Performance:
    def __init__(self, y_target, y_mean, y_var, dimensions):
        self.y_target = y_target
        self.y_mean = y_mean
        self.y_var = y_var
        self.dimensions = dimensions

    def residual(self):
        return (self.y_target - self.y_mean) ** 2

    def gaussian(self, x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor):
        normalisation = 1 / (2 * np.pi * var) ** 0.5
        return normalisation * np.exp(-((x - mu) ** 2) / (2 * var))

    def log_likelihood(self) -> torch.Tensor:
        # Gaussian
        return self.gaussian(self.y_target, self.y_mean, self.y_var)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def plot_losses(loss, lupi_loss, conv_loss, plot_moving_average=True):
    fig = plt.figure()
    plt.axhline(color="black")
    plt.axvline(color="black")
    plt.title("Epoch Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss per Epoch")
    plt.ylim(-800000, 500000)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(5, 5))
    if loss is not None:
        plt.plot(loss, label="basic NP", alpha=0.8)
    if lupi_loss is not None:
        plt.plot(lupi_loss, label="LUPI NP", alpha=0.8)
    if conv_loss is not None:
        plt.plot(conv_loss, label="Convolutional Encoder NP", alpha=0.8)
    plt.legend()
    plt.grid()
    plt.show()

    if plot_moving_average:
        fig = plt.figure()
        plt.axhline(color="black")
        plt.axvline(color="black")
        plt.title("Epoch Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss per Epoch")
        plt.ylim(-800000, 500000)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(5, 5))
        if loss is not None:
            sparse_loss = moving_average(loss, 10)
            plt.plot(sparse_loss, label="basic NP", alpha=0.8)
        if lupi_loss is not None:
            sparse_lupi_loss = moving_average(lupi_loss, 10)
            plt.plot(sparse_lupi_loss, label="LUPI NP", alpha=0.8)
        if conv_loss is not None:
            sparse_conv_loss = moving_average(conv_loss, 10)
            plt.plot(sparse_conv_loss, label="Convolutional Encoder NP", alpha=0.8)
        plt.legend()
        plt.grid()
        plt.show()


def single_hm(result_k, k: int, max_iter_time: int, which_plot: str, title=None):
    if which_plot == "Target":
        i = 0
    elif which_plot == "Mean":
        i = 1
    elif which_plot == "Variance":
        i = 2
    else:
        print("Please put either Target, Mean or Variance for argument which_plot")
        raise KeyError

    # Clear the current plot figure
    plt.clf()
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    if title is None:
        plt.title(
            "Temperature at t = {:.3f} unit time, {}".format(
                k / max_iter_time, which_plot
            )
        )
    else:
        plt.title("{}, t = {:.3f}".format(title, k / max_iter_time))
    plt.pcolormesh(result_k[i], cmap=plt.cm.jet, vmin=0, vmax=2.0)
    plt.colorbar()
    plt.show()
    return plt


def plot_noisey_response(noise_output, time_iter, title=None):
    plt.xlabel("x")
    plt.ylabel("y")
    # This is to plot u_k (u at time-step k)
    plt.title("{}, t = {:.3f}".format(title, time_iter / noise_output.shape[0]))
    plt.pcolormesh(noise_output[time_iter], cmap=plt.cm.jet, vmin=0, vmax=2.0)
    plt.colorbar()
    plt.show()
    return plt


def visualise_model_solution(y, time_iter, title="", vmin=0.0, vmax=2.0):
    plt.xlabel("x")
    plt.ylabel("y")
    # This is to plot u_k (u at time-step k)
    plt.title("{}, t = {:.3f}".format(title, time_iter / y.shape[0]))
    plt.pcolormesh(y[time_iter], cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()
    return plt


def plotheatmap(result, k, max_iter_time, mean_var_only=False):
    """
    If we input noise we don't want a target result
    """

    result_k = result[k]

    # Clear the current plot figure
    plt.clf()

    if mean_var_only:
        fig, axes = plt.subplots(
            ncols=2,
            figsize=(10, 4),
            gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1]},
        )
        ax2, ax3 = axes

    else:
        fig, axes = plt.subplots(
            ncols=3,
            figsize=(15, 4),
            gridspec_kw={"width_ratios": [1, 1, 1], "height_ratios": [1]},
        )
        ax1, ax2, ax3 = axes
        ax1.set_title(f"Target, t = {k / max_iter_time:.3f}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")
        im1 = ax1.pcolormesh(result_k[0], cmap=plt.cm.jet, vmin=0, vmax=2)

    ax2.set_title(f"Mean, t = {k / max_iter_time:.3f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax3.set_title(f"Variance, t = {k / max_iter_time:.3f}")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

    ax2.set_aspect("equal")
    ax3.set_aspect("equal")

    # This is to plot result_k (result at time-step k)
    im2 = ax2.pcolormesh(result_k[1], cmap=plt.cm.jet, vmin=0, vmax=2)
    im3 = ax3.pcolormesh(result_k[2], cmap=plt.cm.jet, vmin=0, vmax=2)
    fig.colorbar(im2, ax=axes, fraction=0.05)
    plt.show()


def plot_residuals(result, k, max_iter_time):
    result_k = result[k]

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1]},
    )
    ax1, ax2 = axes

    # plt.title(f"Temperature at t = {k * delta_t:.3f} unit time")
    ax1.set_title(f"Residual Squared, t = {k / max_iter_time:.3f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax2.set_title(f"Variance, t = {k / max_iter_time:.3f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    im1 = ax1.pcolormesh(
        (result_k[0] - result_k[1]) ** 2, cmap=plt.cm.jet, vmin=0, vmax=2
    )
    im2 = ax2.pcolormesh(result_k[2], cmap=plt.cm.jet, vmin=0, vmax=2)
    fig.colorbar(im1, ax=axes, fraction=0.05)
    plt.show()
    return im1, im2


def get_energy_and_entropy(result):
    max_iter_time = len(result)
    grid_size = result[0][0].shape[0]
    avg_energy_target = torch.full((max_iter_time,), np.nan)
    avg_energy_mean = torch.full((max_iter_time,), np.nan)
    avg_energy_var = torch.full((max_iter_time,), np.nan)
    avg_entropy_target = torch.full((max_iter_time,), np.nan)
    avg_entropy_mean = torch.full((max_iter_time,), np.nan)
    avg_entropy_var = torch.full((max_iter_time,), np.nan)

    for k in range(max_iter_time):
        avg_energy_target[k] = torch.sum(result[k][0]) / (grid_size**2)
        avg_energy_mean[k] = torch.sum(result[k][1]) / (grid_size**2)
        avg_energy_var[k] = torch.sum(result[k][2]) / (grid_size**2)
        avg_entropy_target[k] = torch.count_nonzero(result[k][0]) / (grid_size**2)
        avg_entropy_mean[k] = torch.count_nonzero(result[k][1]) / (grid_size**2)
        avg_entropy_var[k] = torch.count_nonzero(result[k][2]) / (grid_size**2)

    return (
        (avg_energy_target, avg_energy_mean, avg_energy_var),
        (avg_entropy_target, avg_entropy_mean, avg_entropy_var),
    )


def plot_pi_data(target, np_mean, lupi_mean, conv_mean, title, xlabel, ylabel):
    xvalues = [i / len(target) for i in range(len(target))]
    fig = plt.figure()
    plt.axhline(color="black")
    plt.axvline(color="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.ticklabel_format(axis="both", style="sci", scilimits=(1, 1))
    if target is not None:
        plt.plot(xvalues, target, label="Target", ls="--", linewidth=2)
    if np_mean is not None:
        plt.plot(xvalues, np_mean, label="NP", linewidth=2, color="orange")
    if lupi_mean is not None:
        plt.plot(xvalues, lupi_mean, label="LUPI", linewidth=2, color="red")
    if conv_mean is not None:
        plt.plot(xvalues, conv_mean, label="Conv", linewidth=2, color="green")

    if title == "Entropy":
        plt.ylim(5, 8)
    plt.legend()
    plt.grid()
    plt.show()

def get_pi(y):
    energy = torch.sum(y, dim=(1,2)) / (y.shape[1]**2)

    norm = torch.sum(y, dim=(1,2)).unsqueeze(1).unsqueeze(2)
    entropy = y / norm * torch.log(y / norm)
    entropy[np.isnan(entropy)] = 0
    entropy = - torch.sum(entropy, dim=(1,2))

    max_temp = torch.Tensor([torch.max(y[k]) for k in range(y.shape[0])])

    return max_temp, energy, entropy