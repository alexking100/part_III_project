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
    def __init__(self, result, dimensions):
        self.result = result
        self.dimensions = dimensions
        self.mean_accuracy, self.var_accuracy = self.check_accuracy()

    def check_accuracy(self):
        accuracy = torch.empty(self.dimensions["max_iter_time"])
        for k in range(self.dimensions["max_iter_time"]):
            accuracy[k] = torch.sum((self.result[k][0] - self.result[k][1]) ** 2) / (
                self.dimensions["grid_size"] ** 2
            )
        return torch.mean(accuracy), torch.var(accuracy)

    # add some more metrics to this class
    def more_metrics_go_here(self):
        pass


def plot_losses(loss, lupi_loss, conv_loss):
    fig = plt.figure()
    plt.axhline(color="black")
    plt.axvline(color="black")
    plt.title("Epoch Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss per Epoch")
    plt.ylim(-500000, 500000)
    plt.ticklabel_format(style="sci", axis="both")
    if loss is not None:
        plt.plot(loss, label="basic NP", alpha=0.8)
    if lupi_loss is not None:
        plt.plot(lupi_loss, label="LUPI NP", alpha=0.8)
    if conv_loss is not None:
        plt.plot(conv_loss, label="Convolutional Encoder NP", alpha=0.8)
    plt.legend()
    plt.grid()
    plt.show()


def single_hm(u_k, k: int, max_iter_time: int, which_plot: str):
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
    plt.title(
        "Temperature at t = {:.3f} unit time, {}".format(k / max_iter_time, which_plot)
    )
    plt.pcolormesh(u_k[i], cmap=plt.cm.jet, vmin=0, vmax=2.0)
    plt.colorbar()
    # plt.show()
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

    # This is to plot u_k (u at time-step k)
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
