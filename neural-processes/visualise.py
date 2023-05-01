import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np


class Performance:
    def __init__(self, result, **params):
        self.result = result
        self.params = params
        self.mean_accuracy, self.var_accuracy = self.check_accuracy()

    def check_accuracy(self):
        accuracy = torch.empty(self.params["max_iter_time"])
        for k in range(self.params["max_iter_time"]):
            accuracy[k] = torch.sum((self.result[k][0] - self.result[k][1]) ** 2) / (
                self.params["grid_size"] ** 2
            )
        return torch.mean(accuracy), torch.var(accuracy)

    # add some more metrics to this class
    def more_metrics_go_here(self):
        pass


def single_hm(
    u_k,
    k: int,
    max_iter_time: int,
    plot_target: bool,
    plot_solution: bool,
    plot_variance: bool,
):
    # Clear the current plot figure
    plt.clf()
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    if plot_target:
        plt.title(f"Temperature at t = {k/max_iter_time:.3f} unit time, Target")
        plt.pcolormesh(u_k[0], cmap=plt.cm.jet, vmin=0, vmax=1)
    elif plot_solution:
        plt.title(
            f"Temperature at t = {k/max_iter_time:.3f} unit time, Prediction Mean"
        )
        plt.pcolormesh(u_k[1], cmap=plt.cm.jet, vmin=0, vmax=1)
    elif plot_variance:
        plt.title(f"Temperature at t = {k/max_iter_time:.3f} unit time, Prediction Var")
        plt.pcolormesh(u_k[2], cmap=plt.cm.jet, vmin=0, vmax=1)
    else:
        raise ValueError

    plt.colorbar()
    # plt.show()
    return plt


def plotheatmap(result_k, max_iter_time, fig=None, axes=None, ani=False):
    """
    if ani==True, need to input fig and axes
    if ani==False, create fig, axes within function
    """

    k = 0

    # Clear the current plot figure
    plt.clf()

    if not ani:
        fig, axes = plt.subplots(ncols=3, figsize=(15, 4))

    ax1, ax2, ax3 = axes

    # plt.title(f"Temperature at t = {k * delta_t:.3f} unit time")
    ax1.set_title(f"Target at t = {k / max_iter_time:.3f} unit time")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2.set_title(f"Predicted Mean at t = {k / max_iter_time:.3f} unit time")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax3.set_title(f"Predicted Var at t = {k / max_iter_time:.3f} unit time")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

    # This is to plot u_k (u at time-step k)
    im1 = ax1.pcolormesh(result_k[0], cmap=plt.cm.jet, vmin=0, vmax=1)
    im2 = ax2.pcolormesh(result_k[1], cmap=plt.cm.jet, vmin=0, vmax=1)
    im3 = ax3.pcolormesh(result_k[2], cmap=plt.cm.jet, vmin=0, vmax=1)
    fig.colorbar(im1, ax=ax3)

    # fig.colorbar()
    # ax2.colorbar()
    # ax3.colorbar()
    plt.show()
    return im1, im2, im3
