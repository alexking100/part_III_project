import torch
from torch.utils.data import Dataset
import random
from visualise import *
from scipy.ndimage import median_filter, gaussian_filter

"""
TODO

get blobs in the middle to be heated
implement some kind of wind? 
implement a way of having a non-constant diffusion coefficient
implement a heat-conserving one (so it reflects off the sides?)
can we make creation of a new dataset any faster?

Run visualisations on them all (can run here too)
run the code here so no need to go to ipynb to debug
"""


class Initial_Conditions:
    """
    Make it easy to select a standard initial condition
    Can also select a random initial condition (this would be nice)
    """

    def __init__(self, max_iter_time: int, grid_size: int):
        self.grid_size = grid_size
        self.max_iter_time = max_iter_time
        self.top = 0.0
        self.left = 0.0
        self.right = 0.0
        self.bottom = 0.0

        # initialize grid is indexed [time, y, x]
        self.u = torch.full((max_iter_time, grid_size, grid_size), 0.0)

    def reset_conditions(self):
        self.top = 0.0
        self.left = 0.0
        self.right = 0.0
        self.bottom = 0.0
        self.u = torch.zeros_like(self.u)

    def initialise_u(self):
        # get initial conditions into grid
        self.u[:, (self.grid_size - 1) :, :] = self.top
        self.u[:, :, :1] = self.left
        self.u[:, :1, :] = self.bottom
        self.u[:, :, (self.grid_size - 1) :] = self.right

    def from_top(self, temp: float):
        # heat starts from top of grid
        self.reset_conditions()
        self.top = temp
        self.initialise_u()
        return self.u

    def from_left(self, temp: float):
        # heat starts from left side of grid
        self.reset_conditions()
        self.left = temp
        self.initialise_u()
        return self.u

    def from_right(self, temp: float):
        # heat starts from right side of grid
        self.reset_conditions()
        self.right = temp
        self.initialise_u()
        return self.u

    def from_bottom(self, temp: float):
        # heat starts from bottom of grid
        self.reset_conditions()
        self.bottom = temp
        self.initialise_u()
        return self.u

    def from_center(self, temp: float):
        """
        create block in center of grid with non-zero temp
        """
        self.reset_conditions()
        start_idx = int(self.grid_size / 4)
        stop_idx = int(3 * self.grid_size / 4)
        self.u[:, start_idx:stop_idx, start_idx:stop_idx] = temp
        return self.u

    def create_square(self, center: tuple, square: int, temp: float):
        assert len(center) == 2
        assert center[0] + square <= self.grid_size
        assert center[0] - square >= 0
        assert center[1] + square <= self.grid_size
        assert center[1] - square >= 0

        self.reset_conditions()
        x_start = center[0] - square
        x_end = center[0] + square
        y_start = center[1] - square
        y_end = center[1] + square

        self.u[:, x_start:x_end, y_start:y_end] = temp
        return self.u

    def random_square(self, square_range: tuple, temp_range: tuple):
        self.reset_conditions()
        assert square_range[1] > square_range[0]
        square = np.random.randint(square_range[0], square_range[1])
        allowed_center = (square, self.grid_size - square)
        center = (
            np.random.randint(allowed_center[0], allowed_center[1]),
            np.random.randint(allowed_center[0], allowed_center[1]),
        )
        assert temp_range[1] >= temp_range[0]
        temp = temp_range[0] + (temp_range[1] - temp_range[0]) * random.random()
        self.u = self.create_square(center=center, square=square, temp=temp)
        return self.u

    def random_edge(self, temp_range):
        assert temp_range[1] >= temp_range[0]
        temp = temp_range[0] + (temp_range[1] - temp_range[0]) * random.random()

        picker = np.random.randint(0, 4)
        if picker == 0:
            return self.from_top(temp=temp)
        elif picker == 1:
            return self.from_left(temp=temp)
        elif picker == 2:
            return self.from_right(temp=temp)
        elif picker == 3:
            return self.from_bottom(temp=temp)
        else:
            raise ValueError


class Diffusion_Data(Dataset):
    """
    Given some initial conditions, the dataset will return
    a scalar field of values over some grid
    """

    def __init__(
        self,
        num_samples: int,
        max_iter_time: int,
        grid_size: int,
        initial_conditions: Initial_Conditions,
        square_range: tuple,
        temp_range: tuple,
        diffusion_coef: float,
        sigma: int,
    ):
        # choice is [top, left, right, bottom]
        self.num_samples = num_samples
        self.square_range = square_range
        self.temp_range = temp_range
        self.max_iter_time = max_iter_time
        self.grid_size = grid_size
        self.initial_conditions = initial_conditions
        self.diffusion_coef = diffusion_coef
        self.data = []
        self.time_array = torch.linspace(0, 1, max_iter_time)
        self.dt = 1.0 / self.max_iter_time
        self.sigma = sigma
        self.pi_dim = 4

        for i in range(self.num_samples):
            self.run_neumann_boundary_conditions()

        # self.run_finite_difference()

    def get_privileged_info(self, out, k):
        # get privileged info
        max_temp = torch.max(out[k, :])
        avg_heat = torch.sum(out[k, :]) / (self.grid_size**2)
        avg_dilution = torch.count_nonzero(out[k, :]) / (self.grid_size**2)

        entropy = (
            out[k, :]
            / torch.sum(out[k, :])
            * torch.log(out[k, :] / torch.sum(out[k, :]))
        )
        entropy[np.isnan(entropy)] = 0
        avg_entropy = -torch.sum(entropy)

        return torch.Tensor([self.diffusion_coef, max_temp, avg_heat, avg_entropy])

    def neumann_step(self, u, dt, D, left_bc, right_bc, top_bc, bottom_bc):
        """
        Define the function to advance the solution in time
        """

        # Calculate the Laplacian using the five-point stencil
        laplacian = np.zeros_like(u)
        laplacian[1:-1, 1:-1] = (u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) + (
            u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]
        )

        # Apply the Neumann boundary conditions
        laplacian[:, 0] = (u[:, 1] - u[:, 0]) + left_bc(u, 0)
        laplacian[:, -1] = (u[:, -2] - u[:, -1]) + right_bc(u, 0)
        laplacian[0, :] = (u[1, :] - u[0, :]) + top_bc(u, 0)
        laplacian[-1, :] = (u[-2, :] - u[-1, :]) + bottom_bc(u, 0)

        # Update the solution using the forward Euler method
        u_new = u + dt * D * laplacian

        return u_new

    def run_neumann_boundary_conditions(self):
        out = torch.zeros((self.max_iter_time, self.grid_size * self.grid_size))
        pi = torch.zeros((self.max_iter_time, self.pi_dim))
        # re-randomise initial conditions
        self.u0 = self.initial_conditions.random_square(
            self.square_range, self.temp_range
        )[0, :, :]
        # apply gaussian filter so laplacian is stable
        self.u0 = torch.Tensor(gaussian_filter(self.u0, sigma=self.sigma))

        # Define the Neumann boundary conditions
        left_bc = (
            lambda u, t: u[:, 0] - u[:, 1]
        )  # normal derivative of the temperature at the left boundary is zero
        right_bc = (
            lambda u, t: u[:, -1] - u[:, -2]
        )  # normal derivative of the temperature at the right boundary is zero
        top_bc = (
            lambda u, t: u[0, :] - u[1, :]
        )  # normal derivative of the temperature at the top boundary is zero
        bottom_bc = (
            lambda u, t: u[-1, :] - u[-2, :]
        )  # normal derivative of the temperature at the bottom boundary is zero

        # Integrate the heat equation in time
        u = torch.clone(self.u0)
        for k in range(self.max_iter_time):
            u = torch.Tensor(
                self.neumann_step(
                    u,
                    self.dt,
                    self.diffusion_coef,
                    left_bc,
                    right_bc,
                    top_bc,
                    bottom_bc,
                )
            )
            out[k, :] = torch.flatten(u)
            pi[k, :] = self.get_privileged_info(out, k)

        self.data.append((self.time_array.unsqueeze(1), out, pi))

    def run_finite_difference(self):
        self.data = []

        for i in range(self.num_samples):
            # self.u = self.initial_conditions.random_edge(temp_range=temp_range)

            self.u = self.initial_conditions.random_square(
                square_range=self.square_range, temp_range=self.temp_range
            )
            self.u = torch.Tensor(gaussian_filter(self.u, sigma=self.sigma))
            self.solve_by_finite_elements()

    def solve_by_finite_elements(self):
        out = torch.zeros((self.max_iter_time, self.grid_size * self.grid_size))

        for k in range(self.max_iter_time - 1):
            laplacian = np.zeros_like(self.u[0])
            laplacian[1:-1, 1:-1] = (
                self.u[k, 1:-1, :-2] - 2 * self.u[k, 1:-1, 1:-1] + self.u[k, 1:-1, 2:]
            ) + (self.u[k, :-2, 1:-1] - 2 * self.u[k, 1:-1, 1:-1] + self.u[k, 2:, 1:-1])

            self.u[k + 1] = self.u[k] + self.dt * self.diffusion_coef * laplacian
            out[k, :] = torch.flatten(self.u[k, :, :])
            self.get_privileged_info(out, k)

        self.data.append((self.time_array.unsqueeze(1), out, self.pi))

        # last time value won't get filled in by this method
        # for k in range(0, self.max_iter_time - 1, 1):
        #     for i in range(1, self.grid_size - 1):
        #         for j in range(1, self.grid_size - 1):
        #             self.u[k + 1, i, j] = (
        #                 self.diffusion_coef
        #                 * (
        #                     self.u[k][i + 1][j]
        #                     + self.u[k][i - 1][j]
        #                     + self.u[k][i][j + 1]
        #                     + self.u[k][i][j - 1]
        #                     - 4 * self.u[k][i][j]
        #                 )
        #                 + self.u[k][i][j]
        #             )

        #     out[k, :] = torch.flatten(self.u[k, :, :])
        #     self.get_privileged_info(out, k)
        # self.data.append((self.time_array.unsqueeze(1), out, self.pi))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples

    def visualise_solution(self, show_until: int, ds_num=0):
        for k in range(show_until):
            u = self.data[ds_num][1][k, :].reshape(self.grid_size, self.grid_size)
            plt.imshow(u, cmap="hot", origin="lower", extent=[0, 10, 0, 10])
            plt.colorbar()
            plt.title("Heatmap t={:.2f}".format(self.time_array[k]))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()

        # for k in range(show_until):
        #     fig = single_hm(
        #         self.data[0][1][k, :].view(self.grid_size, self.grid_size).unsqueeze(0),
        #         k,
        #         max_iter_time=max_iter_time,
        #         plot_target=True,
        #         plot_solution=False,
        #         plot_variance=False,
        #     )
        #     fig.show()

    def save_tensors(self):
        # time series is same for all datasets
        torch.save(self.data[0][0], "./metadata/time_series")

        for i, ds in enumerate(self.data):
            torch.save(ds[1], "./metadata/heat_data_{}".format(i))
            torch.save(ds[2], "./metadata/pi_data_{}".format(i))

        # read these tensors back later and reproduce the dataset in the notebook


class RestoredData(Dataset):
    def __init__(
        self,
        num_samples: int,
        max_iter_time: int,
        grid_size: int,
        temp_range: tuple,
    ):
        self.num_samples = num_samples
        self.temp_range = temp_range
        self.max_iter_time = max_iter_time
        self.time_array = torch.linspace(0, 1, max_iter_time)
        self.grid_size = grid_size
        self.t = torch.load("./metadata/time_series")
        self.data = []

        for i in range(self.num_samples):
            y = torch.load("./metadata/heat_data_{}".format(i))
            pi = torch.load("./metadata/pi_data_{}".format(i))
            self.data.append((self.t, y, pi))

    def generate_noise(self):
        out = torch.rand((self.max_iter_time, self.grid_size * self.grid_size))
        return out

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


"""
Run code below when debugging, or to get graph of target data
"""
# max_iter_time = 100
# grid_size = 50
# num_samples = 1
# square_range = (5, 10)
# temp_range = (0.7, 1.0)
# diffusion_coef = 0.5

# initial_conditions = Initial_Conditions(
#     max_iter_time=max_iter_time, grid_size=grid_size
# )

# process_dataset = Diffusion_Data(
#     num_samples=num_samples,
#     max_iter_time=max_iter_time,
#     grid_size=grid_size,
#     initial_conditions=initial_conditions,
#     square_range=square_range,
#     temp_range=temp_range,
#     diffusion_coef=diffusion_coef,
# )

# process_dataset.visualise_solution(max_iter_time=max_iter_time, show_until=30)
