import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot as plt

from pyofss.field import max_peak_params


class FibrePlotter(object):
    """
    Draws diffrent kind of graphs for x, y, z array
    param double array *Shape: (N)* x: spectral or temporal x axis 
    param double array *Shape: (MxN)* y: spectral or temporal power 
    param double array *Shape: (N)* z: z coordinates array
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def draw_animation(self, interval=50):
        fig = plt.figure()
        ax = plt.axes()
        (line,) = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            ax.set_xlim(np.amin(self.x), np.amax(self.x))
            ax.set_ylim(0, np.amax(self.y))
            return (line,)

        def animate(i):
            y = self.y[int(i)]
            line.set_data(self.x, y)
            return (line,)

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=np.linspace(0, len(self.z), len(self.z)),
            init_func=init,
            interval=interval,
            blit=True,
        )
        return anim

    def draw_heat_map(self, z_start=0):
        fig, ax = plt.subplots(2, 1)
        self.z += z_start
        temp_power = self.y[-1]
        d_t = abs(self.x[1] - self.x[0])

        heigth_fwhm, fwhm, left_ind, right_ind = max_peak_params(temp_power)

        ax[0].plot(self.x, temp_power)
        x_left, x_right = np.interp(
            [left_ind, right_ind], np.linspace(0, len(self.x), len(self.x)), self.x)
        ax[0].hlines(heigth_fwhm, x_left, x_right, color="C2")

        left_ind = int(left_ind - 4*d_t*heigth_fwhm)
        right_ind = int(right_ind + 4*d_t*heigth_fwhm)

        X, Y = np.meshgrid(self.z, self.x[left_ind:right_ind])
        h = self.y[:, left_ind:right_ind]
        cf = ax[1].pcolormesh(X, Y, np.transpose(h))
        fig.colorbar(cf, ax=ax[1])
        z_start = self.z[-1]
        return z_start

    def draw_peak_power(self, z_start=0):
        fig, ax = plt.subplots(1, 1)
        power_peaks = np.zeros(len(self.z))
        for k in range(len(self.z)):
            power_peaks[k] = np.amax(self.y[k])
        self.z = self.z + z_start
        ax.plot(self.z, power_peaks)
        z_start = self.z[-1]
        return z_start
