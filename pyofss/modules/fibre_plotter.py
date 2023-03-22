import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

from pyofss.field import max_peak_params, spectrum_width_params


class FibrePlotter(object):
    """
    Draws diffrent kind of graphs for x, y, z array
    param double array *Shape: (N)* x: spectral or temporal x axis 
    param double array *Shape: (MxN)* y: spectral or temporal power 
    param double array *Shape: (N)* z: z coordinates array
    """

    def __init__(self, x, y, z, xlabel="", ylabel="", zlabel=""):
        self.x = x
        self.y = y
        self.z = z

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

        self.font = {'family': 'serif',
                     'weight': 'normal',
                     'size': 8}

        matplotlib.rc('font', **self.font)

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
            frames=np.linspace(0, len(self.z)-1, len(self.z)),
            init_func=init,
            interval=interval,
            blit=True,
        )
        return anim

    def draw_heat_map(self, z_start=0, prominence=1e-20, is_temporal=True, subplot_spec=None, fig=None, title=None, vmin=None, vmax=None, width_param = 4, is_active=True):
        if (fig and subplot_spec is not None):
            inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                     subplot_spec=subplot_spec, wspace=0.1, hspace=0.1)
            axes = []
            for j in range(inner.ncols):
                ax = plt.subplot(inner[0, j])
                fig.add_subplot(ax)
                axes.append(ax)
        else:
            fig, axes = plt.subplots(1, 2)
            fig.tight_layout()

        if(title):
            axes[1].set_title(title)

        self.z += z_start
        P = self.y[-1] if is_active else self.y[0]
        d_t = abs(self.x[1] - self.x[0])

        if (is_temporal):
            heigth_fwhm, fwhm, left_ind, right_ind = max_peak_params(
                P, np.amax(P)/100)
        else:
            heigth_fwhm, fwhm, left_ind, right_ind = spectrum_width_params(
                P, np.amax(P)/100)

        axes[0].plot(self.x, P)
        # x_left, x_right = np.interp(
        #     [left_ind, right_ind], np.arange(len(self.x)), self.x)
        axes[0].hlines(heigth_fwhm, self.x[int(left_ind)],
                       self.x[int(right_ind)], color="C2")
        axes[0].set_xlabel(self.xlabel)
        axes[0].set_ylabel(self.ylabel)
        axes[0].set_title(f"Power profile at {self.z[-1]}km")

        left_ind = int(left_ind - width_param*d_t*fwhm)
        right_ind = int(right_ind + width_param*d_t*fwhm)
        X, Y = np.meshgrid(self.z, self.x[left_ind:right_ind])
        h = self.y[:, left_ind:right_ind]
        cf = axes[1].pcolormesh(X, Y, np.transpose(
            h), shading='auto', cmap=plt.cm.get_cmap('plasma'), vmin=vmin, vmax=vmax)
        fig.colorbar(cf, ax=axes[1])
        axes[1].set_xlabel(self.zlabel)
        axes[1].set_ylabel(self.xlabel)
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
