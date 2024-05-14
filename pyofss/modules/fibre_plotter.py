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
    

def visualise_fields_df(fields_df, y_arr, y_label="", y_lims=None, auto_lims=False, cbar_label="", figname=None, title=None, max_value=None, min_value=None):
    cycle_names = list(set(fields_df.index.get_level_values('cycle').values))
    cycle_names.sort()
    cycle_df = fields_df.loc[cycle_names[0]]
    fibre_names = list(set(cycle_df.index.get_level_values('fibre').values))
    fibre_names.sort()

    if max_value is None:
        max_value = fields_df.values.max()
    if max_value is None:
        min_value = fields_df.values.min()

    nrows = int(np.ceil(len(cycle_names) / 2))
    ncols = min(len(fibre_names), 2)

    fig, ax = plt.subplots(nrows=len(cycle_names), ncols=len(fibre_names), figsize=(ncols*10,nrows*5))

    if (type(ax) is not np.ndarray):
        ax = np.array([[ax]])
        print(f"{ax.shape}")
    elif len(ax.shape) == 1:
        ax = ax.reshape(1, -1)
        print(f"{ax.shape}")

    cycle_names = cycle_names
    for i, cycle_name in enumerate(cycle_names):
        cycle_df = fields_df.loc[cycle_name]
        fibre_names = list(set(cycle_df.index.get_level_values('fibre').values))
        fibre_names.sort()
        for j, fibre_name in enumerate(fibre_names):
            fibre_df = fields_df.loc[cycle_name].loc[fibre_name]
            z = fibre_df.index.get_level_values('z [mm]').values
            h = fibre_df.values.transpose()

            if auto_lims:
                _, _, left_idx_start, right_idx_start = spectrum_width_params(h[:, 0], h[:, 0].max()/10)
                _, _, left_idx_end, right_idx_end = spectrum_width_params(h[:, -1], h[:, -1].max()/10)

                left_idx = int(min(left_idx_start, left_idx_end) - 100)
                right_idx = int(max(right_idx_start, right_idx_end) + 100)

                # left_lim_start, right_lim_start = y_arr[[int(left_idx), int(right_idx)]]
                # ax[i, j].set_ylim(left_lim_start, right_lim_start)
                
                X, Y = np.meshgrid(z, y_arr[left_idx:right_idx])
                h = fibre_df.values[:, left_idx:right_idx].transpose()

            elif y_lims is not None:
                # ax[i, j].set_ylim(*y_lims)
                indices = np.where((y_arr > y_lims[0]) & (y_arr < y_lims[1]))
                left_idx = np.min(indices)
                right_idx = np.max(indices)
                X, Y = np.meshgrid(z, y_arr[left_idx:right_idx])
                h = fibre_df.values[:, left_idx:right_idx].transpose()
            else:
                X, Y = np.meshgrid(z, y_arr)
            
            cf = ax[i, j].pcolormesh(X, Y, h, shading='auto', cmap=plt.cm.get_cmap('plasma'), vmin=min_value, vmax=max_value)
            
            ax[i, j].set_ylabel(y_label)
            ax[i, j].set_xlabel('z [mm]')
            if (len(cycle_names) > 1):
                ax[i, j].set_title(f"{cycle_name}, {fibre_name}")
            else:
                ax[i, j].set_title(f"{fibre_name}")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.ax.set_xlabel(cbar_label, labelpad=15) 

    if title is not None:
        fig.suptitle(title)

    if figname is not None:
        plt.savefig(figname)

    return fig

def visualise_results_df(df_results, figname=None, title=None):
    len_characts = len(df_results.columns)

    nrows = int(np.ceil(len_characts / 2))
    ncols = min(len_characts, 2)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*10,nrows*5))

    z_arr = df_results.index.get_level_values("z [mm]").values

    for i, col_name in enumerate(df_results.columns):
        row = i // ncols
        col = i % ncols

        if col_name != "Peaks [idx]":
            axs[row, col].plot(z_arr, df_results[col_name])
            axs[row, col].set_xlabel(f"z [mm]")
            axs[row, col].set_ylabel(f"{col_name}")
        else:
            peak_num_arr = df_results.loc[:, [col_name]].apply(len, axis=1).values
            axs[row, col].plot(z_arr, peak_num_arr)
            axs[row, col].set_xlabel(f"z [mm]")
            axs[row, col].set_ylabel(f"{col_name}")

    if title is not None:
        fig.suptitle(title)

    if figname is not None:
        plt.savefig(figname)

    return fig
    
