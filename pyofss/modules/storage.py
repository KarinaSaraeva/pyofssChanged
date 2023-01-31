"""
    Copyright (C) 2012  David Bolt

    This file is part of pyofss.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from re import X
import numpy as np
import os
import warnings

from pyofss import field
from pyofss.field import temporal_power, spectral_power

import pandas as pd


class StorageError(Exception):
    pass


class DirExistenceError(StorageError):
    pass


class DifferentAxisError(StorageError):
    pass


def reduce_to_range(x, ys, first_value, last_value):
    """
    :param array_like x: Array of x values to search
    :param array_like ys: Array of y values corresponding to x array
    :param first_value: Initial value of required range
    :param last_value: Final value of required range
    :return: Reduced x and y arrays
    :rtype: array_like, array_like

    From a range given by first_value and last_value, attempt to reduce
    x array to required range while also reducing corresponding y array.
    """
    print("Attempting to reduce storage arrays to specified range...")
    if last_value > first_value:

        def find_nearest(array, value):
            """ Return the index and value closest to those provided. """
            index = (np.abs(array - value)).argmin()
            return index, array[index]

        first_index, x_first = find_nearest(x, first_value)
        last_index, x_last = find_nearest(x, last_value)

        print(
            "Required range: [{0}, {1}]\nActual range: [{2}, {3}]".format(
                first_value, last_value, x_first, x_last
            )
        )

        # The returned slice does NOT include the second index parameter. To
        # include the element corresponding to last_index, the second index
        # parameter should be last_index + 1:
        sliced_x = x[first_index: last_index + 1]

        # ys is a list of arrays. Does each array contain additional arrays:
        import collections

        if isinstance(ys[0][0], collections.Iterable):
            sliced_ys = [
                [
                    y_c0[first_index: last_index + 1],
                    y_c1[first_index: last_index + 1],
                ]
                for (y_c0, y_c1) in ys
            ]
        else:
            sliced_ys = [y[first_index: last_index + 1] for y in ys]

        return sliced_x, sliced_ys
    else:
        print("Cannot reduce storage arrays unless last_value > first_value")


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("Directory: ", dir_name, " is created!")
    else:
        if not os.listdir(dir_name):
            print("Directory is empty")
        else:
            warnings.warn("Directory is not empty")


def get_value_from_str(x):
    if x == "None":
        return None
    else:
        return float(x)


class Storage(object):
    """
    Contains A arrays for multiple z values. Also contains t array and
    functions to modify the stored data.
    """

    def __init__(self, dir=None, cycle=None, fibre_name=None, **file_import_arguments):
        if dir is not None:
            self.dir_temp = os.path.join(dir, "temp")
            self.dir_spec = os.path.join(dir, "spec")
            check_dir(self.dir_temp)
            check_dir(self.dir_spec)
        else:
            self.dir_temp = None
            self.dir_spec = None

        self.cycle = cycle
        self.fibre_name = fibre_name
        self.plt_data = None
        self.t = []
        self.As = []
        self.z = []

        self.nu = []

        self.file_import_arguments = file_import_arguments

        # List of tuples of the form (z, h); one tuple per successful step:
        self.step_sizes = []

        # Accumulate number of fft and ifft operations used for a stepper run:
        self.fft_total = 0

    def reset_array(self):
        self.As = []
        self.z = []

    @staticmethod
    def reset_fft_counter():
        """ Resets the global variable located in the field module. """
        field.fft_counter = 0

    def store_current_fft_count(self):
        """ Store current value of the global variable in the field module. """
        self.fft_total = field.fft_counter

    def append(self, z, A):
        """
        :param double z: Distance along fibre
        :param array_like A: Field at distance z

        Append current fibre distance and field to stored array
        """
        self.z.append(z)
        self.As.append(A)

    def save_all_storage_to_dir(
        self,
        is_temporal=True,
        channel=None,
        **file_import_arguments,
    ):
        for i in range(len(self.As)):
            self.save_step_to_file(is_temporal=is_temporal,
                                   channel=channel,
                                   i=i,
                                   **file_import_arguments,
                                   )

    def save_all_storage_to_dir_as_df(
        self,
        is_temporal=True,
        channel=None,
    ):
        if self.dir_temp and self.dir_spec is not None:
            if is_temporal:
                dir = self.dir_temp
            else:
                dir = self.dir_spec

            (x, y, z) = self.get_plot_data(is_temporal=is_temporal)
            index = pd.MultiIndex.from_product([z], names=["z [mm]"])
            df1 = pd.DataFrame(y, index=index)
            file_name = os.path.join(dir, f"{self.fibre_name}.csv")
            with open(file_name, "w") as f:
                df1.to_csv(file_name)

    def get_df(
        self,
        is_temporal=True,
        z_curr=0,
        channel=None,
    ):
        (x, y, z) = self.get_plot_data(is_temporal=is_temporal)
        arr_z = np.array(z)*10**6 + z_curr
        if self.cycle and self.fibre_name is not None:
            iterables = [[self.cycle], [self.fibre_name], arr_z]
            index = pd.MultiIndex.from_product(
                iterables,  names=["cycle", "fibre", "z [mm]"])
        else:
            iterables = [z*10**6]
            index = pd.MultiIndex.from_product(iterables, names=["z [mm]"])
        return pd.DataFrame(y, index=index)

    def save_step_to_file(self, is_temporal=True, channel=None, i=-1, **file_import_arguments):
        if self.dir_temp and self.dir_spec is not None:
            if is_temporal:
                x = self.t
                x_label = "t"
                calculate_power = temporal_power
                dir = self.dir_temp
            else:
                x = self.nu
                x_label = "nu"
                calculate_power = spectral_power
                dir = self.dir_spec

            if channel is None:
                temp = self.As[i]
                y = temp
                df = pd.DataFrame(np.column_stack(
                    [x, calculate_power(y)]), columns=[x_label, "P"])
                file_name = os.path.join(
                    dir,
                    f"z{i if i!=-1 else len(self.z)}_{self.z[i]*1e6:.2f}mm.csv",
                )
                with open(file_name, "w") as f:
                    if len(file_import_arguments) != 0:
                        self.file_import_arguments = file_import_arguments
                    for arg in self.file_import_arguments.items():
                        f.write(f"# {arg[0]}={arg[1]} \n")
                    f.write(f"# z_current={self.z[i]} \n")
                    df.to_csv(f, sep=" ")

    def set_file_export_arguments(self, commentlines):
        for line in commentlines:
            if "alpha=" in line:
                first, remainder = line.split("alpha=")
                alpha = get_value_from_str(remainder.split()[0])
            elif "beta2=" in line:
                first, remainder = line.split("beta2=")
                beta2 = get_value_from_str(remainder.split()[0])
            elif "beta3=" in line:
                first, remainder = line.split("beta3=")
                beta3 = get_value_from_str(remainder.split()[0])
            elif "gamma=" in line:
                first, remainder = line.split("gamma=")
                gamma = get_value_from_str(remainder.split()[0])
            elif "small_signal_gain=" in line:
                first, remainder = line.split("small_signal_gain=")
                small_signal_gain = get_value_from_str(remainder.split()[0])
            elif "E_sat=" in line:
                first, remainder = line.split("E_sat=")
                E_sat = get_value_from_str(remainder.split()[0])
            elif "length=" in line:
                first, remainder = line.split("length=")
                length = get_value_from_str(remainder.split()[0])
            elif "z_current=" in line:
                first, remainder = line.split("z_current=")
                z_current = get_value_from_str(remainder.split()[0])

        self.file_export_arguments = {
            "alpha": alpha,
            "beta": [0, 0, beta2, beta3],
            "gamma": gamma,
            "small_signal_gain": small_signal_gain,
            "E_sat": E_sat,
        }

        return z_current

    def read_all_from_dir(self, is_temporal=True,):
        if is_temporal:
            dir = self.dir_temp
        else:
            dir = self.dir_spec

        z = np.zeros(len(os.listdir(dir)))
        y = None
        x = None

        list = os.listdir(dir)

        def sorting_func(name):
            name = name.replace("z", "")
            sub = name.split("_", 1)[1]
            name = name.replace(sub, "")
            name = name.replace("_", "")
            return int(name)

        list.sort(key=sorting_func)

        for i in range(len(list)):
            commentlines = []
            filename = os.path.join(dir, list[i])
            with open(filename) as f:
                for line in f:
                    if line.startswith("#"):
                        commentlines.append(line)

                z_current = self.set_file_export_arguments(commentlines)
                df = pd.read_csv(filename, index_col=0, comment="#", sep=" ")
                x_current = df[df.columns[0]].values
                y_current = df[df.columns[1]].values
                if y is None:
                    y = np.zeros((len(z), len(x_current)))
                    x_previos = x_current
                    x = x_current
                if not np.array_equal(x_previos, x_current):
                    raise DifferentAxisError(
                        "the x-axis is different when reading files, you can not get data for plotting"
                    )

                z[i] = z_current
                y[i] = y_current

        self.z = z
        self.plt_data = (x, y, z)

        return self.plt_data

    def get_plot_data(
        self,
        is_temporal=True,
        reduced_range=None,
        normalised=False,
        channel=None,
    ):
        """
        :param bool is_temporal: Use temporal domain data (else spectral
                                 domain)
        :param Dvector reduced_range: Reduced x_range. Reduces y array to
                                      match.
        :param bool normalised: Normalise y array to first value.
        :param Uint channel: Channel number if using WDM simulation.
        :return: Data for x, y, and z axis
        :rtype: Tuple

        Generate data suitable for plotting. Includes temporal/spectral axis,
        temporal/spectral power array, and array of z values for the x,y data.
        """
        if is_temporal:
            x = self.t
            calculate_power = temporal_power
        else:
            x = self.nu
            calculate_power = spectral_power

        if channel is not None:
            temp = [calculate_power(A[channel]) for A in self.As]
        else:
            temp = [calculate_power(A) for A in self.As]

        if normalised:
            factor = max(temp[0])
            y = [t / factor for t in temp]
        else:
            y = temp

        if reduced_range is not None:
            x, y = reduce_to_range(x, y, reduced_range[0], reduced_range[1])

        z = self.z
        self.plt_data = (x, y, z)
        return (x, y, z)

    @staticmethod
    def find_nearest(array, value):
        """
        :param array_like array: Array in which to locate value
        :param double value: Value to locate within array
        :return: Index and element of array corresponding to value
        :rtype: Uint, double
        """
        index = (np.abs(array - value)).argmin()
        return index, array[index]

    def interpolate_As_for_z_values(self, zs):
        """
        :param array_like zs: Array of z values for which A is required

        Split into separate arrays, interpolate each, then re-join.
        """
        import collections

        # Check if As[0] is itself a list of iterable elements, e.g.
        # As[0] = [ [A_0, B_0], [A_1, B_1], ... , [A_N-1, B_N-1] ]
        # rather than just a list of (non-iterable) elements, e.g.
        # As[0] = [ A_0, A_1, ... , A_N-1 ]
        if isinstance(self.As[0][0], collections.Iterable):
            # Separate into channel_0 As and channel_1 As:
            As_c0, As_c1 = list(zip(*self.As))

            As_c0 = self.interpolate_As(zs, As_c0)
            As_c1 = self.interpolate_As(zs, As_c1)

            # Interleave elements from both channels into a single array:
            self.As = list(zip(As_c0, As_c1))
        else:
            self.As = self.interpolate_As(zs, self.As)

        # Finished using original z; can now overwrite with new values (zs):
        self.z = zs

    def interpolate_As(self, zs, As):
        """
        :param array_like zs: z values to find interpolated A
        :param array_like As: Array of As to be interpolated
        :return: Interpolated As
        :rtype: array_like

        Interpolate array of A values, stored at non-uniform z-values, over a
        uniform array of new z-values (zs).
        """
        from scipy import interpolate

        IUS = interpolate.InterpolatedUnivariateSpline

        As = np.array(As)

        if As[0].dtype.name.startswith("complex"):
            # If using complex data, require separate interpolation functions
            # for the real and imaginary parts. This is due to the used
            # routine being unable to process complex data type:
            functions = [
                (IUS(self.z, np.real(A)), IUS(self.z, np.imag(A)))
                for A in As.transpose()
            ]
            As = np.vstack(
                np.array(f(zs) + 1j * g(zs)) for f, g in functions
            ).transpose()
        else:
            # Generate the interpolation functions for each column in As. This
            # is achieved by first transposing As, then calculating the
            # interpolation function for each row.
            functions = [IUS(self.z, A) for A in As.transpose()]
            # Apply the functions to the new z array (zs), stacking together
            # the resulting arrays into columns. Transpose the array of
            # columns to recover the final As:
            As = np.vstack(f(zs) for f in functions).transpose()

        return As
