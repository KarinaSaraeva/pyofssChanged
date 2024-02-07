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
from pyofss.field import temporal_power, spectral_power, energy, get_duration, get_duration_spec, get_peaks

from scipy.signal import find_peaks

import pandas as pd


class StorageError(Exception):
    pass


class DirExistenceError(StorageError):
    pass


class DifferentAxisError(StorageError):
    pass

class SavingWarning(Warning):
    pass

class InvalidArgumentError(StorageError):
    """Raised when the type argument is not valid"""
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

    def __init__(self, dir=None, cycle=None, fibre_name=None, f=None):
        if dir is not None:
            self.dir = dir
            check_dir(self.dir)
        else:
            self.dir = None

        self.f = f
        self.cycle = cycle
        self.fibre_name = fibre_name
        self.plt_data = None
        self.domain = None
        self.As = []
        self.z = []

        # List of tuples of the form (z, h); one tuple per successful step:
        self.step_sizes = []

        # Accumulate number of fft and ifft operations used for a stepper run:
        self.fft_total = 0

        self.energy_list = []
        self.max_power_list = []
        self.peaks_list = []
        self.duration_list = []
        self.spec_width_list = []
        self.l_nl_list = []
        self.l_d_list = []

    def __call__(self, domain):
        self.domain = domain

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
        self.update_characts(A)

    def update_characts(self, A):
        """" 
        :param array_like A: Field at distance z
        
        updates the list of pulse characteristics appending new items to the characteristics lists 
        """
        
        self.energy_list.append(energy(A, self.domain.t))
        temporal_power_arr = temporal_power(A)
        spectral_power_arr = spectral_power(A)
        self.max_power_list.append(np.amax(temporal_power_arr))
        self.peaks_list.append(get_peaks(temporal_power_arr))
        self.duration_list.append(get_duration(temporal_power_arr, self.domain.dt))
        self.spec_width_list.append(get_duration_spec(spectral_power_arr, self.domain.dnu))
        self.l_d_list.append(self.f.l_d(A))
        self.l_nl_list.append(self.f.l_nl(A))

    def save_all_storage_to_dir_as_df(
        self,
        save_power = True,
        channel=None,
    ):
        """
        :param boolean save_power: flag to save either the complex field as one dataframe or to save temporal and spectral intensity dataframes
        
        saves all field evolution along the fibre propagation 
        """
        if self.dir is not None:
            if save_power:
                dir_temp = os.path.join(dir, "temp")
                dir_spec = os.path.join(dir, "spec")
                check_dir(dir_temp)
                check_dir(dir_spec)

                df_temp = self.get_df("temp")
                file_name_temp = os.path.join(dir_temp, f"{self.fibre_name}.csv")
                df_temp.to_csv(file_name_temp)

                df_spec = self.get_df("spec")
                file_name_spec = os.path.join(dir_spec, f"{self.fibre_name}.csv")
                df_spec.to_csv(file_name_spec)
            else:
                dir_complex = os.path.join(dir, "complex")
                check_dir(dir_complex)
                
                df_complex = self.get_df("complex")
                file_name_complex = os.path.join(dir_complex, f"{self.fibre_name}.csv")
                df_complex.to_csv(file_name_complex)

            file_name_info = os.path.join(self.dir, f"current_info.txt")

            with open(file_name_info, 'w') as f:
                f.write(f"current cycle: {self.cycle}, current fibre: {self.fibre_name}")
        else:
            warnings.warn("Nothing will be saved - the base fibre directory is not stated!", SavingWarning)

    def get_df(self, type = "complex", z_curr=0, channel=None):
        if type == "temp":
            x, y, z = self.get_plot_data(is_temporal=True)
        elif type == "spec":
            x, y, z = self.get_plot_data(is_temporal=False)
        elif type == "complex":
            y = self.As
            z = self.z
        else:
            raise InvalidArgumentError(f"{type} is not a valid argument, type param can be 'temp', 'spec' or 'complex'")

        arr_z = np.array(z)*10**6 + z_curr # mm
        if self.cycle and self.fibre_name is not None:
            iterables = [[self.cycle], [self.fibre_name], arr_z]
            index = pd.MultiIndex.from_product(
                iterables,  names=["cycle", "fibre", "z [mm]"])
        else:
            iterables = [arr_z]
            index = pd.MultiIndex.from_product(iterables, names=["z [mm]"])
        return pd.DataFrame(y, index=index)
    
    def get_df_result(
        self,
        z_curr=0,
    ):
        z = self.z

        arr_z = np.array(z)*10**6 + z_curr
        characteristic = ["Peak Power [W]", "Energy [nJ]", "Temp width [ps]", "Spec width [THz]", "Dispersion length [km]", "Nonlinear length [km]", "Peaks [idx]"]
        if self.cycle and self.fibre_name is not None:
            iterables = [[self.cycle], [self.fibre_name], arr_z]
            index = pd.MultiIndex.from_product(
                iterables,  names=["cycle", "fibre", "z [mm]"])
            
        else:
            iterables = [arr_z]
            index = pd.MultiIndex.from_product(iterables, names=["z [mm]"])

        df_results = pd.DataFrame(index=index, columns=characteristic)
        df_results["Peak Power [W]"] = self.max_power_list
        df_results["Energy [nJ]"] = self.energy_list
        df_results["Temp width [ps]"] = self.duration_list
        df_results["Spec width [THz]"] = self.spec_width_list
        df_results["Peaks [idx]"] = self.peaks_list
        df_results["Dispersion length [km]"] = self.l_d_list
        df_results["Nonlinear length [km]"] = self.l_nl_list
        return df_results

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
            x = self.domain.t
            calculate_power = temporal_power
        else:
            x = self.domain.nu
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
