"""
    Copyright (C) 2024 Denis Kharenko, Karina Saraeva

    This file is part of pyofss-2.

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

import numpy as np
import math
import pandas as pd
import os
import scipy.integrate as integrate
from scipy.interpolate import interpn

import os
import warnings

from pyofss import field
from pyofss.field import temporal_power, spectral_power
from pyofss.field import energy, get_duration, get_bandwidth, get_peaks
try:
    from .opencl_fibre import OpenclFibre
except ImportError:
    OpenclFibre = None

from .fibre import Fibre
from ..field import energy, spectrum_width_params
from scipy.signal import find_peaks

from scipy.signal import find_peaks

import pandas as pd


class CollectorError(Exception):
    pass


class DirExistenceError(CollectorError):
    pass


class DifferentAxisError(CollectorError):
    pass


class SavingWarning(Warning):
    pass


class InvalidArgumentError(CollectorError):
    """Raised when the type argument is not valid"""
    pass


try:
    import pyopencl.array as pycl_array
except ImportError:
    pycl_array = None
    print("OpenCL is not activated, Collector will work with NumPy arrays only")


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


class Collector(object):
    """
    :param string name: Name of this module
    :param object system: The system to get modules from
    :param array_like modules: A list of module names to save data from
    :param string charact_dir: Directory to save data
    :param string save_represent: Type of the data to be saved
    :param int downsampling: the number of times by which you need to reduce the data

    Collector can be added to the system to collect data for multiple modules.
    """

    def __init__(self, name="collector", system=None, module_names=(),
                 charact_dir=None, save_represent="complex", downsampled=None,
                 save_evolution=True, save_fields=False):

        self.name = name
        self.module_names = module_names
        self.system = system
        if self.system is None:
            raise InvalidArgumentError("System argument should be not None")

        if charact_dir is not None:
            self.charact_dir = charact_dir
            check_dir(self.charact_dir)
        else:
            self.charact_dir = None

        self.cycle = 0
        self.downsampled = downsampled
        self.save_evolution = save_evolution
        self.save_fields = save_fields

        self.domain = None

        self.energy_list = []
        self.max_power_list = []
        self.peaks_list = []
        self.duration_list = []
        self.bandwidth_list = []
        self.l_nl_list = []
        self.l_d_list = []

        self.df_results = None
        self.df_fields = None
        df_temp, df_spec, df_complex = None, None, None
        self.df_type_dict = {"temp": df_temp, "spec": df_spec, "complex": df_complex}
        self.z_curr = 0

    def __call__(self, domain, field):
        for name in self.module_names:
            if isinstance(self.system[name], Collector):
                print("Collector appears in the list to save, skipped...")
                continue  # skip myself

            self.domain = domain
            self.append_result_to_df(self.system[name])

            iterables = [[self.cycle], [name]]
            index = pd.MultiIndex.from_product(iterables, names=["cycle", "fibre"])
            self.update_fields_df(pd.DataFrame([self.system.fields[name]], index=index))
            if self.charact_dir is not None and self.save_fields:
                self.save_fields_df_to_csv()

        # INFO дублируется точка конца прошлого элемента и начала следующего

        if self.charact_dir and self.save_evolution:
            print("New characts are saved")
            self.save_result_df_to_csv()

        self.cycle += 1
        return field

    @property
    def df_temp(self):
        if self.df_type_dict["temp"] is not None:
            return self.df_type_dict["temp"]
        else:
            self.init_df("temp")
            return self.df_type_dict["temp"]

    @property
    def df_spec(self):
        if self.df_type_dict["spec"] is not None:
            return self.df_type_dict["spec"]
        else:
            self.init_df("spec")
            return self.df_type_dict["spec"]

    @property
    def df_complex(self):
        if self.df_type_dict["complex"] is not None:
            return self.df_type_dict["complex"]
        else:
            self.init_df("complex")
            return self.df_type_dict["complex"]

    # must be called only for already calculated system
    def init_df(self, df_type="complex"):
        df = pd.DataFrame()
        z_curr = 0

        for name in self.module_names:
            obj = self.system[name]
            if type(obj) is Fibre:
                df_new = obj.stepper.storage.get_df(type=df_type, z_curr=z_curr)
                # concatenate dataframes that were received from different fibres if not empty
                if not df_new.empty:
                    z_curr = df_new.iloc[-1].name[-1]
                    df = pd.concat([df, df_new])
            if type(obj) is OpenclFibre:
                df_new = obj.get_df(type=df_type, z_curr=z_curr)
                # concatenate dataframes that were received from different fibres if not empty
                if not df_new.empty:
                    z_curr = df_new.iloc[-1].name[-1]
                    df = pd.concat([df, df_new])
        if df_type in self.df_type_dict.keys():
            self.df_type_dict[df_type] = df
        else:
            raise ValueError("df_type should be one of the list [complex, temp, spec]")

    # save the whole laser dataframe only if needed
    def save_df_to_csv(self, df_type="complex", name="laser"):
        check_dir(self.charact_dir)
        if df_type in self.df_type_dict.keys():
            if self.df_type_dict[df_type] is None:
                self.init_df(df_type=df_type)
            file_name = os.path.join(self.charact_dir, name + "_" + df_type + ".csv")
            with open(file_name, "w") as f:
                self.df_type_dict[df_type].to_csv(f, float_format='%.4g')
        else:
            raise ValueError("df_type should be one of the list [complex, temp, spec]")

    def append_result_to_df(self, obj):
        """Get results from a storage and concatenate with existing DataFrame"""
        if type(obj) is Fibre:
            df_new_results = self.get_df_result_from_storage(obj.stepper.storage, obj, self.z_curr)
        elif type(obj) is OpenclFibre:
            df_new_results = self.get_df_result_from_storage(obj.storage, obj, self.z_curr)
        else:
            df_new_results = None

        if df_new_results is not None:
            self.z_curr = df_new_results.index.get_level_values("z [m]").values[-1]
            if self.df_results is None:
                self.df_results = df_new_results
            else:
                self.df_results = pd.concat([self.df_results, df_new_results])

    def save_result_df_to_csv(self):
        if self.df_results is not None:
            self.df_results.to_csv(os.path.join(self.charact_dir, "system_info.csv"), float_format='%.4g')

    def update_fields_df(self, df_field_new):
        if self.df_fields is None:
            self.df_fields = df_field_new
        else:
            self.df_fields = pd.concat([self.df_fields, df_field_new])

    def save_fields_df_to_csv(self):
        if self.df_fields is not None:
            self.df_fields.to_csv(os.path.join(self.charact_dir, "fields.csv"))

    def clear_characts_lists(self):
        self.energy_list = []
        self.max_power_list = []
        self.peaks_list = []
        self.duration_list = []
        self.bandwidth_list = []
        self.l_nl_list = []
        self.l_d_list = []

    def append_characts_to_lists(self, domain, A, obj):
        """"
        :param array_like A: Field at distance z

        updates the list of pulse characteristics appending new items to the characteristics lists
        """

        self.energy_list.append(energy(A, domain.t))
        temporal_power_arr = temporal_power(A)
        spectral_power_arr = spectral_power(A)
        self.max_power_list.append(np.amax(temporal_power_arr))
        self.peaks_list.append(get_peaks(temporal_power_arr))
        self.duration_list.append(get_duration(temporal_power_arr, domain.dt))
        self.bandwidth_list.append(get_bandwidth(spectral_power_arr, domain.dnu))
        self.l_d_list.append(obj.get_dispersion_length(A))
        self.l_nl_list.append(obj.get_nonlinear_length(A))

    def save_all_storage_to_dir_as_df(self, obj, path, save_power=True, channel=None):
        """
        :param boolean save_power: flag to save either the complex field as one dataframe or to save temporal and spectral intensity dataframes
        
        saves all field evolution along the fibre propagation 
        """
        if type(obj) is Fibre:
            storage = obj.stepper.storage
        elif type(obj) is OpenclFibre:
            storage = obj.storage
        else:
            raise InvalidArgumentError("Object '{obj.name}' must contain a storage")

        if save_power:
            dir_temp = os.path.join(path, "temp")
            dir_spec = os.path.join(path, "spec")
            check_dir(dir_temp)
            check_dir(dir_spec)

            df_temp = self.get_df(storage, "temp")
            file_name_temp = os.path.join(dir_temp, f"{obj.name}.csv")
            df_temp.to_csv(file_name_temp)

            df_spec = self.get_df(storage, "spec")
            file_name_spec = os.path.join(dir_spec, f"{obj.name}.csv")
            df_spec.to_csv(file_name_spec)
        else:
            dir_complex = os.path.join(path, "complex")
            check_dir(dir_complex)

            df_complex = self.get_df(storage, "complex")
            file_name_complex = os.path.join(dir_complex, f"{obj.name}.csv")
            df_complex.to_csv(file_name_complex)

        file_name_info = os.path.join(self.dir, f"current_info.txt")

        with open(file_name_info, 'w') as f:
            f.write(f"current cycle: {self.cycle}, current fibre: {obj.name}")

    def get_df(self, obj, type="complex", z_curr=0, channel=None):
        if not isinstance(obj, Storage):
            raise InvalidArgumentError(f"{obj} is not a valid argument, obj param must be a Storage")

        if type == "temp":
            x, y, z = obj.get_plot_data(is_temporal=True, downsampled=self.downsampled)
        elif type == "spec":
            x, y, z = odj.get_plot_data(is_temporal=False, downsampled=self.downsampled)
        elif type == "complex":
            y = obj.As
            z = np.array(obj.z)
        else:
            raise InvalidArgumentError(f"{type} is not a valid argument, type param can be 'temp', 'spec' or 'complex'")

        arr_z = z * 10**3 + z_curr  # meters
        if self.cycle and self.fibre_name is not None:
            iterables = [[self.cycle], [self.fibre_name], arr_z]
            index = pd.MultiIndex.from_product(
                iterables,  names=["cycle", "fibre", "z [m]"])
        else:
            iterables = [arr_z]
            index = pd.MultiIndex.from_product(iterables, names=["z [m]"])
        return pd.DataFrame(y, index=index)

    def get_df_result_from_storage(self, storage, obj=None, z_curr=0):
        z = np.array(storage.z)
        fibre_name = obj.name

        arr_z = z * 10**3 + z_curr  # meters
        characteristic = ["Peak Power [W]", "Energy [nJ]", "Temp width [ps]",
                          "Spec width [THz]", "Dispersion length [km]",
                          "Nonlinear length [km]", "Peaks [idx]"]
        iterables = [[self.cycle], [fibre_name], arr_z]
        index = pd.MultiIndex.from_product(
            iterables,  names=["cycle", "fibre", "z [m]"])

        self.clear_characts_lists()
        for i in range(len(z)):
            self.append_characts_to_lists(self.domain, storage.As[i], obj)

        df_results = pd.DataFrame(index=index, columns=characteristic)
        df_results["Peak Power [W]"] = self.max_power_list
        df_results["Energy [nJ]"] = self.energy_list
        df_results["Temp width [ps]"] = self.duration_list
        df_results["Spec width [THz]"] = self.bandwidth_list
        df_results["Peaks [idx]"] = self.peaks_list
        df_results["Dispersion length [km]"] = self.l_d_list
        df_results["Nonlinear length [km]"] = self.l_nl_list
        return df_results


if __name__ == "__main__":
    from pyofss import Domain, System, Gaussian, Fibre, Collector
    from pyofss import multi_plot, double_plot, labels

    import time
    import matplotlib.pyplot as plt

    domain = Domain(bit_width=200.0, total_bits=8, samples_per_bit=512 * 32)
    gaussian = Gaussian(peak_power=1.0, width=1.0)
    beta = [0.0, 0.0, 1.0, 1.0]

    # Prepare an initial field
    sysinit = System(domain)
    sysinit.add(gaussian)
    sysinit.run()

    # Make two roundtrips
    sys = System(domain, sysinit.field)
    sys.add(Fibre("f1", length=0.020, beta=beta, gamma=1.5, total_steps=20, traces=5))
    sys.add(Fibre("f2", length=0.020, beta=beta, gamma=1.5, total_steps=20, traces=5))
    sys.add(Collector(system=sys, module_names=("f1", "f2"), charact_dir="testdir", save_represent="complex"))

    start = time.time()
    sys.run()
    sys.run()
    stop = time.time()

    print(f"Run time: {stop-start}")
