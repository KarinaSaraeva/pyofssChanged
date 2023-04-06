
"""
    Copyright (C) 2011, 2012  David Bolt, 2019-2020 Vlad Efremov, Denis Kharenko

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

try:
    from pyfftw import byte_align
except ImportError:
    def byte_align(v):
        return v
    pass


import numpy as np
import pandas as pd
import os
import scipy.integrate as integrate

from .domain import Domain
from .modules.fibre import Fibre
from .modules.splitter import Splitter
from .modules.storage import check_dir
from .field import energy, spectrum_width_params
from scipy.signal import find_peaks

def field_save(field, filename='field_out'):
    np.savez_compressed(filename, field=field)


def field_load(filename='field_out'):
    if filename.endswith(".npz"):
        d = np.load(filename)['field']
    elif filename.endswith(".npy"):
        d = np.load(filename)
    else:
        try:
            d = np.load(filename + '.npz')['field']
        except:
            d = np.load(filename + '.npy')
    return d


class System(object):
    """
    :param object domain: A domain to be used with contained modules
    :param array_like field: Start field array if not use modules for generate
    start field

    A system consists of a list of modules, each of which may be called with a
    domain and field as parameters. The result of each module call is stored
    in a dictionary.
    """

    def __init__(self, domain=Domain(), field=None, charact_dir=None):
        self.domain = domain
        self.field = None
        self.fields = None
        self.modules = None
        self.clear(remove_modules=True)
        self.df_temp = None
        self.df_spec = None
        self.df_complex = None

        if field is not None:
            self.field = field

        if charact_dir is not None:
            check_dir(charact_dir)
            self.charact_dir = charact_dir

    def clear(self, remove_modules=False):
        """
        Clear contents of all fields.
        Clear (remove) all modules if requested.
        """
        if(self.domain.channels > 1):
            self.field = [np.zeros([self.domain.total_samples], complex)
                          for channel in range(self.domain.channels)]
        else:
            self.field = np.zeros([self.domain.total_samples], complex)

        self.fields = {}

        if(remove_modules):
            self.modules = []

    def add(self, module):
        """ Append a module to the system. """
        self.modules.append(module)

    def __getitem__(self, module_name):
        for index, module in enumerate(self.modules):
            if(module.name == module_name):
                return self.modules[index]

    def __setitem__(self, module_name, new_module):
        for index, module in enumerate(self.modules):
            if(module.name == module_name):
                self.modules[index] = new_module
                return

        raise Exception("Tried to modify non-existing module in system")

    def init_df(self, is_temporal=True, save_power=True):
        df = pd.DataFrame()
        z_curr = 0
        for obj in self.modules:
            if type(obj) is Fibre:
                df_new = obj.stepper.storage.get_df(
                    z_curr=z_curr, is_temporal=is_temporal, save_power=save_power)
                # concatenate dataframes that were received from different fibres if not empty
                if not df_new.empty:
                    z_curr = df_new.iloc[-1].name[-1]
                    df = pd.concat([df, df_new])
        if save_power:
            if is_temporal:
                self.df_temp = df
            else:
                self.df_spec = df
        else:
            self.df_complex = df

    # save the whole laser dataframe only if needed
    def save_df_to_csv(self, dir, name="laser", is_temporal=True, save_power=True):
        check_dir(dir)
        if is_temporal:
            if self.df_temp is None:
                self.init_df(is_temporal=is_temporal, save_power=save_power)
            file_name = os.path.join(dir, name + "_temp.csv")
            with open(file_name, "w") as f:
                self.df_temp.to_csv(file_name)
        else:
            if self.df_spec is None:
                self.init_df(is_temporal=is_temporal, save_power=save_power)
            file_name = os.path.join(dir, name + "_spec.csv")
            with open(file_name, "w") as f:
                self.df_spec.to_csv(file_name)

    def get_last_cycle_df(self, df_laser):
        cycle_names = list(
            set(df_laser.index.get_level_values('cycle').values))
        cycle_names.sort()
        return df_laser.loc[(cycle_names[-1])]

    def save_last_cycle_df_to_csv(self, dir, name="last_cycle", save_represent="complex"):
        check_dir(dir)
        if (save_represent == "both"):
            if self.df_temp is None:
                self.init_df(is_temporal=True, save_power=True)
            file_name = os.path.join(dir, name + "_temp.csv")
            with open(file_name, "w") as f:
                self.get_last_cycle_df(self.df_temp).to_csv(file_name)

            if self.df_spec is None:
                self.init_df(is_temporal=False, save_power=True)
            file_name = os.path.join(dir, name + "_spec.csv")
            with open(file_name, "w") as f:
                self.get_last_cycle_df(self.df_spec).to_csv(file_name)
        elif (save_represent == "temporal"):
            if self.df_temp is None:
                self.init_df(is_temporal=True, save_power=True)
            file_name = os.path.join(dir, name + "_temp.csv")
            with open(file_name, "w") as f:
                self.get_last_cycle_df(self.df_temp).to_csv(file_name)
        elif (save_represent == "spectral"):
            if self.df_spec is None:
                self.init_df(is_temporal=False, save_power=True)
            file_name = os.path.join(dir, name + "_spec.csv")
            with open(file_name, "w") as f:
                self.get_last_cycle_df(self.df_spec).to_csv(file_name)
        elif (save_represent == "complex"):
            if self.df_complex is None:
                self.init_df(is_temporal=True, save_power=False)
            file_name = os.path.join(dir, name + "_complex.csv")
            with open(file_name, "w") as f:
                self.get_last_cycle_df(self.df_complex).to_csv(file_name)

    def get_laser_info(self):
        def get_duration_spec(P, d_x):
            heigth_fwhm, fwhm, left_ind, right_ind = spectrum_width_params(
                P, prominence=(np.amax(P)/10))
            return abs(fwhm)*d_x if fwhm is not None else None
        def get_peaks(P):
            peaks, _ = find_peaks(P, height=0, prominence=(np.amax(P)/10))
            return peaks
        
        if self.df_temp is None: self.init_df(is_temporal=True, save_power=True)
        if self.df_spec is None: self.init_df(is_temporal=False, save_power=True)
            
        characteristic = ["max_value", "energy", "duration", "spec_width", "peaks"]
        iterables = [characteristic]
        df_results = pd.DataFrame(index=self.df_temp.index, columns=characteristic)
        df_results['max_value'] = self.df_temp.max(axis=1).values
        df_results['energy'] = self.df_temp.apply(
            lambda row: integrate.simps(row, self.domain.t*1e-3), axis=1).values

        df_results['duration'] = self.df_temp.apply(
            lambda row: get_duration_spec(row, self.domain.dt), axis=1).values
        df_results['spec_width'] = self.df_spec.apply(
            lambda row: get_duration_spec(row, self.domain.dnu), axis=1).values
        df_results['peaks'] = self.df_temp.apply(
            lambda row: get_peaks(row), axis=1)
        
        return df_results
            
    def update_charact_file(self):
        if self.charact_dir is not None:
            df_results = self.get_laser_info()
            df_results.to_csv(os.path.join(self.charact_dir, "laser_info.csv"))

    def run(self):
        """
        Propagate field through each module, with the resulting field at the
        exit of each module stored in a dictionary, with module name as key.
        """
        self.field = byte_align(self.field)
        for module in self.modules:
            self.field = module(self.domain, self.field)
            self.fields[module.name] = self.field
            if type(module) is Splitter:
                self.update_charact_file()
