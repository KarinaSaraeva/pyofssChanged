
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
import psutil

from .domain import Domain
from .modules.fibre import Fibre
from .modules.opencl_fibre import OpenclFibre
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
        
        self.df_results = None
        df_temp, df_spec, df_complex = None, None, None
        self.df_type_dict = {"temp": df_temp, "spec": df_spec, "complex": df_complex}
        self.z_curr = 0

        if field is not None:
            self.field = field

        self.charact_dir =None

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

    # must be called only for already calculated laser
    def init_df(self, df_type="complex"):
        df = pd.DataFrame()
        z_curr = 0
        for obj in self.modules:
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
            raise ValueError()

    # save the whole laser dataframe only if needed
    def save_df_to_csv(self, dir, df_type="complex", name="laser"):
        check_dir(dir)
        if df_type in self.df_type_dict.keys():
            if self.df_type_dict[df_type] is None:
                self.init_df(df_type=df_type)
            file_name = os.path.join(dir, name + "_" + df_type + ".csv")
            with open(file_name, "w") as f:
                self.df_type_dict[df_type].to_csv(file_name)
        else:
            raise ValueError()

    def update_laser_info(self, obj):
        df_new_results = None
        if type(obj) is Fibre:
                df_new_results = obj.stepper.storage.get_df_result(self.z_curr)
        if type(obj) is OpenclFibre:
                df_new_results = obj.get_df_result(self.z_curr)
        
        if df_new_results is not None:
            self.z_curr = df_new_results.index.get_level_values("z [mm]").values[-1]
            if self.df_results is None: 
                self.df_results = df_new_results
            else:
                self.df_results = pd.concat([self.df_results, df_new_results])

    def update_result_df(self, obj):
        self.update_laser_info(obj)

    def save_result_df_to_scv(self, dir):            
        check_dir(dir)
        if self.df_results is not None:
            self.df_results.to_csv(os.path.join(dir, "laser_info.csv"))

    def run(self):
        """
        Propagate field through each module, with the resulting field at the
        exit of each module stored in a dictionary, with module name as key.
        """
        self.field = byte_align(self.field)
        for module in self.modules:
            print(module.name)
            self.field = module(self.domain, self.field)
            print(f"module run: {psutil.virtual_memory().used >> 20} MiB, free: {psutil.virtual_memory().free >> 20} MiB")
            self.fields[module.name] = self.field
            self.update_result_df(module)
            print(f"results updated: {psutil.virtual_memory().used >> 20} MiB, free: {psutil.virtual_memory().free >> 20} MiB")
            self.save_result_df_to_scv(self.charact_dir)
