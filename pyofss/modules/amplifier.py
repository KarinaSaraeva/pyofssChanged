
"""
    Copyright (C) 2011, 2012  David Bolt

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
import pyopencl as cl
import pyopencl.array as cl_array
from scipy import power, sqrt
from pyofss.field import fft, ifft, fftshift, energy, spectral_power
import numpy as np
from abc import ABC, abstractmethod
from pyofss.domain import Domain, lambda_to_omega, lambda_to_nu
import pandas as pd
import os.path
import warnings
    
class AmplifierBase(ABC):
    @abstractmethod
    def factor(self, A, h):
        pass

    @abstractmethod
    def set_domain(self, domain):
        pass


class Amplifier(AmplifierBase):
    """ 
    :param string name: name of this module
    :param double gain: amount of (logarithmic) gain. *Unit: dB*
    :param double E_sat: energy of saturation gain. *Unit: nJ*
    :param double P_sat: power of saturation gain. *Unit: W*
    :param double Tr: repetition rate of pulse. *Unit: ns*
    :param double length: length of the amplifier. *Unit m*
    :parma double lamb0: amplification lorenz profile central wavelength. *Unit: nm*
    :parma double bandwidth: amplification lorenz profile width. *Unit: nm*

    Amplifier provides gain based on simple amplification model with saturation with possible usage of an amplification lorenz curve 
    """

    def __init__(self, name="simple_saturation", gain=None,
                 E_sat=None, P_sat=None, Tr=None, length=1.0, lamb0=None, bandwidth=None, use_Er_profile=False, prg=None, queue=None, ctx=None, dorf="double"):
        self.length = length
        #print(f"amplifier length equals {self.length}")

        if gain is None:
            warnings.warn("The gain is not defined.")

        if (P_sat is not None) and (E_sat is not None):
            raise Exception(
                "There should be only one parameter of saturation.")

        if (bandwidth is not None and lamb0 is not None):
            self.delta = (Domain.vacuum_light_speed * bandwidth) / (lamb0 * lamb0)
            self.lamb = lamb0
            #print(f"YDF lanmbda = {self.lamb} nm")
            #print(f"YDF bandwidth = {bandwidth} nm")
        else: 
            self.delta = None
            self.lamb = None

        self.name = name
        self.gain = gain

        if E_sat is not None:
            self.E_sat = E_sat
        elif P_sat is not None:
            if Tr is None:
                raise Exception("Repetition rate is not defined.")
            self.E_sat = P_sat*Tr  # nJ
        else:
            self.E_sat = None
        self.field = None

        self.use_Er_profile = use_Er_profile

        self.prg = prg
        self.queue = queue
        self.ctx = ctx
        self.np_float = None
        float_conversions = {"float": np.float32, "double": np.float64}
        self.np_float = float_conversions[dorf]

    def prepare_arrays_on_device(self):
        self.spectral_power_buffer = cl_array.zeros(self.queue, self.shape, self.np_float)   
        self.g_buffer = cl_array.zeros(self.queue, self.shape, self.np_float)  

    def send_array_to_device_const(self, array):
        return cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=array.astype(self.np_float))

    @property
    def spectal_filtration_array(self):
        try:
            return self._spectal_filtration_array
        except:
            if self.use_Er_profile:
                A = pd.read_csv(os.path.dirname(__file__) + '/../data/Er_profile.csv')
                factorArray = np.interp(self.domain.Lambda, A["lambda [nm]"].values, A["gain"].values)
                factorArray = factorArray/np.amax(factorArray)
                factorArray[factorArray<0] = 0
            elif self.delta is not None and self.lamb is not None:
                factorArray = 1/(1 + 4*(((self.domain.nu -lambda_to_nu(self.lamb))**2)/((self.delta)**2)))
            else:
                factorArray = np.ones(len(self.domain.Lambda))

            # DF = pd.DataFrame(np.column_stack((self.domain.Lambda, factorArray)), columns=["lambda [nm]", "gain"])
            # DF.to_csv(os.path.dirname(__file__) + "/../data/factor_arrray.csv")

            self._spectal_filtration_array = fftshift(factorArray)

            return self._spectal_filtration_array

    @property
    def gain_times_log(self):
        try:
            return self.gain_times
        except:
            gain_times = np.power(10, 0.1 * self.gain)
            if self.use_Er_profile:
                self.gain_times = self.spectal_filtration_array*gain_times
            else:
                self.gain_times = np.ones(self.shape)*np.power(10, 0.1 * self.gain)

            self.gain_times = np.log(self.gain_times)

            if self.prg:
                self.gain_times = self.send_array_to_device_const(self.gain_times)

            return self.gain_times

    def factor(self, A, h):
        """ amplification factor used in an exponent of a linearity step """        
        G0 = (self.gain_times_log*h) / (2*self.length)

        E = energy(A, self.domain.t)
        factor = G0/(1.0 + E/self.E_sat)

        if not self.use_Er_profile:
            factor *= self.spectal_filtration_array
        return factor
    
    def cl_exp_factor(self, energy, h):
        """ amplification factor used in an exponent of a linearity step """
        factor = h / (2*self.length*(1.0 + energy/self.E_sat))
        
        self.prg.cl_array_exponent_with_factor(self.queue, self.shape, None, self.gain_times_log, self.np_float(factor), self.g_buffer.data)

        if not self.use_Er_profile:
            self.prg.cl_multiply_array_by_factor(self.queue, self.shape, None, self.gain_times_log, self.np_float(factor), self.g_buffer.data)
            self.prg.cl_multiply_array_by_another_const(self.queue, self.shape, None, self.g_buffer.data, self.spectal_filtration_array)
        

    def set_domain(self, domain):
        self.domain = domain
        self.shape = self.domain.nu.shape
        if self.prg:
            self.prepare_arrays_on_device()


class Amplifier2LevelModel(AmplifierBase):
    def __init__(self, name="amplifier2LevelModel", Pp=None, N=None, Rr=None, prg=None, queue=None, ctx=None, dorf="double"):
        """ 
        :param string name: name of this module
        :param double Pp: pump power *Unit: W*
        :param double N: the total number of Yb-ions integrated over the fibre mode
        :param double Rr: round trip frequency *Unit: THz*

        For OpenCl usage:

        :param object prg: opencl programm that includes all the needed opencl operations
        :param object queue: opencl command queue
        :parma object ctx: opencl context
        :parma string dorf: type of opencl arrays

        Amplifier2LevelModel provides wavelength dependent gain based on a Yb rate equations.
        All the formulas are similar to those used in the article "Kirsch, D.C. at all., 2022, Communications Physics, 5(1), p.219."
        """
        #print(f"use two level Yb gain model")
        #constatnts 
        self.h_p = 6.62 * 1e-34
        self.T = 850.0 * 1e-6 # units s
        self.a = 3.0 * 1e-6 # units? m
        self.b = 60.0 * 1e-6
        self.NA = 0.13
        self.lamb_p = 976.0
        self.N = N * (1e25) * np.pi * self.a **2 # density per unit length
        self._Pp = Pp
        self.Tr = 1/Rr

        self.domain = None

        self.Pp_list = []
        self.inversion_factor_list = []
        self.gs_list = []

        self.sigma12_p = 2.5 * 1e-24 # units? m2
        self.sigma21_p = 2.44 * 1e-24

        self.prg = prg
        self.queue = queue
        self.ctx = ctx
        self.np_float = None
        float_conversions = {"float": np.float32, "double": np.float64}
        self.np_float = float_conversions[dorf]
        self.physical_power_factor = None


    def prepare_arrays_on_device(self):
        self.spectral_power_buffer = cl_array.zeros(self.queue, self.shape, self.np_float)   
        self.g_s_buffer = cl_array.zeros(self.queue, self.shape, self.np_float)  

        # use read-only memory for constant to be cashed on devise

    def send_array_to_device_const(self, array):
        return cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=array.astype(self.np_float))

    def load_sigma_s(self):
        import os.path
        df = pd.read_csv(os.path.dirname(__file__) + '/../data/CrossSectionData.dat', delimiter="\t", header=None)

        self.delta_nu = df[df.columns[0]].values
        self.sigma12_s = df[df.columns[1]].values
        self.sigma21_s = df[df.columns[2]].values

    def interpolate_sigma_s(self):
        import os.path
        from scipy import interpolate
        IUS = interpolate.InterpolatedUnivariateSpline
        df = pd.read_csv(os.path.dirname(__file__) + "/../data/CrossSectionDataExp.txt", delimiter=",")

        spl12_s_raw = IUS(df[df.columns[0]] * 1e9, df[df.columns[1]] * 1e27, ext=1)
        spl21_s_raw = IUS(df[df.columns[0]] * 1e9, df[df.columns[2]] * 1e27, ext=1)

        self.sigma12_s = spl12_s_raw(self.domain.Lambda)*1e-27
        self.sigma21_s = spl21_s_raw(self.domain.Lambda)*1e-27

    def set_domain(self, domain):
        self.domain = domain
        self.interpolate_sigma_s()
        self.shape = self.domain.nu.shape
        self.physical_power_factor = self.np_float(self.shape[0]*self.domain.dt/(self.Tr))
        if self.prg:
            self.prepare_arrays_on_device()
            self.physical_power_factor = self.send_array_to_device_const(self.physical_power_factor)

    @property
    def rho_s(self):
        try:
            return self._rho_s
        except:
            V = (2.0 * np.pi / self.domain.Lambda) * self.a * self.NA * 1e9
            omega_s = self.a * (0.616 + 1.66/pow(V,1.5) + 0.987/pow(V,6.0))
            Gamma_s = 1.0 - np.exp(-2.0 * self.a**2 / (omega_s * omega_s))
            self._rho_s = Gamma_s / (np.pi * self.a**2)
            return self._rho_s
        
    @property
    def rho_p(self):
        try:
            return self._rho_p
        except:
            Gamma_p    = (self.a**2) / (self.b**2)
            self._rho_p      = Gamma_p / (np.pi * self.a**2)
            return self._rho_p

    @property
    def Psat_s(self):
        try:
            return self._Psat_s
        except:
            self._Psat_s = (self.h_p * self.domain.nu * 1e12) / (self.T * (self.sigma12_s + self.sigma21_s) * self.rho_s) 
            if self.prg:
                self._Psat_s = self.send_array_to_device_const(self._Psat_s)
            return self._Psat_s

    @property
    def Psat_p(self):
        try:
            return self._Psat_p
        except:
            self._Psat_p = (self.h_p * lambda_to_nu(self.lamb_p) * 1e12) / (self.T * (self.sigma12_p + self.sigma21_p) * self.rho_p)
            return self._Psat_p
        
    @property
    def eta_s(self):
        try:
            return self._eta_s
        except:
            self._eta_s = self.sigma12_s * self.rho_s * self.N
            if self.prg:
                self._eta_s = self.send_array_to_device_const(self._eta_s)
            return self._eta_s

    @property
    def eta_p(self):
        try:
            return self._eta_p
        except:
            self._eta_p = self.sigma12_p * self.rho_p * self.N
            return self._eta_p

    @property
    def alpha_s(self):
        try:
            return self._alpha_s
        except:
            self._alpha_s = (self.sigma12_s + self.sigma21_s) * self.rho_s
            if self.prg:
                self._alpha_s = self.send_array_to_device_const(self._alpha_s)
            return self._alpha_s
        
    @property
    def alpha_p(self):
        try:
            return self._alpha_p
        except:
            self._alpha_p = (self.sigma12_p + self.sigma21_p) * self.rho_p
            return self._alpha_p
        
    @property
    def Pp(self):
        return self._Pp
    
    @property
    def ratio_p(self):
        try:
            return self._ratio_p
        except:
            self._ratio_p = self.sigma12_p / (self.sigma12_p + self.sigma21_p)
            return self._ratio_p
        
    @property
    def ratio_s(self):
        try:
            return self._ratio_s
        except:
            self._ratio_s = self.sigma12_s / (self.sigma12_s + self.sigma21_s)
            if self.prg:
                mask = np.isnan(self._ratio_s)
                self._ratio_s[mask] = 0
                self._ratio_s = self.send_array_to_device_const(self._ratio_s)
            return self._ratio_s

    def calculate_N2(self, Ps):
        numerator = self.ratio_p * self.Pp / self.Psat_p + np.sum((lambda x: x[~np.isnan(x)])(self.ratio_s * Ps / self.Psat_s))
        denominator = 1 + self.Pp / self.Psat_p + np.sum((lambda x: x[~np.isnan(x)])(Ps / self.Psat_s))
        N2 = (numerator/denominator) * self.N
        # self.inversion_factor_list.append(numerator/denominator)
        return N2
    
    def calculate_g_s(self, N2):
        return self.alpha_s * N2 - self.eta_s
    
    def calculate_g_p(self, N2):
        return self.alpha_p * N2 - self.eta_p
    
    def update_Pp(self, g_p, h):
        # self.Pp_list.append(self._Pp)
        self._Pp = self._Pp * np.exp(g_p * h * 1e3)
    
    def factor(self, A, h): 
        """ amplification factor used in an exponent of a linearity step """
        # g_s: array in frequency domain calculated with respect to current N2 and N1: recalculated on yeach z step
        # g_p: value calculated with respect to current N2 and N1: recalculated on yeach z step 
        if self.domain is None:
            raise Exception("Domain is not preset.")

        N2 = self.calculate_N2(spectral_power(A)*len(A)*self.domain.dt/(self.Tr))
        g_s = self.calculate_g_s(N2)
        g_p = self.calculate_g_p(N2)
        # self.gs_list.append(np.exp(g_s * h * 1e3 / 2))
        self.update_Pp(g_p, h)
        return fftshift(g_s * h * 1e3 / 2)
    
    def cl_copy(self, dst_buffer, src_buffer):
        """ Copy contents of one buffer into another. """
        cl.enqueue_copy(self.queue, dst_buffer.data, src_buffer.data).wait() #check how it works
    
    def cl_calculate_g_s_exponent(self, N2, h):
        self.prg.cl_calculate_g_s_exponent_const(self.queue, self.shape, None, self.g_s_buffer.data, self.alpha_s, self.eta_s, self.np_float(N2), self.np_float(h * 1e3 / 2)) # cant increase number of args
    
    def cl_calculate_N2(self, temp_arr_s): # Ps already sent to device
        self.prg.devide_array_by_another_const(self.queue, self.shape, None, temp_arr_s.data, self.Psat_s) # some values can be None
        temp_p = self.Pp / self.Psat_p
        denominator = 1 + temp_p + cl.array.sum(temp_arr_s, queue=self.queue).get()
        self.prg.multiply_array_by_another_const(self.queue, self.shape, None, temp_arr_s.data, self.ratio_s)
        numerator = self.ratio_p * temp_p + cl.array.sum(temp_arr_s, queue=self.queue).get()
        inversion_factor = numerator/denominator
        N2 = inversion_factor * self.N
        # self.inversion_factor_list.append(inversion_factor)
        return N2

    def cl_exp_factor(self, A, h): 
        """ same as self.factor: but update array self.gs_buffer stored on device """
        # A must be Fourie transormed and already sent to device
        self.prg.cl_physical_power(self.queue, self.shape, None, A.data, self.physical_power_factor, self.spectral_power_buffer.data)
        self.prg.cl_fftshift(self.queue, self.shape, None, self.spectral_power_buffer.data, self.np_float(self.shape[0]))
        N2 = self.cl_calculate_N2(self.spectral_power_buffer)
        self.cl_calculate_g_s_exponent(N2, h) # stored in self.g_s_buffer
        g_p = self.calculate_g_p(N2)
        # self.gs_list.append(self.g_s_buffer.get())
        self.update_Pp(g_p, h)

    def cl_clear(self, cl_arr):
        """ clear array on a device """
        if cl_arr is not None:
            if cl_arr.size > 0:
                cl_arr.data.release() 

    def cl_clear_const(self, cl_arr):
        """ clear const on a device """
        if cl_arr is not None:
            if cl_arr.size > 0:
                cl_arr.release() 

    def clear_arrays_on_device(self):
        """ clear everything stored on a device """
        if self.prg:
            self.cl_clear(self.spectral_power_buffer)
            self.cl_clear(self.g_s_buffer)

            self.cl_clear_const(self.Psat_s)
            self.cl_clear_const(self.eta_s)
            self.cl_clear_const(self.ratio_s)
            self.cl_clear_const(self.alpha_s)