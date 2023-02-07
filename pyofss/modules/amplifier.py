
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

from scipy import power, sqrt
from pyofss.field import fft, ifft, fftshift, energy, spectral_power
import numpy as np
from pyofss.domain import Domain, lambda_to_omega, lambda_to_nu
import pandas as pd
import warnings
    

class Amplifier(object):
    """
    :param string name: Name of this module
    :param double gain: Amount of (logarithmic) gain. *Unit: dB*
    :param double E_saturation: Energy of saturation gain. *Unit: nJ*
    :param double P_saturation: Power of saturation gain. *Unit: W*
    :param double rep_rate: Repetition rate of pulse. *Unit: MHz*

    Simple amplifier provides gain but no noise
    """

    def __init__(self, name="amplifier", gain=None,
                 E_sat=None, P_sat=None, rep_rate=None, length=1.0, lamb0 = None, bandwidth=None, steps=1):
        self.total_steps = steps
        self.length = length
        print(f"amplifier length equals {self.length}")

        if gain is None:
            warnings.warn("The gain is not defined.")

        if (P_sat is not None) and (E_sat is not None):
            raise Exception(
                "There should be only one parameter of saturation.")

        if (bandwidth is not None and lamb0 is not None):
            self.delta = (Domain.vacuum_light_speed * bandwidth) / (lamb0 * lamb0)
            self.lamb = lamb0
            print(f"YDF lanmbda = {self.lamb} nm")
            print(f"YDF bandwidth = {bandwidth} nm")
        else: 
            self.delta = None
            self.lamb = None

        self.name = name
        self.gain = gain

        if E_sat is not None:
            self.E_sat = E_sat
        elif P_sat is not None:
            if rep_rate is None:
                raise Exception("Repetition rate is not defined.")
            self.E_sat = 1e3*P_sat/rep_rate  # nJ
        else:
            self.E_sat = None
        self.field = None

    def __call__(self, field):
        # Convert field to spectral domain:
        self.field = fft(field)

        # Calculate linear gain from logarithmic gain (G_dB -> G_linear)
        M = power(10, 0.1 * self.gain)
        G = M / self.total_steps
        print(G)
        if self.E_sat is not None:
            E = energy(field, self.domain.t)
            G = G/(1.0 + E/self.E_sat)
        sqrt_G = sqrt(G)

        if self.domain.channels > 1:
            self.field[0] *= sqrt_G
            self.field[1] *= sqrt_G
        else:
            self.field *= sqrt_G

        # convert field back to temporal domain:
        return ifft(self.field)

    @property
    def spectal_filtration_array(self):
        try:
            return self._spectal_filtration_array
        except:
            if self.delta is not None and self.lamb is not None:
                factorArray = 1/(1 + 4*(((self.domain.nu -lambda_to_nu(self.lamb))**2)/((self.delta)**2)))
                DF = pd.DataFrame(fftshift(factorArray))
                DF.to_csv("LorentzFactorArray.csv")
                self._spectal_filtration_array = fftshift(factorArray)
            else:
                self._spectal_filtration_array = np.ones(len(self.domain.Lambda))
            
            return self._spectal_filtration_array

    def factor(self, A, h):
        if self.domain is None:
            raise Exception("Domain is not preset.")

        M = np.log(power(10, 0.1 * self.gain))
        G = (M*h) / (2*self.length)

        if self.E_sat is not None:
            E = energy(A, self.domain.t)
            G = G/(1.0 + E/self.E_sat)
        else:
            warnings.warn("saturation is not stated")

        factor = G * self.spectal_filtration_array
        return factor

    def set_domain(self, domain):
        self.domain = domain


class Amplifier2LevelModel(Amplifier):
    def __init__(self, name="amplifier2LevelModel"):
        print(f"use two level Yb gain model")
        super().__init__(name=name)
        #constatnts 
        self.h_p = 6.62*1e-34
        self.T = 850.0 * pow(0.1,6) # units? s
        self.a = 3.0 * pow(0.1,6)
        self.b = 60.0 * pow(0.1,6)
        self.NA = 0.13
        self.lamb_p = 976.0
        self.N = None
        self.domain = None

        self.sigma12_p = 2.5 * pow(0.1,24) # units? m2
        self.sigma21_p = 2.44* pow(0.1,24)
        self.load_sigma_s()

    def load_sigma_s(self):
        df = pd.read_csv("../data/CrossSectionData.dat", delimiter="\t", header=None)
        self.delta_nu = df[df.columns[0]]
        self.sigma12_s = df[df.columns[1]]
        self.sigma21_s = df[df.columns[2]]

    def set_domain(self, domain):
        self.domain = domain
        self.interpolate_sigma()

    @property
    def rho_s(self):
        try:
            return self._rho_s
        except:
            V = (2.0 * np.pi / self.domain.Lambda) * self.a * self.NA
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
            self._Psat_s = (self.h_p * self.domain.nu)/ (self.T * (self.sigma12_s + self.sigma21_s) * self.rho_s) # 10**24 to got from ps to s to get Wt
            return self._Psat_s

    @property
    def Psat_p(self):
        try:
            return self._Psat_p
        except:
            self._Psat_p = (self.h_p * lambda_to_nu(self.lamb_p)) / (self.T * (self.sigma12_p + self.sigma21_p) * self.rho_p) # 10**24 to got from ps to s to get Wt
            return self._Psat_p
        
    @property
    def eta_s(self):
        try:
            return self._eta_s
        except:
            self._eta_s =self.sigma12_s * self.rho_s * self.N

    @property
    def eta_p(self):
        try:
            return self._eta_p
        except:
            self._eta_p = self.sigma12_p * self.rho_p * self.N

    @property
    def alpha_s(self):
        try:
            return self._alpha_s
        except:
            self._alpha_s = (self.sigma12_s + self.sigma21_s) * self.rho_s
            return self._alpha_s
        
    @property
    def alpha_p(self):
        try:
            return self._alpha_p
        except:
            self._alpha_p = (self.sigma12_p + self.sigma21_p) * self.rho_p
            return self._alpha_p

    def interpolate_sigma(self):
        from scipy import interpolate
        IUS = interpolate.InterpolatedUnivariateSpline

        spl12_s = IUS(self.delta_nu, self.sigma12_s)
        spl21_s = IUS(self.delta_nu, self.sigma21_s)

        self.delta_nu = self.domain.nu - self.domain.centre_nu
        self.sigma12_s = spl12_s(self.delta_nu)
        self.sigma21_s = spl21_s(self.delta_nu)

    def calculate_N2(self, Pp, Ps):
        numerator = Pp / self.Psat_p + np.sum(Ps / self.Psat_s)
        denominator = 1 + numerator
        N2 = (numerator/denominator) * self.N
        return N2
    
    def calculate_g_s(self, N2):
        return self.alpha_s * N2 - self.eta_s
    
    def calculate_g_p(self, N2):
        return self.alpha_p * N2 - self.eta_p
    
    def factor(self, A, P): 
        # g_s: array in frequency domain is calculated with respect to current N2 and N1: recalculated on yeach z step
        # g_p: value is calculated with respect to current N2 and N1: recalculated on yeach z step 

        if self.domain is None:
            raise Exception("Domain is not preset.")

        N2 = self.calculate_N2(P, spectral_power(A))
        g_s = self.calculate_g_s(N2)
        g_p = self.calculate_g_p(N2)
        return g_s, g_p