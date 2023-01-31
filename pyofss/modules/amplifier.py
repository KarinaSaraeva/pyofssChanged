
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
from pyofss.field import fft, ifft, fftshift
from pyofss.field import energy
import numpy as np
from pyofss.domain import Domain, lambda_to_omega, lambda_to_nu
import pandas as pd

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
        self.spectalFiltrationArray = None

        if gain is None:
            raise Exception("The gain is not defined.")

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

    def __call__(self, field, step=1):
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

    def findSpectalFiltrationArray(self):
        if self.delta is not None and self.lamb is not None:
            factorArray = 1/(1 + 4*(((self.domain.nu -lambda_to_nu(self.lamb))**2)/((self.delta)**2)))
            DF = pd.DataFrame(fftshift(factorArray))
            DF.to_csv("LorentzFactorArray.csv")
            print(f"max value in factorArray: {np.amax(factorArray)}, at {self.domain.Lambda[np.argmax(factorArray)]} nm")
            return fftshift(factorArray)
        else:
            return np.ones(len(self.domain.Lambda))

    def factor(self, A, h):
        M = np.log(power(10, 0.1 * self.gain))
        G = (M*h) / (2*self.length)
        if self.E_sat is not None:
            E = energy(A, self.domain.t)
            G = G/(1.0 + E/self.E_sat)
        else:
            print("saturation is not stated!!!")
        if (self.spectalFiltrationArray is not None):
            factor = G * self.spectalFiltrationArray
        else:
            factor = G * self.findSpectalFiltrationArray()
            print(f"max value in gainArray: {np.amax(factor)}, at {self.domain.Lambda(np.argmax(factor))} nm")
        return factor

    def setDomain(self, domain):
        self.domain = domain
        self.spectalFiltrationArray = self.findSpectalFiltrationArray()
