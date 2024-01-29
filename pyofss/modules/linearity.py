
"""
    Copyright (C) 2012  David Bolt, 2020 Denis Kharenko

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

from scipy.constants import constants
# factorial location changes in scipy 1.3
try:
    from scipy.special import factorial
except ImportError:  # try the old one for compatibility reson
    from scipy.misc import factorial
from scipy import log10, exp
import numpy as np
import pandas as pd
from pyofss.field import fft, ifft, fftshift, ifftshift

from pyofss.field import fft, ifft, fftshift, temporal_power


def convert_dispersion_to_physical(D=0.0, S=0.0, Lambda=1550.0):
    """
    :param double D: Dispersion. *Unit:* :math:`ps / (nm \cdot km)`
    :param double S: Dispersion slope. *Unit:* :math:`ps / (nm^2 \cdot km)`
    :param double Lambda: Centre wavelength. *Unit: nm*
    :return: Second and third order dispersion
    :rtype: double, double

    Require D and S. Return a tuple containing beta_2 and beta_3.
    """
    if (D == 0.0) and (S == 0.0):
        return (0.0, 0.0)

    # Constant 1.0e-3 modifies units of c to be nm / ps, see Domain.
    factor = Lambda ** 2 / (2.0 * np.pi * 1.0e-3 * constants.c)
    square_factor = factor ** 2

    beta_2 = -factor * D
    beta_3 = square_factor * (S + (2.0 * D / Lambda))

    return (beta_2, beta_3)


def convert_dispersion_to_engineering(beta_2=0.0, beta_3=0.0, Lambda=1550.0):
    """
    :param double beta_2: Second-order dispersion. *Unit:* :math:`ps^2 / km`
    :param double beta_3: Third-order dispersion. *Unit:* :math:`ps^3 / km`
    :param double Lambda: Centre wavelength. *Unit: nm*
    :return: Dispersion, dispersion slope
    :rtype: double, double

    Require beta_2 and beta_3. Return a tuple containing D and S.
    """
    if (beta_2 == 0.0) and (beta_3 == 0.0):
        return (0.0, 0.0)

    factor_1 = (2.0 * np.pi * 1.0e-3 * constants.c) / (Lambda ** 2)
    square_factor_1 = factor_1 * factor_1

    factor_2 = (2.0 * factor_1) / Lambda

    D = -factor_1 * beta_2
    S = square_factor_1 * beta_3 + factor_2 * beta_2

    return (D, S)


def convert_alpha_to_linear(alpha_dB=0.0):
    """
    :param double alpha_dB: Logarithmic attenuation factor
    :return: Linear attenuation factor
    :rtype: double

    Converts a logarithmic attenuation factor to a linear attenuation factor
    """
    factor = 10.0 * log10(exp(1.0))
    return alpha_dB / factor


def convert_alpha_to_dB(alpha_linear=0.0):
    """
    :param double alpha_linear: Linear attenuation factor
    :return: Logarithmic attenuation factor
    :rtype: double

    Converts a linear attenuation factor to a logarithmic attenuation factor
    """
    factor = 10.0 * log10(exp(1.0))
    return alpha_linear * factor


class Linearity(object):
    """
    :param double alpha: Attenuation factor
    :param array_like beta: Array of dispersion parameters
    :param string sim_type: Type of simulation, "default" or "wdm"
    :param bool use_cache: Cache calculated values if using fixed step-size
    :param double centre_omega: Angular frequency to use for dispersion array
    :param bool phase_lim: Limits phase values in range of [-2*pi,2*pi] and use it periodic nature
                    could be important for GPU computations

    Dispersion is used by fibre to generate a fairly general dispersion array.
    """
    def __init__(self, alpha=None, beta=None, sim_type=None,
                 use_cache=False, centre_omega=None, phase_lim=False, 
                 amplifier = None, use_Er_noise=False):

        self.alpha = alpha
        self.beta = beta
        self.centre_omega = centre_omega
        self.phase_lim = phase_lim

        self.amplifier = amplifier
        self.use_Er_noise = use_Er_noise

        self.generate_linearity = getattr(self, "%s_linearity" % sim_type,
                                          self.default_linearity)
        self.generate_cache = getattr(self, "%s_cache" % sim_type,
                                          self.default_cache)
        self.lin = getattr(self, "%s_f" % sim_type, self.default_f)

        if use_cache:
            self.exp_lin = getattr(self, "%s_exp_f_cached" % sim_type,
                                   self.default_exp_f_cached)
        else:
            self.exp_lin = getattr(self, "%s_exp_f" % sim_type,
                                   self.default_exp_f)

        # Allows storing of calculation involving an exponential. Provides a
        # significant speed increase if using a fixed step-size.
        self.cached_factor = None

        self.factor = None
        self.Domega = None

        self.gain_arr = []
        self.sigma_arr = []

    def __call__(self, domain):
        self.domain = domain    
        return self.generate_linearity(domain)

    def default_linearity(self, domain):
        # Calculate dispersive terms:
        if self.amplifier is not None:
            self.amplifier.set_domain(domain)
            
        if self.beta is None:
            self.factor = 0.0
        else:
            if self.centre_omega is None:
                self.Domega = domain.omega - domain.centre_omega
            else:
                self.Domega = domain.omega - self.centre_omega

            # Allow general dispersion:
            terms = 0.0
            for n, beta in enumerate(self.beta):
                terms += beta * np.power(self.Domega, n) / factorial(n)
            self.factor = 1j * fftshift(terms)

        # Include attenuation term if available:
        if self.alpha is None:
            return self.factor
        else:
            self.factor -= 0.5 * self.alpha
            return self.factor

    def wdm_linearity(self, domain):
        # Calculate dispersive terms:
        if self.beta is None:
            self.factor = (0.0, 0.0)
        else:
            if self.centre_omega is None:
                self.Domega = (domain.omega - domain.centre_omega,
                               domain.omega - domain.centre_omega)
            else:
                self.Domega = (domain.omega - self.centre_omega[0],
                               domain.omega - self.centre_omega[1])

            terms = [0.0, 0.0]
            for n, beta in enumerate(self.beta[0]):
                terms[0] += beta * np.power(self.Domega[0], n) / factorial(n)
            for n, beta in enumerate(self.beta[1]):
                terms[1] += beta * np.power(self.Domega[1], n) / factorial(n)
            self.factor = (1j * fftshift(terms[0]), 1j * fftshift(terms[1]))

        # Include attenuation terms if available:
        if self.alpha is None:
            return self.factor
        else:
            self.factor[0] -= 0.5 * self.alpha[0]
            self.factor[1] -= 0.5 * self.alpha[1]
            return self.factor

    def _limit_imag_part(self, hf):
        """
        Align values into [-2*pi,2*pi] range
        """
        lhf = np.imag(hf) / (2 * np.pi)
        return np.real(hf) + 1j * 2 * np.pi * np.modf(lhf)[0]

    def default_cache(self, h):
        hf = self.factor * h
        if self.phase_lim:
            hf = self._limit_imag_part(hf)
        self.cached_factor = np.exp(hf)

    def wdm_cache(self, h):
        hf0 = self.factor[0] * h
        hf1 = self.factor[1] * h
        if self.phase_lim:
            hf0 = self._limit_imag_part(hf0)
            hf1 = self._limit_imag_part(hf1)
        self.cached_factor = [np.exp(hf0),
                              np.exp(hf1)]

    def cache(self, h):
        print("Caching linear factor")
        self.generate_cache(h)

    def default_f(self, A, z):
        return ifft(self.factor * fft(A))

    def default_exp_f(self, A, h):
        hf = self.factor * h
        if self.phase_lim:
            hf = self._limit_imag_part(hf)
        if self.amplifier is None:
            return ifft(np.exp(hf) * fft(A))
        else:        
            amp_factor = self.amplifier.factor(A, h)   
            field = ifft(np.multiply(np.exp(amp_factor), np.exp(hf) * fft(A)))
            if self.use_Er_noise:
                self.gain_arr.append(ifftshift(np.exp(2*amp_factor))[int(self.domain.Lambda.shape[0]/2)]) #!!!
                noise = self.get_Er_noise(A, ifftshift(np.exp(2*amp_factor))[int(self.domain.Lambda.shape[0]/2)])  
                field += noise
            return field
    
    def default_exp_f_cached(self, A, h):
        if self.cached_factor is None:
            self.cache(h)
        if self.amplifier is None:
            return ifft(self.cached_factor * fft(A))
        else:
            amp_factor = self.amplifier.factor(A, h)   
            field = ifft(np.multiply(np.exp(amp_factor), self.cached_factor * fft(A)))
            if self.use_Er_noise:
                self.gain_arr.append(ifftshift(np.exp(2*amp_factor))[int(self.domain.Lambda.shape[0]/2)]) #!!!
                noise = self.get_Er_noise(A, ifftshift(np.exp(2*amp_factor))[int(self.domain.Lambda.shape[0]/2)])  
                field += noise
            return field
        
    def get_Er_noise(self, A, gain):
        # s_power = np.mean(temporal_power(A))
        plank = 6.626e-10; # [W*ps^2]
        NF_dB = 4.5
        NF = np.power(10, NF_dB / 10)
        n_sp = NF * gain / (2 * (gain - 1))
        nyu0 = self.domain.centre_nu
        h = self.domain.dt
        alfa = 1
        sigma = np.sqrt(plank*nyu0*n_sp*(gain-1.0)*alfa/h)
        self.sigma_arr.append(sigma) #!!!
        re_noise = np.random.normal(loc=0.0, scale=sigma, size=A.shape)
        im_noise = np.random.normal(loc=0.0, scale=sigma, size=A.shape)
        noise = re_noise + 1j * im_noise
      
        return noise

    def wdm_f(self, As, z):
        return np.asarray([ifft(self.factor[0] * fft(As[0])),
                           ifft(self.factor[1] * fft(As[1]))])

    def wdm_exp_f(self, As, h):
        hf0 = h * self.factor[0]
        hf1 = h * self.factor[1]
        if self.phase_lim:
            hf0 = self._limit_imag_part(hf0)
            hf1 = self._limit_imag_part(hf1)
        return np.asarray([ifft(np.exp(hf0) * fft(As[0])),
                           ifft(np.exp(hf1) * fft(As[1]))])

    def wdm_exp_f_cached(self, As, h):
        if self.cached_factor is None:
            self.cache(h)
        return np.asarray([ifft(self.cached_factor[0] * fft(As[0])),
                           ifft(self.cached_factor[1] * fft(As[1]))])
