
"""
    Copyright (C) 2011, 2012  David Bolt,
    2019-2021 Vladislav Efremov, Denis Kharenko

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

import numpy as np
import scipy.fftpack
import scipy.integrate as integrate
from scipy.signal import find_peaks, peak_widths

try:
    import pyfftw
    scipy.fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()
    print("PyFFTW has been imported, use PYFFTW_NUM_THREADS and PYFFTW_PLANNER_EFFORT for tuning")
except:
    pass


# Although use of global variables is generally a bad idea, in this case it is
# a simple solution to recording the number of ffts used:
fft_counter = 0


def temporal_power(A_t, normalise=False):
    """
    :param array_like A_t: Input field array in the temporal domain
    :param bool normalise: Normalise returned array to that of maximum value
    :return: Array of power values
    :rtype: array_like

    Generate an array of temporal power values from complex amplitudes array.
    """
    P = np.abs(A_t) ** 2

    if(normalise):
        P /= max(P)

    return P


def spectral_power(A_t, normalise=False):
    """
    :param array_like A_t: Input field array in the temporal domain
    :param bool normalise: Normalise returned array to that of maximum value
    :return: Array of power values
    :rtype: array_like

    Generate an array of spectral power values from complex amplitudes array.
    *Note: Expect input field to be in temporal domain. Never input A_nu!*
    """
    P = np.abs(fft(A_t)) ** 2

    if(normalise):
        P /= max(P)

    return ifftshift(P)


def phase(A_t, unwrap=True):
    """
    :param array_like A_t: Input field array in the temporal domain
    :param bool unwrap: Whether to unwrap phase angles from fixed range
    :return: Array of phase angle values
    :rtype: array_like

    Generate an array of phase angles from complex amplitudes array.
    """
    if(unwrap):
        return np.unwrap(np.angle(A_t))
    else:
        return np.angle(A_t)


def chirp(A_t, window_nu, unwrap=True):
    """
    :param array_like A_t: Input field array in the temporal domain
    :param double window_nu: Spectral window of the simulation
    :param bool unwrap: Whether to unwrap phase angles from fixed range
    :return: Array of chirp values
    :rtype: array_like

    Generate an array of chirp values from complex amplitudes array.
    """
    if(unwrap):
        return -np.gradient(phase(A_t, True)) * window_nu
    else:
        return -np.gradient(phase(A_t, False)) * window_nu


def fft(A_t):
    """
    :param array_like A_t: Input field array in the temporal domain
    :return: Output field array in the spectral domain
    :rtype: array_like

    Fourier transform field from temporal domain to spectral domain.
    *Note: Physics convention -- positive sign in exponential term.*
    """
    global fft_counter
    fft_counter += 1

    return scipy.fftpack.ifft(A_t)


def ifft(A_nu):
    """
    :param array_like A_nu: Input field array in the spectral domain
    :return: Output field array in the temporal domain
    :rtype: array_like

    Inverse Fourier transform field from spectral domain to temporal domain.
    *Note: Physics convention -- negative sign in exponential term.*
    """
    global fft_counter
    fft_counter += 1

    return scipy.fftpack.fft(A_nu)


def ifftshift(A_nu):
    """
    :param array_like A_nu: Input field array in the spectral domain
    :return: Shifted field array in the spectral domain
    :rtype: array_like

    Shift the field values from "FFT order" to "consecutive order".
    """
    return scipy.fftpack.fftshift(A_nu)


def fftshift(A_nu):
    """
    :param array_like A_nu: Input field array in the spectral domain
    :return: Shifted field array in the spectral domain
    :rtype: array_like

    Shift the field values from "consecutive order" to "FFT order".
    """
    return scipy.fftpack.ifftshift(A_nu)


def energy(A_t, t):
    """
    :param array_like A_t: Input field array in the temporal domain
    :param double t: Temporal window of the simulation
    :return: Energy of the field
    :rtype: double

    Energy calculation
    """
    E = integrate.simps(temporal_power(A_t), t*1e-3)  # nJ
    
    return E


def inst_freq(A_t, dt):
    """
    :param array_like A_t: Input field array in the temporal domain
    :return: Instant frequency array
    :rtype: array_like

    Generate an array of instantaneous frequency
    """
    ph = phase(A_t)
    return np.append(np.diff(ph)/dt, 0.)


def loss_infrared_db(wl):
    """
    :param wavelength, nm
    :return: losses in dB

    See Agrawal "Fiber-Optic Communication Systems" ch2 page 56
    x = [1500, 1600, 1650, 1700, 1750, 1800] wavelengths
    y = [0.01, 0.05, 0.1, 0.3, 1, 2.5] losses in dB
    """
    p = np.array([-1.75292532e+03,  1.95607318e-02])
    return np.exp(p[1]*(wl+p[0]))


def loss_rayleigh_db(wl):
    """
    :param wavelength, nm
    :return: losses in dB

    See, eg. Argawal "Fiber-Optic Communication Systems" ch2
    """
    factor = (1-0.14/(wl*1e-3)**4)
    factor[factor <= 0] = 1e-16
    factor = -10*np.log10(factor)
    return factor


def max_peak_params(P, prominence):
    """
    :param power array *Unit: W*

    :return: 
    param double power_max: maximum power *Unit: W*,  
    param double pulse_fwhm: FWHM *Unit: input arr indexes*, 
    param double left_ind, right_ind: interpolated positions of left and right intersection points of a FWHM line *Unit: input arr indexes*,
    """
    peaks, properties = find_peaks(P, height=0)
    max_peak_ind = np.argmax(properties['peak_heights'])
    results_fwhm = peak_widths(P, [peaks[max_peak_ind]], rel_height=0.5)

    ind_max = np.argmax(results_fwhm[1])

    heigth_fwhm = results_fwhm[1][ind_max]
    fwhm = results_fwhm[0][ind_max]  # the hightest peak's fwhm

    left_ind = results_fwhm[2][ind_max]
    right_ind = results_fwhm[3][ind_max]

    return heigth_fwhm, fwhm, left_ind, right_ind


def spectrum_width_params(P, prominence=0.0001):
    """
    :param power array *Unit: W*

    :return: 
    param double power_max: maximum power *Unit: W*,  
    param double pulse_fwhm: FWHM *Unit: input arr indexes*, 
    param double left_ind, right_ind: interpolated positions of left and right intersection points of a FWHM line *Unit: input arr indexes*,
    """
    def find_x(y_peak, x_peak, is_left, ydata):
        if (is_left):
            x = np.where(ydata < y_peak)[0]
            filtered_x = x[np.where(x < x_peak)]
            boundary = np.amax(filtered_x) if len(filtered_x) > 0  else 0 
        else:
            x = np.where(ydata < y_peak)[0]
            filtered_x = x[np.where(x > x_peak)]
            boundary = np.amin(filtered_x) if len(filtered_x) > 0  else 0 
        return int(boundary)

    peaks, _ = find_peaks(P, height=0, prominence=prominence)

    if (len(peaks) > 1):
        heigth_fwhm = np.amax(P)/2
        peaks.sort()
        left_ind = find_x(heigth_fwhm, peaks[0], True, P)
        right_ind = find_x(heigth_fwhm, peaks[-1], False, P)
        fwhm = right_ind - left_ind
    elif (len(peaks) == 0):
        heigth_fwhm = None
        left_ind = None
        right_ind = None
        fwhm = None
    else:
        results_fwhm = peak_widths(P, peaks, rel_height=0.5)
        heigth_fwhm = np.amax(results_fwhm[1])
        fwhm = results_fwhm[0][0]
        left_ind = results_fwhm[2][0]
        right_ind = results_fwhm[3][0]

    return heigth_fwhm, fwhm, left_ind, right_ind


def get_duration_spec(P, d_x, prominence=None):
    if prominence is None:
        prominence = np.amax(P)/100 
    heigth_fwhm, fwhm, left_ind, right_ind = spectrum_width_params(
        P, prominence=prominence)
    return abs(fwhm)*d_x

def get_duration(P, d_x, prominence=None):
    if prominence is None:
        prominence = np.amax(P)/100 
    heigth_fwhm, fwhm, left_ind, right_ind = max_peak_params(
        P, prominence=prominence)
    return fwhm*d_x