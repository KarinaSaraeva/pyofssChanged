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

# Numpy is used widely so import by default
import numpy as np

# Import simulation modules
from .system import System
from .domain import Domain

# Import system modules
from .modules.generator import Generator
from .modules.gaussian import Gaussian
from .modules.sech import Sech
from .modules.amplifier import Amplifier, Amplifier2LevelModel
from .modules.bit import Bit, Bit_stream
from .modules.fibre import Fibre
from .modules.storage import reduce_to_range
from .modules.filter import Filter
from .modules.sat_absorber import Sat_absorber
from .modules.modulation import Modulation
from .modules.splitter import Splitter
from .modules.plotter import *
from .modules.amplifier_ch import Amplifier_ch
from .modules.amplifier_ch import Amplifier_ch_numb
from .modules.coupler import Coupler
from .modules.filter_doubl import Filter_doubl
from .modules.diss_soliton import Diss_soliton
from .modules.mirror import Mirror
from .modules.delay import Delay
from .modules.dispersion import Dispersion
from .modules.static_pump_wdm import StaticPumpWDM
from .modules.storage import Storage
from .modules.fibre_plotter import FibrePlotter, visualise_results_df, visualise_fields_df
from .modules.noise import Noise

try:
    from .modules.opencl_fibre import OpenclFibre, OpenclProgramm
except ImportError:
    print("OpenclFibre is not activated, check if pyopencl and reikna modules are installed")

# Import helper functions
from .domain import nu_to_omega, nu_to_lambda
from .domain import omega_to_nu, omega_to_lambda
from .domain import lambda_to_nu, lambda_to_omega
from .domain import dnu_to_dlambda, dlambda_to_dnu
from .domain import align_with_nu_grid

# Import useful conversions
from .field import fft, ifft, fftshift, ifftshift
from .field import temporal_power, spectral_power, spectrum_width_params
from .field import phase, chirp
from .field import energy
from .field import inst_freq
from .field import loss_infrared_db
from .field import loss_rayleigh_db
from .field import get_downsampled, get_duration, get_duration_spec

# Import pulse width conversion function
from .modules.generator import convert_pulse_width

# Import conversions for dispersion and attenuation
from .modules.linearity import convert_dispersion_to_physical
from .modules.linearity import convert_dispersion_to_engineering
from .modules.linearity import convert_alpha_to_linear
from .modules.linearity import convert_alpha_to_dB

# Import helper functions for nonlinear parameter and raman term
from .modules.nonlinearity import calculate_gamma
from .modules.nonlinearity import calculate_raman_term
from .modules.nonlinearity import calculate_raman_term_silica

# Import function for work with files
from .system import field_save
from .system import field_load
