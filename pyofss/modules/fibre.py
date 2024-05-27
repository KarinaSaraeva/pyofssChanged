
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

from numpy import amax, abs
from pyofss.modules.amplifier import AmplifierDistributed, Amplifier2LevelModel
from .linearity import Linearity
from .nonlinearity import Nonlinearity
from .stepper import Stepper
from pyofss.domain import Domain
from pyofss.field import get_duration, temporal_power

import warnings

class FiberInitError(Exception):
    pass


class Fibre(object):
    """
    :param string name: Name of this module
    :param double length: Length of fibre
    :param object alpha: Attenuation of fibre
    :param object beta: Dispersion of fibre
    :param double gamma: Nonlinearity of fibre
    :param string sim_type: Type of simulation
    :param Uint traces: Number of field traces required
    :param double local_error: Relative local error used in adaptive stepper
    :param string method: Method to use in ODE solver
    :param Uint total_steps: Number of steps to use for ODE integration
    :param bool self_steepening: Toggles inclusion of self-steepening effects
    :param bool raman_scattering: Toggles inclusion of raman-scattering effects
    :param float rs_factor: Factor determining the amount of raman-scattering
    :param bool use_all: Toggles use of general expression for nonlinearity
    :param double centre_omega: Angular frequency used within dispersion class
    :param double tau_1: Constant used in Raman scattering calculation
    :param double tau_2: Constant used in Raman scattering calculation
    :param double f_R: Constant setting the fraction of Raman scattering used
    :param double small_signal_gain: Constant used in amplification fiber modeling [dB]
    :param double E_sat: Saturation energy used in amplification fiber modeling [nJ]

    sim_type is either default or wdm.

    traces: If greater than 1, will save the field at uniformly-spaced points
    during fibre propagation. If zero, will output all saved points used.
    This is useful if using an adaptive stepper which will likely save
    points non-uniformly.

    method: simulation method such as RK4IP, ARK4IP.

    total_steps: If a non-adaptive stepper is used, this will be used
    to set the step-size between successive points along the fibre.

    local_error: Relative local error to aim for between propagtion points.
    """

    def __init__(self, name="fibre", length=1.0, alpha=None,
                 beta=None, gamma=0.0, sim_type=None, traces=1,
                 local_error=1.0e-6, method="ss_symmetric", total_steps=100,
                 self_steepening=False, raman_scattering=False,
                 rs_factor=0.003, use_all=False, centre_omega=None,
                 tau_1=12.2e-3, tau_2=32.0e-3, f_R=0.18, 
                 small_signal_gain=None, E_sat=None, P_sat=None, Tr=None, lamb0=None, bandwidth=None, 
                 use_Er_profile=False, use_Er_noise=False,
                 use_Yb_model=False, Pp_0 = None, N = None, Rr=None):

        use_cache = not(method.upper().startswith('A'))

        self.name = name
        self.length = length
        self.small_signal_gain = small_signal_gain
        self.domain = None
        self.amplifier = None

        if (use_Yb_model):
            self.amplifier = Amplifier2LevelModel(Pp=Pp_0, N=N, Rr=Rr)
        else:
            if (small_signal_gain is not None) and (E_sat is not None or (P_sat is not None and Tr is not None)):
                self.amplifier = AmplifierDistributed(
                    gain=self.small_signal_gain, E_sat=E_sat, P_sat=P_sat, rep_rate=1e3/Tr, length=self.length, lamb0=lamb0, bandwidth=bandwidth, use_Er_profile=use_Er_profile)

        self.linearity = Linearity(alpha=alpha, beta=beta, sim_type=sim_type,
                                   use_cache=use_cache, centre_omega=centre_omega, amplifier=self.amplifier, use_Er_noise=use_Er_noise)
        self.nonlinearity = Nonlinearity(gamma, sim_type, self_steepening,
                                         raman_scattering, rs_factor,
                                         use_all, tau_1, tau_2, f_R)

        class FunctionStep():
            """ Class to hold linear and nonlinear functions. """

            def __init__(self, l, n, linear, nonlinear, amplifier_step):
                self.l = l
                self.n = n
                self.linear = linear
                self.nonlinear = nonlinear
                self.amplifier_step = amplifier_step

            def __call__(self, A, z):
                return self.l(A, z) + self.n(A, z)
            
        class FunctionCharacts():
            def __init__(self, l_d, l_nl):
                self.l_d = l_d
                self.l_nl = l_nl

        self.function_step = FunctionStep(self.l, self.n, self.linear, self.nonlinear, self.amplifier_step)
        self.function_characts = FunctionCharacts(self.get_dispersion_length, self.get_nonlinear_length)

        self.stepper = Stepper(traces, local_error, method, self.function_step, self.function_characts,
                               self.length, total_steps)

    def __call__(self, domain, field):
        if self.domain != domain or \
                self.linearity.factor is None or \
                self.nonlinearity.factor is None:
            self.linearity(domain)
            self.nonlinearity(domain)
            self.domain = domain

        # Check relation between the step size and reference length
        reflen = self.calculate_reference_length(field)
        h = self.stepper.h
        if h > reflen * 1e-2:
            warnings.warn(
                f"WARN: {self.name}: h must be much less than dispersion length (L_D) and the nonlinear length (L_NL)\n        \
                now the minimum of the characteristic distances is equal to {reflen:.6f}*km* \n         \
                step is equal to {h}*km*"
            )

        if self.amplifier:  # TODO retrieve the pump power from the spectral domain?
            self.amplifier.init()

        # Propagate field through fibre:
        return self.stepper(field, domain)

    def l(self, A, z):
        """ Linear term. """
        return self.linearity.lin(A, z)

    def linear(self, A, h):
        """ Linear term in exponential factor. """
        return self.linearity.exp_lin(A, h)

    def n(self, A, z):
        """ Nonlinear term. """
        return self.nonlinearity.non(A, z)

    def nonlinear(self, A, h, B):
        """ Nonlinear term in exponential factor. """
        return self.nonlinearity.exp_non(A, h, B)
    
    def amplifier_step(self, A, h):
        return self.linearity.amplification_step(A, h)

    def calculate_reference_length(self, field):
        L_D = self.get_dispersion_length(field)
        L_NL = self.get_nonlinear_length(field)
        if (L_NL is not None) and (L_D is not None):
            return min(L_NL, L_D)
        elif (L_NL is None) and (L_D is None):
            return None
        else:
            return L_NL if L_D is None else L_D

    def get_nonlinear_length(self, field):  # TODO move to Nonlinearity?
        P_0 = amax(temporal_power(field))
        L_NL = 1 / (self.nonlinearity.gamma * P_0)
        return L_NL

    def get_dispersion_length(self, field):  # TODO move to Linearity?
        beta = self.linearity.beta
        beta_2 = beta[2] if beta is not None else None
        T_0 = get_duration(temporal_power(field), self.domain.dt)
        L_D =  T_0**2 / abs(beta_2)
        return L_D


if __name__ == "__main__":
    """
    Plot the result of a Gaussian pulse propagating through optical fibre.
    Simulates both (third-order) dispersion and nonlinearity.
    Use five different methods: ss_simple, ss_symmetric, ss_sym_rk4,
    ss_sym_rkf, and rk4ip. Expect all five methods to produce similar results;
    plot traces should all overlap. Separate traces should only be seen at
    a high zoom level.
    """
    from pyofss import Domain, System, Gaussian, Fibre
    from pyofss import temporal_power, multi_plot, labels

    import time

    domain = Domain(bit_width=200.0, samples_per_bit=2048*2)
    gaussian = Gaussian(peak_power=1.0, width=1.0)

    P_ts = []
    methods = ['ss_simple',
               'ss_symmetric', 'ss_symmetric+ss', 'ss_symmetric+raman', 'ss_symmetric+all',
               'ss_sym_rk4',
               'rk4ip', 'rk4ip+ss', 'rk4ip+raman', 'rk4ip+all']

    for m in methods:
        sys = System(domain)
        sys.add(gaussian)
        if m.count('+') == 0:
            sys.add(Fibre(length=5.0, method=m, total_steps=50,
                          beta=[0.0, 0.0, 0.0, 1.0], gamma=1.0))
        else:
            if m.split('+')[1] == 'ss':
                sys.add(Fibre(length=5.0, method=m.split('+')[0], total_steps=50,
                              beta=[0.0, 0.0, 0.0, 1.0], gamma=1.0, self_steepening=True))
            elif m.split('+')[1] == 'raman':
                sys.add(Fibre(length=5.0, method=m.split('+')[0], total_steps=50,
                              beta=[0.0, 0.0, 0.0, 1.0], gamma=1.0, use_all='hollenbeck'))
            else:
                sys.add(Fibre(length=5.0, method=m.split('+')[0], total_steps=50,
                              beta=[0.0, 0.0, 0.0, 1.0], gamma=1.0, self_steepening=True, use_all='hollenbeck'))

        start = time.time()
        sys.run()
        stop = time.time()
        P_ts.append(temporal_power(sys.field))

        print("Run time for {} method is {}".format(m, stop-start))

    multi_plot(sys.domain.t, P_ts, methods, labels["t"], labels["P_t"],
               methods, x_range=(-20.0, 40.0), use_fill=False)
