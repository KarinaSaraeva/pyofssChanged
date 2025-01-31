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

import warnings
import numpy as np
from scipy import linalg

# from tqdm import tqdm
from pyofss.field import temporal_power

from .storage import Storage
from .solver import Solver

# Define exceptions


class StepperError(Exception):
    pass


class SmallStepSizeError(StepperError):
    pass


class SuitableStepSizeError(StepperError):
    pass


class MaximumStepsAllocatedError(StepperError):
    pass


class Stepper(object):
    """
    :param Uint traces: Number of ouput trace to use
    :param double local_error: Relative local error required in adaptive method
    :param string method: ODE solver method to use
    :param object f: Derivative function to be solved
    :param double length: Length to integrate over
    :param Uint total_steps: Number of steps to use for ODE integration
    :param string dir: directory to save all the storage
    :param string save_represent: "power" or "complex" type of dataframe saved
    :param cycle string: cycle name in dataframe
    :param fibre_name: fibre name in dataframe

    method:
      * EULER -- Euler method;
      * MIDPOINT -- Midpoint method;
      * RK4 -- Fourth order Runge-Kutta method;
      * BS -- Bogacki-Shampine method;
      * RKF -- Runge-Kutta-Fehlberg method;
      * CK -- Cash-Karp method;
      * DP -- Dormand-Prince method;
      * SS_SIMPLE -- Simple split-step method;
      * SS_SYMMETRIC -- Symmetric split-step method;
      * SS_REDUCED -- Reduced split-step method;
      * SS_AGRAWAL -- Agrawal (iterative) split-step method;
      * SS_SYM_MIDPOINT -- Symmetric split-step method (MIDPOINT for nonlinear)
      * SS_SYM_RK4 -- Symmetric split-step method (RK4 for nonlinear);
      * RK4IP -- Runge-Kutta in the interaction picture method.

      Each method may use an adaptive stepper by prepending an 'A' to the name.

    traces:
      * 0 -- Store A for each succesful step;
      * 1 -- Store A at final value (length) only;
      * >1 -- Store A for each succesful step then use interpolation to get A
         values for equally spaced z-values, calculated using traces.
    """

    def __init__(
        self,
        traces=1,
        local_error=1.0e-6,
        method="RK4",
        f=None,
        f_characts=None,
        length=1.0,
        total_steps=100,
        dir=None,
        save_represent="power",
        cycle=None,
        fibre_name=None,
        downsampling=None,
    ):
        self.traces = traces
        self.local_error = local_error
        try:
            self.save_df = getattr(self, "save_df_" + save_represent)
        except AttributeError:
            print("No such method: save_represent should be either 'complex' or 'power'")
        self.cycle = cycle
        self.fibre_name = fibre_name
        # Check if adaptive stepsize is required:
        if method.upper().startswith("A"):
            self.adaptive = True
            self.method = method[1:]
        else:
            self.adaptive = False
            self.method = method

        # ~print "Using {0} method".format( self.method )

        # Delegate method and function to solver
        self.solver = Solver(self.method, f)
        self.step = self.solver

        self.length = length
        self.total_steps = total_steps

        # Use a list of tuples ( z, A(z) ) for dense output if required:
        self.storage = Storage(
            dir, cycle=self.cycle, fibre_name=self.fibre_name, f=f_characts, downsampling=downsampling
        )

        # Store constants for adaptive method:
        self.total_attempts = 1000
        self.steps_max = 100000
        self.step_size_min = 1e-37  # some small value

        self.safety = 0.9
        self.max_factor = 10.0
        self.min_factor = 0.2

        # Store local error of method:
        if self.method.lower() != "step_amplifier":
            self.eta = self.solver.errors[self.method.lower()]

        self.A_out = None

    def __call__(self, A, refrence_length):
        """Delegate to appropriate function, adaptive- or standard-stepper"""

        self.storage.reset_fft_counter()
        self.storage.reset_array()

        if self.adaptive:
            return self.adaptive_stepper(A, refrence_length)
        else:
            return self.standard_stepper(A, refrence_length)

    def save_df_power(self):
        self.storage.save_all_storage_to_dir_as_df(save_power=True)

    def save_df_complex(self):
        self.storage.save_all_storage_to_dir_as_df(save_power=False)

    def standard_stepper(self, A, refrence_length):
        """Take a fixed number of steps, each of equal length"""
        # ~print( "Starting ODE integration with fixed step-size... " ),

        # Initialise:
        self.A_out = A

        # Require an initial step-size:
        h = self.length / self.total_steps
        if h > refrence_length * (10 ** (-2)):
            warnings.warn(
                f"{self.cycle}: {self.fibre_name}: h must be much less than dispersion length (L_D) and the nonlinear length (L_NL)\n        \
                now the minimum of the characteristic distances is equal to {refrence_length:.6f}*km* \n         \
                step is equal to {h}*km*"
            )

        # Construct mesh points for z:
        zs = np.linspace(0.0, self.length, self.total_steps + 1)

        storage_step = int(self.total_steps / self.traces)

        # Construct mesh points for traces:
        if self.traces != self.total_steps:
            trace_zs = np.linspace(0.0, self.length, self.traces + 1)

        # Make sure to store the initial A if more than one trace is required:

        # Start at z = 0.0 and repeat until z = length - h (inclusive),
        # i.e. z[-1]

        # for i, z in enumerate(tqdm(zs[:-1])):
        for i, z in enumerate(zs[:-1]):
            # Currently at L = z

            if self.solver.embedded:
                self.A_out, A_other = self.step(self.A_out, z, h)
            else:
                self.A_out = self.step(self.A_out, z, h)
            # Now at L = z + h

            # If multiple traces required, store A_out at each relavant z
            # value:
            if self.traces != 1:
                # MODIFIED: If multiple traces required, store A_out at z only IF needed
                if i % storage_step == 0:
                    self.storage.append(z + h, self.A_out)

        # Store total number of fft and ifft operations that were used:
        self.storage.store_current_fft_count()

        # Need to interpolate dense output to grid points set by traces:

        # MODIFIED: no longer needed
        # if self.traces > 1 and (self.traces != self.total_steps):
        #     self.storage.interpolate_As_for_z_values(trace_zs)

        self.save_df()

        return self.A_out

    def save_power(self):
        self.storage.save_all_storage_to_dir_as_df(save_power=True)

    def save_complex_field(self):
        self.storage.save_all_storage_to_dir_as_df(save_power=False)

    @staticmethod
    def relative_local_error(A_fine, A_coarse):
        """Calculate an estimate of the relative local error"""

        # Large step can result in infs or NaNs values
        # so, check it first
        if np.isnan(A_fine).any() or np.isinf(A_fine).any():
            return 1.0

        norm_fine = linalg.norm(A_fine)

        # Avoid possible divide by zero:
        if norm_fine != 0.0:
            return linalg.norm(A_fine - A_coarse) / norm_fine
        else:
            return linalg.norm(A_fine - A_coarse)

    def adaptive_stepper(self, A, refrence_length):
        """Take multiple steps, with variable length, until target reached"""

        print("Starting ODE integration with adaptive step-size... ")

        # Initialise:
        self.A_out = A
        z = 0.0
        # Require an initial step-size which will be adapted by the routine:
        if self.traces > self.total_steps:
            h = self.length / self.traces
        else:
            h = self.length / self.total_steps

        # if (h > 0.01*refrence_length):
        #     h = 0.01*refrence_length
        if h > 1e-6:
            h = 1e-6

        print(f"initial step size equals {h}")
        total_amount_of_steps = 0
        step_storage = self.length / self.traces
        z_ignore = step_storage
        # Constants used for approximation of solution using local
        # extrapolation:
        f_eta = np.power(2, self.eta - 1.0)
        f_alpha = f_eta / (f_eta - 1.0)
        f_beta = 1.0 / (f_eta - 1.0)

        hmin = h
        # Calculate z-values at which to save traces.
        if self.traces > 1:
            # zs contains z-values for each trace, as well as the initial
            # trace:
            zs = np.linspace(0.0, self.length, self.traces + 1)

        # Store initial trace:
        if self.traces != 1:
            self.storage.append(z, self.A_out)

        # Limit the number of steps in case of slowly converging runs:
        # for s in tqdm(range(1, self.steps_max)):
        for s in range(1, self.steps_max):
            # If step-size takes z our of range [0.0, length], then correct it:
            if (z + h) > self.length:
                h = self.length - z

            # Take an adaptive step:
            for ta in range(0, self.total_attempts):
                h_half = 0.5 * h
                z_half = z + h_half

                # Calculate A_fine and A_coarse internally if using an
                # embedded method. Otherwise use method of step-doubling:
                if self.solver.embedded:
                    A_fine, A_coarse = self.step(self.A_out, z, h)
                else:
                    # Calculate fine solution using two steps of size h_half:
                    A_half = self.step(self.A_out, z, h_half)
                    A_fine = self.step(A_half, z_half, h_half)
                    # Calculate coarse solution using one step of size h:
                    A_coarse = self.step(self.A_out, z, h)

                # Calculate an estimate of relative local error:
                delta = self.relative_local_error(A_fine, A_coarse)

                # Store current stepsize:
                h_temp = h
                if hmin > h:
                    hmin = h
                # Adjust stepsize for next step:
                if delta > 0.0:
                    error_ratio = self.local_error / delta
                    factor = self.safety * np.power(error_ratio, 1.0 / self.eta)
                    h = h_temp * min(self.max_factor, max(self.min_factor, factor))
                else:
                    # Error approximately zero, so use largest stepsize
                    # increase:
                    h = h_temp * self.max_factor

                if delta < 2.0 * self.local_error:
                    # Successful step, so increment z h_temp (which is the
                    # stepsize that was used for this step):
                    z += h_temp

                    if self.solver.embedded:
                        # Accept the higher order method:
                        self.A_out = A_fine
                    else:
                        # Use local extrapolation to form a higher order
                        # solution:
                        self.A_out = f_alpha * A_fine - f_beta * A_coarse

                    # Store data on current z and stepsize used for each
                    # succesful step:
                    if z > z_ignore:
                        self.storage.step_sizes.append((z, h_temp))
                        self.storage.append(z, self.A_out)
                        z_ignore += step_storage
                    break  # Successful attempt at step, move on to next step.
                # Otherwise error was too large, continue with next attempt,
                # but check the minimal step size first
                else:
                    if h < self.step_size_min:
                        raise SmallStepSizeError("Step size is extremely small")

            else:
                raise SuitableStepSizeError("Failed to set suitable step-size")

            # If the desired z has been reached, then finish:
            if z >= self.length:
                # Store total number of fft and ifft operations that were used:
                self.storage.store_current_fft_count()

                # MODIFIED: no longer needed
                # Interpolate dense output to uniformly-spaced z values:
                # if self.traces > 1:
                #     self.storage.interpolate_As_for_z_values(zs)
                if self.storage.dir_spec and self.storage.dir_temp:
                    if self.save_represent == "power":
                        self.storage.save_all_storage_to_dir_as_df(save_power=True)
                    elif self.save_represent == "complex":
                        self.storage.save_all_storage_to_dir_as_df(save_power=False)
                    else:
                        print(f"flag should be one of these: 'power', 'complex'")
                return self.A_out

            total_amount_of_steps = total_amount_of_steps + 1

        raise MaximumStepsAllocatedError("Failed to complete with maximum steps allocated")


if __name__ == "__main__":
    """
    Exact solution: A(z) = 0.5 * ( 5.0 * exp(-2.0 * z) - 3.0 * exp(-4.0 * z) )
    A(0) = 1.0
    A(0.5) = 0.71669567807368684
    Numerical solution (RK4, total_steps = 5):      0.71668876283331295
    Numerical solution (RK4, total_steps = 50):     0.71669567757603803
    Numerical solution (RK4, total_steps = 500):    0.71669567807363854
    Numerical solution (RKF, total_steps = 5):      0.71669606109336026
    Numerical solution (RKF, total_steps = 50):     0.71669567807672185
    Numerical solution (RKF, total_steps = 500):    0.71669567807368773
    """
    import matplotlib.pyplot as plt

    def simple(A, z):
        """Just a simple function."""
        return 3.0 * np.exp(-4.0 * z) - 2.0 * A

    stepper = Stepper(f=simple, length=0.5, total_steps=50, method="RKF", traces=50)
    A_in = 1.0
    A_out = stepper(A_in)
    print("A_out = %.17f" % (A_out))

    x = stepper.storage.z
    y = stepper.storage.As

    title = r"$\frac{dA}{dz} + 2A = 3 e^{-4z}$"
    plt.title(r"Numerical integration of ODE:" + title)
    plt.xlabel("z")
    plt.ylabel("A(z)")
    plt.plot(x, y, label="RKF: 50 steps")
    plt.legend()
    plt.show()
