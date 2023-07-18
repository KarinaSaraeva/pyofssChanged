"""
    Copyright (C) 2013 David Bolt,
    2020-2021 Vladislav Efremov, Denis Kharenko

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
import pyopencl as cl
import pyopencl.array as cl_array
from pynvml import *
import gc
import pandas as pd
from pyofss.field import fft, ifft, fftshift
import sys as sys0
version_py = sys0.version_info[0]
if version_py == 3:
    from reikna import cluda
    from reikna.fft import FFT
else:
    from pyfft.cl import Plan  
from string import Template

from .linearity import Linearity
from .nonlinearity import Nonlinearity
from pyofss.modules.amplifier import Amplifier, Amplifier2LevelModel
from scipy.signal import find_peaks, peak_widths
import warnings

class FiberInitError(Exception):
    pass

OPENCL_OPERATIONS = Template("""
    #ifdef cl_arm_printf
        #pragma OPENCL EXTENSION cl_amd_printf: enable
    #endif
    #ifdef cl_khr_fp64 // Khronos extension
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #define PYOPENCL_DEFINE_CDOUBLE
    #elif defined(cl_amd_fp64) // AMD extension
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #define PYOPENCL_DEFINE_CDOUBLE
    #else
        #warning "Double precision floating point is not supported"
    #endif

    #include <pyopencl-complex.h>


    __kernel void cl_cache(__global c${dorf}_t* factor,
                           const ${dorf} stepsize) {
        int gid = get_global_id(0);

        factor[gid] = c${dorf}_exp(c${dorf}_mulr(factor[gid], stepsize));
    }

    __kernel void cl_linear_cached(__global c${dorf}_t* field,
                                   __global c${dorf}_t* factor) {
        int gid = get_global_id(0);

        field[gid] = c${dorf}_mul(field[gid], factor[gid]);
    }

    __kernel void cl_linear_cached_with_amplification(__global c${dorf}_t* field,
                                   __global c${dorf}_t* factor, __global ${dorf}* amp_factor) {
        int gid = get_global_id(0);
        c${dorf}_t local_factor = factor[gid];

        local_factor.real = local_factor.real*amp_factor[gid];
        local_factor.imag = local_factor.imag*amp_factor[gid];
        field[gid] = c${dorf}_mul(field[gid], local_factor);
    }

    __kernel void cl_linear(__global c${dorf}_t* field,
                            __global c${dorf}_t* factor,
                            const ${dorf} stepsize) {
        int gid = get_global_id(0);

        // IMPORTANT: Cannot just multiply two complex numbers!
        // Must use the appropriate function (e.g. cfloat_mul).
        field[gid] = c${dorf}_mul(field[gid],
                                  c${dorf}_exp(c${dorf}_mulr(factor[gid], stepsize)));
    }

    c${dorf}_t cl_square_abs(const c${dorf}_t element) {
        return c${dorf}_mul(element, c${dorf}_conj(element));
    }

    __kernel void cl_square_abs2(__global c${dorf}_t* field) {
        int gid = get_global_id(0);
        field[gid] = c${dorf}_new(c${dorf}_abs_squared(field[gid]), (${dorf})0.0f);
    }

    __kernel void cl_nonlinear(__global c${dorf}_t* field,
                               const ${dorf} gamma,
                               const ${dorf} stepsize) {
        int gid = get_global_id(0);

        const c${dorf}_t im_gamma = c${dorf}_new((${dorf})0.0f, stepsize * gamma);

        field[gid] = c${dorf}_mul(
            field[gid], c${dorf}_mul(im_gamma, cl_square_abs(field[gid])));
    }
    
    __kernel void cl_nonlinear_exp(__global c${dorf}_t* fieldA,
                               const __global c${dorf}_t* fieldB,
                               const ${dorf} gamma,
                               const ${dorf} stepsize) {
        int gid = get_global_id(0);

        const c${dorf}_t im_gamma = c${dorf}_new((${dorf})0.0f, stepsize * gamma);

        fieldA[gid] = c${dorf}_mul(
            fieldB[gid], c${dorf}_exp(
                c${dorf}_mul(im_gamma, cl_square_abs(fieldA[gid]))));
    }

    __kernel void cl_sum(__global c${dorf}_t* first_field,
                         const ${dorf} first_factor,
                         __global c${dorf}_t* second_field,
                         const ${dorf} second_factor) {
        int gid = get_global_id(0);

        first_field[gid] = c${dorf}_mulr(first_field[gid], first_factor);
        first_field[gid] = c${dorf}_add(first_field[gid], c${dorf}_mulr(second_field[gid], second_factor));
    }

    __kernel void cl_mul(__global c${dorf}_t* field,
                         __global c${dorf}_t* factor) {
        int gid = get_global_id(0);

        field[gid] = c${dorf}_mul(field[gid], factor[gid]);
    }

    __kernel void cl_nonlinear_with_all_st1(__global c${dorf}_t* field,
                                          __global c${dorf}_t* field_mod,
                                          __global c${dorf}_t* conv,
                                          const ${dorf} fR,
                                          const ${dorf} fR_inv) {
        int gid = get_global_id(0);

        conv[gid] = c${dorf}_new(c${dorf}_real(conv[gid]), (${dorf})0.0f);

        field[gid] = c${dorf}_add(
            c${dorf}_mul(c${dorf}_mulr(field[gid], fR_inv), field_mod[gid]),
                c${dorf}_mul(c${dorf}_mulr(field[gid], fR), conv[gid]));

    }

    __kernel void cl_nonlinear_with_all_st2_with_ss(__global c${dorf}_t* field,
                                __global c${dorf}_t* factor,
                                const ${dorf} gamma) {
        int gid = get_global_id(0);

        const c${dorf}_t im_gamma = c${dorf}_new((${dorf})0.0f, gamma);

        field[gid] = c${dorf}_mul(
            im_gamma, c${dorf}_mul(
                field[gid], factor[gid]));
    }

    __kernel void cl_step_mul(__global c${dorf}_t* field,
                                const ${dorf} stepsize) {
        int gid = get_global_id(0);

        field[gid] = c${dorf}_mulr(
            field[gid], stepsize);
    }

    __kernel void cl_nonlinear_with_all_st2_without_ss(__global c${dorf}_t* field,
                                const ${dorf} gamma) {
        int gid = get_global_id(0);

        const c${dorf}_t im_gamma = c${dorf}_new((${dorf})0.0f, gamma);

        field[gid] = c${dorf}_mul(
            im_gamma, field[gid]);
    }

    __kernel void cl_power(__global c${dorf}_t* field,
                                __global ${dorf}* power_buffer) {
        int gid = get_global_id(0);
    
        c${dorf}_t c = field[gid];
        power_buffer[gid] = c.real * c.real + c.imag * c.imag;
    }

    kernel void cl_interpolate(__global ${dorf}* input, const ${dorf} n_input, __global ${dorf}* output, const ${dorf} n_output) {
        int i = get_global_id (0); // index of output element
        float t = (float)i / ((float)n_output); // normalized coordinate in [0, 1]
        int j = (int)(t * (n_input - 1)); // lower index of input element
        ${dorf} u = t * (n_input - 1) - j; // fractional part of input element
        ${dorf} v0 = input[j]; // lower input value
        ${dorf} v1 = input[j + 1]; // upper input value
        ${dorf} v = mix(v0, v1, u); // linear interpolation
        output[i] = v; // store interpolated value
    }

    __kernel void cl_physical_power(__global c${dorf}_t* field, __global const ${dorf}* factor,
                                __global ${dorf}* power_buffer) {
        int gid = get_global_id(0);
    
        c${dorf}_t c = field[gid];
        power_buffer[gid] = (c.real * c.real + c.imag * c.imag)*factor[0];
    }
    
    __kernel void cl_simpson(__global ${dorf}* power_buffer, __global ${dorf}* output, const ${dorf} h) {
        int gid = get_global_id(0);  
        if (!(gid % 2)) {
            ${dorf} x0 = power_buffer[gid];
            ${dorf} x1 = power_buffer[gid + 1];
            ${dorf} x2 = power_buffer[gid + 2];
            
            ${dorf} result = (x0 + (${dorf})4.0f * x1 + x2);
            output[gid] = result * h / ((${dorf})3.0f) ;
        }    
    }

    __kernel void cl_calculate_g_s_exponent_const(__global ${dorf}* g_s, __global const ${dorf}* alpha_s, __global const ${dorf}* eta_s, const ${dorf} N2, const ${dorf} h) {
        int gid = get_global_id(0);
        g_s[gid] = exp((alpha_s[gid] * N2 - eta_s[gid])*h);
    }

    __kernel void multiply_array_by_another_const(__global ${dorf}* array1,
                            __global const ${dorf}* array2) {
        int gid = get_global_id(0);
        array1[gid] = array1[gid]*array2[gid];
    }
    
    __kernel void devide_array_by_another_const(__global ${dorf}* array1, __global const ${dorf}* array2) {
        int gid = get_global_id(0);
        array1[gid] = array1[gid]/array2[gid]; 
    }

    __kernel void cl_fftshift(__global ${dorf}* input, const ${dorf} n) {
        int half_n = (int)(n / 2);
        int gid = get_global_id(0);
        if (gid < half_n)
        {
            ${dorf} tmp = input[gid];
            input[gid] = input[gid + half_n];
            input[gid + half_n] = tmp;
        }
    }

""")

class OpenclProgramm(object):
    """" base program to work with GPU devices Openclfibre mus be initialised with this object """
    def __init__(self, name="cl_programm", dorf='double', ctx=None, fast_math=False, use_all=False, downsampling=500):
        self.cached_factor = False
        self.use_all = use_all
        self.dorf = dorf
        self.downsampling = downsampling

        self.queue = None
        self.np_float = None
        self.np_complex = None
        self.prg = None
        self.compiler_options = None
        self.fast_math = fast_math
        self.ctx = ctx
        self.cl_initialise()

        self.plan = None

        self.buf_field = None
        self.buf_temp = None
        self.buf_interaction = None
        self.buf_factor = None

        self.buf_mod = None
        self.buf_conv = None
        self.buf_h_R = None
        self.buf_nn_factor = None

        self.shape = None

        self.power_buffer = None
        self.downsampled_power_buffer = None
        self.simpson_result = None

    def cl_initialise(self):
        """ Initialise opencl related parameters. """
        float_conversions = {"float": np.float32, "double": np.float64}
        complex_conversions = {"float": np.complex64, "double": np.complex128}

        self.np_float = float_conversions[self.dorf]
        self.np_complex = complex_conversions[self.dorf]

        for platform in cl.get_platforms():
            if platform.name == "NVIDIA CUDA" and self.fast_math is True:
                print("Using compiler optimisations suitable for Nvidia GPUs")
                self.compiler_options = ["-cl-mad-enable", "-cl-fast-relaxed-math"]
            else:
                self.compiler_options = ""
        
        if self.ctx is None:
            self.ctx = cl.create_some_context()

        self.queue = cl.CommandQueue(self.ctx)
        if version_py == 3:
            api = cluda.ocl_api()
            self.thr = api.Thread(self.queue)

        substitutions = {"dorf": self.dorf}
        code = OPENCL_OPERATIONS.substitute(substitutions)
        self.prg = cl.Program(self.ctx, code).build(options=self.compiler_options)

    def set_domain(self, domain):
        if self.plan is None:
            if version_py == 3:
                self.plan = FFT(domain.t.astype(self.np_complex)).compile(self.thr, 
                        fast_math=self.fast_math, compiler_options=self.compiler_options)
                self.plan.execute = self.reikna_fft_execute
            else:
                self.plan = Plan(domain.total_samples, queue=self.queue, dtype=self.np_complex, fast_math=self.fast_math)

    def send_arrays_to_device(self, field, factor, h_R, nn_factor):
        """ Move numpy arrays onto compute device. """
        self.shape = field.shape

        self.buf_field = cl_array.to_device(
            self.queue, field.astype(self.np_complex))

        if self.buf_temp is None:
            self.buf_temp = cl_array.empty_like(self.buf_field)
        if self.buf_interaction is None:
            self.buf_interaction = cl_array.empty_like(self.buf_field)

        if self.power_buffer is None:
            self.power_buffer = cl_array.zeros(self.queue, self.shape, self.np_float)
        if self.downsampled_power_buffer is None:
            self.downsampled_power_buffer = cl_array.zeros(self.queue, self.downsampling, self.np_float)
        if self.simpson_result is None:
            self.simpson_result = cl_array.zeros(self.queue, self.shape, self.np_float)

        if self.use_all:
            if self.buf_h_R is None:
                self.buf_h_R = cl_array.to_device(
                                        self.queue, h_R.astype(self.np_complex))
            if self.buf_nn_factor is None and nn_factor is not None:
                self.buf_nn_factor = cl_array.to_device(
                                        self.queue, nn_factor.astype(self.np_complex))
            if self.buf_mod is None:
                self.buf_mod = cl_array.empty_like(self.buf_field)
            if self.buf_conv is None:
                self.buf_conv = cl_array.empty_like(self.buf_field)

        if self.cached_factor is False:
            self.buf_factor = cl_array.to_device(
                self.queue, factor.astype(self.np_complex))
            
    def reikna_fft_execute(self, d, inverse=False):
        self.plan(d,d,inverse=inverse)


def get_device_memory_info():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"device name {nvmlDeviceGetName(handle)}")
    # print("Total memory: {} MiB".format(info.total >> 20))
    # print("Free memory: {} MiB".format(info.free >> 20))
    print("Used memory: {} MiB".format(info.used >> 20))
    nvmlShutdown()

class OpenclFibre(object):
    """
    This optical module is similar to Fibre, but uses PyOpenCl (Python
    bindings around OpenCL) to generate parallelised code.
    method:
        * cl_rk4ip
        * cl_ss_symmetric
        * cl_ss_sym_rk4
    """
    def __init__(self, cl_programm, name="cl_fibre", length=1.0, alpha=None,
                 beta=None, gamma=0.0, method="cl_ss_symmetric", total_steps=100,
                 self_steepening=False, centre_omega=None,
                 tau_1=12.2e-3, tau_2=32.0e-3, f_R=0.18,
                 small_signal_gain=None, E_sat=None, lamb0=None, bandwidth=None, 
                 use_Yb_model=False, Pp_0 = None, N = None, Rr=None, save_represent="power", cycle='cycle0', traces=None, dir=None, amplifier=None):
        
        self.cl_programm = cl_programm

        self.dir = dir

        self.name = name
        self.cycle = cycle
        self.domain = None

        self.gamma = gamma
        self.beta_2 = beta[2] if beta is not None else None
        
        self.f_R = f_R
        self.f_R_inv = 1.0 - f_R

        self.nn_factor = None
        self.h_R = None
        self.omega = None
        self.ss_factor = None            
        
        # Force usage of cached version of function:
        self.cl_linear = self.cl_linear_cached

        self.length = length
        self.total_steps = total_steps
        self.traces = traces if traces is not None else total_steps

        self.stepsize = self.length / self.total_steps
        self.zs = np.linspace(0.0, self.length, self.total_steps + 1)

        self.method = getattr(self, method.lower())

        if self.use_all:
            if self_steepening is False:
                self.cl_n = getattr(self, 'cl_n_with_all')
            else:
                self.cl_n = getattr(self, 'cl_n_with_all_and_ss')
        elif self_steepening:
            raise NotImplementedError("Self-steepening without general nonlinearity is not implemented")
        else:
            self.cl_n = getattr(self, 'cl_n_default')
        

        if amplifier is not None:
            self.amplifier = amplifier
        else:
            if use_Yb_model:
                print('Use Yb model')
                self.amplifier = Amplifier2LevelModel(Pp=Pp_0, N=N, Rr=Rr, prg=self.prg, queue=self.queue, ctx=self.ctx, dorf=self.dorf)
            else:
                if (small_signal_gain is not None) and (E_sat is not None):
                    print('Use simple saturation model')
                    self.amplifier = Amplifier(
                        gain=small_signal_gain, E_sat=E_sat, length=self.length, lamb0=lamb0, bandwidth=bandwidth, steps=total_steps)
                elif (small_signal_gain is None) and (E_sat is None):
                    print('Passive fibre modulation')
                    self.amplifier = None
                else:
                    assert(FiberInitError(
                        'Not enought parameters to initialise amplification fiber: both small_signal_gain and E_sat must be passed!'))
                
        self.linearity = Linearity(alpha, beta, sim_type="default",
                                    use_cache=True, centre_omega=centre_omega, phase_lim=True, amplifier=self.amplifier)
        self.nonlinearity = Nonlinearity(gamma, None, self_steepening,
                                         False, 0,
                                         self.use_all, tau_1, tau_2, f_R)
        
        # TODO: create local storage here too as it is done in fibre class

        self.factor = None
        self.energy_list = []
        self.max_power_list = []
        self.peaks_list = []

        self.z_list = []
        self.temp_field_list = []
        self.spec_field_list = []
        # self.complex_field_list = []

    @property
    def dorf(self):    
        return self.cl_programm.dorf

    @property
    def use_all(self):    
        return self.cl_programm.use_all

    @property
    def queue(self):    
        return self.cl_programm.queue

    @property
    def np_float(self):
        return self.cl_programm.np_float

    @property
    def np_complex(self):
        return self.cl_programm.np_complex

    @property
    def prg(self):
        return self.cl_programm.prg

    @property
    def compiler_options(self):
        return self.cl_programm.compiler_options

    @property
    def fast_math(self):    
        return self.cl_programm.fast_math

    @property
    def ctx(self):
        return self.cl_programm.ctx

    @property
    def plan(self):
        return self.cl_programm.plan

    @property
    def buf_field(self):
        return self.cl_programm.buf_field

    @property
    def buf_temp(self):
        return self.cl_programm.buf_temp

    @property
    def buf_interaction(self):
        return self.cl_programm.buf_interaction

    @property
    def buf_factor(self):
        return self.cl_programm.buf_factor

    @property
    def buf_mod(self):
        return self.cl_programm.buf_mod

    @property
    def buf_conv(self):
        return self.cl_programm.buf_conv

    @property
    def shape(self):
        return self.cl_programm.shape

    @property
    def power_buffer(self):
        return self.cl_programm.power_buffer

    @property
    def downsampled_power_buffer(self):
        return self.cl_programm.downsampled_power_buffer

    @property
    def simpson_result(self):
        return self.cl_programm.simpson_result
    
    @property
    def cached_factor(self):
        return self.cl_programm.cached_factor
    
    @buf_factor.setter
    def buf_factor(self, value):
        self.cl_programm.buf_factor = value

    @cached_factor.setter
    def cached_factor(self, value):
        self.cl_programm.cached_factor = value

    def __call__(self, domain, field):
        # Setup plan for calculating fast Fourier transforms:
        get_device_memory_info()
        self.calculate_refrence_length(domain, field)
        if self.stepsize > self.refrence_length * (10 ** (-2)):
            warnings.warn(
                f"{self.cycle}: {self.name}: h must be much less than dispersion length (L_D) and the nonlinear length (L_NL)\n        \
                now the minimum of the characteristic distances is equal to {self.refrence_length:.6f}*km* \n         \
                step is equal to {self.stepsize}*km*"
            )

        if self.domain != domain:
            self.domain = domain
            if self.use_all:
                self.nonlinearity(self.domain)
                self.omega = self.nonlinearity.omega
                self.h_R = self.nonlinearity.h_R
                self.ss_factor = self.nonlinearity.ss_factor
                if self.ss_factor != 0.0:
                    self.nn_factor = 1.0 + self.omega * self.ss_factor
                else:
                    self.nn_factor = None

        if self.factor is None:
            self.factor = self.linearity(domain)

        self.cl_programm.send_arrays_to_device(field, self.factor, self.h_R, self.nn_factor)

        storage_step = int(self.total_steps / self.traces)      

        for i in range(len(self.zs[1:])):
            self.method(self.buf_field, self.buf_temp,
                          self.buf_interaction, self.buf_factor, self.stepsize)
            
            # Storage part
            if (i % storage_step == 0):
                self.compute_characts(self.buf_field)
                self.prg.cl_power(self.queue, self.shape, None, self.buf_field.data, self.power_buffer.data)
                self.prg.cl_interpolate(self.queue, tuple([self.downsampled_power_buffer.shape[0]]), None, self.power_buffer.data, self.np_float(self.power_buffer.shape[0]), self.downsampled_power_buffer.data, self.np_float(self.downsampled_power_buffer.shape[0]))
                self.temp_field_list.append(self.downsampled_power_buffer.get())

                self.plan.execute(self.buf_field.data, inverse=True)
                self.prg.cl_power(self.queue, self.shape, None, self.buf_field.data, self.power_buffer.data)
                self.plan.execute(self.buf_field.data)
                self.prg.cl_fftshift(self.queue, self.shape, None, self.power_buffer.data, self.np_float(self.power_buffer.shape[0]))

                self.prg.cl_interpolate(self.queue, tuple([self.downsampled_power_buffer.shape[0]]), None, self.power_buffer.data, self.np_float(self.power_buffer.shape[0]), self.downsampled_power_buffer.data, self.np_float(self.downsampled_power_buffer.shape[0]))
                self.spec_field_list.append(self.downsampled_power_buffer.get())

                # TODO: add saving complex field
                # self.complex_field_list.append(self.buf_field.get())
                self.z_list.append(self.zs[i+1])
            
        # gpu memory should be cleared here manualy all needed info is already loaded
        # self.clear_arrays_on_device()
        return self.buf_field.get()
    
    def calculate_refrence_length(self, domain, field):
        temp_power = abs(field) ** 2
        d_t = abs(domain.t[1] - domain.t[0])
        P_0 = np.amax(temp_power)
        peaks, _ = find_peaks(temp_power, height=0)
        results_half = peak_widths(temp_power, peaks, rel_height=1)
        T_0 = d_t * np.amax(results_half[0])
        if (self.beta_2 is not None):
            self.L_D = T_0**2 / (10**3 * self.beta_2)
        if (self.gamma is not None):
            self.L_NL = 1 / (self.gamma * P_0)
        if (self.L_NL and self.L_D is not None):
            self.refrence_length = min(self.L_NL, self.L_D)
        elif (self.L_NL or self.L_D is None):
            self.refrence_length = None
        else:
            self.refrence_length = self.L_NL if self.L_D is None else self.L_D

    # for usual fibre this info is in storage
    def get_df(self, type = "complex", z_curr=0, channel=None):
        if type == "temp":
            y = self.temp_field_list
        elif type == "spec":
            y = self.spec_field_list   
        elif type == "complex":
            warnings.warn("Saving downsampled complex power has not been made yet!")
        else:
            raise ValueError()
         
        arr_z = np.array(self.z_list)*10**6 + z_curr # mm
        if self.cycle and self.name is not None:
            iterables = [[self.cycle], [self.name], arr_z]
            index = pd.MultiIndex.from_product(
                iterables,  names=["cycle", "fibre", "z [mm]"])
        else:
            iterables = [arr_z]
            index = pd.MultiIndex.from_product(iterables, names=["z [mm]"])
        return pd.DataFrame(y, index=index)           

    # for usual fibre this info is in storage
    def get_df_result(
        self,
        z_curr=0,
    ):
        z = np.linspace(0.0, self.length, self.traces + 1)[1:]

        arr_z = np.array(z)*10**6 + z_curr
        characteristic = ["max_value", "energy", "duration", "spec_width", "peaks"]
        if self.cycle and self.name is not None:
            iterables = [[self.cycle], [self.name], arr_z]
            index = pd.MultiIndex.from_product(
                iterables,  names=["cycle", "fibre", "z [mm]"])
        else:
            iterables = [arr_z]
            index = pd.MultiIndex.from_product(iterables, names=["z [mm]"])

        df_results = pd.DataFrame(index=index, columns=characteristic)
        df_results["max_value"] = self.max_power_list
        df_results["energy"] = self.energy_list
        # duration and spec width are not calculated in OpenclFibre
        # df_results["duration"] = self.duration_list   
        # df_results["spec_width"] = self.spec_width_list
        # df_results["peaks"] = self.peaks_list
        return df_results

    @staticmethod
    def print_device_info():
        """ Output information on each OpenCL platform and device. """
        for platform in cl.get_platforms():
            print("=" * 60)
            print("Platform information:")
            print("Name: ", platform.name)
            print("Profile: ", platform.profile)
            print("Vender: ", platform.vendor)
            print("Version: ", platform.version)

            for device in platform.get_devices():
                print("-" * 60)
                print("Device information:")
                print("Name: ", device.name)
                print("Type: ", cl.device_type.to_string(device.type))
                print("Memory: ", device.global_mem_size // (1024 ** 2), "MB")
                print("Max clock speed: ", device.max_clock_frequency, "MHz")
                print("Compute units: ", device.max_compute_units)
                if "NVIDIA" in device.vendor:
                    free_mem = (device.global_mem_size - cl.Device.get_info(device, cl.device_info.GLOBAL_MEM_SIZE)) // (1024 ** 2)
                else:
                    free_mem = device.get_info(cl.device_info.GLOBAL_FREE_MEMORY_AMD) // (1024 ** 2)
                print("Free global memory:", free_mem, "MB")
                
            print("=" * 60)
            
    def cl_clear(self, cl_arr):
        if cl_arr is not None:
            if cl_arr.size > 0:
                cl_arr.data.release()      

    def clear_arrays_on_device(self): 
        get_device_memory_info()
        if self.amplifier:
            self.amplifier.clear_arrays_on_device()    
            gc.collect()
            get_device_memory_info()
    
        
    def cl_copy(self, dst_buffer, src_buffer):
        """ Copy contents of one buffer into another. """
        cl.enqueue_copy(self.queue, dst_buffer.data, src_buffer.data).wait()

    def compute_characts(self, field):
        def get_peaks(P, prominence):
            peaks, _ = find_peaks(P, height=0, prominence=prominence)
            return peaks

        self.prg.cl_power(self.queue, self.shape, None, field.data, self.power_buffer.data)

        # Get the maximum value in the array
        max_power = cl.array.max(self.power_buffer, queue=self.queue)

        self.prg.cl_simpson(self.queue, tuple([self.shape[0] - 2]), None, self.power_buffer.data, self.simpson_result.data, self.np_float(self.domain.dt*1e-3))
        energy = cl.array.sum(self.simpson_result, queue=self.queue)
        self.energy_list.append(energy.get())
        self.max_power_list.append(max_power.get())
        #self.peaks_list.append(get_peaks(power_buffer_cpu, max_power/10)) # cant be paralleled

    def cl_linear(self, field_buffer, stepsize, factor_buffer):
        """ Linear part of step. """
        self.plan.execute(field_buffer.data, inverse=True)
        self.prg.cl_linear(self.queue, self.shape, None, field_buffer.data,
                           factor_buffer.data, self.np_float(stepsize)) # remake for amplification fibre
        self.plan.execute(field_buffer.data)

    def cl_linear_cached(self, field_buffer, stepsize, factor_buffer):
        """ Linear part of step (cached version). """
        if (self.cached_factor is False):
            self.linearity.cache(stepsize)
            self.buf_factor = cl_array.to_device(
                self.queue, self.linearity.cached_factor.astype(self.np_complex))
            self.cached_factor = True
        
        if self.amplifier is None:
            self.plan.execute(field_buffer.data, inverse=True)
            self.prg.cl_linear_cached(self.queue, self.shape, None,
                                  field_buffer.data, self.buf_factor.data)
        else:
            self.plan.execute(field_buffer.data, inverse=True)
            self.amplifier.cl_exp_factor(field_buffer, stepsize)
            self.prg.cl_fftshift(self.queue, self.shape, None, self.amplifier.g_s_buffer.data, self.np_float(self.amplifier.g_s_buffer.shape[0]))
            self.prg.cl_linear_cached_with_amplification(self.queue, self.shape, None,
                                  field_buffer.data, self.buf_factor.data, self.amplifier.g_s_buffer.data)

        self.plan.execute(field_buffer.data) 

    def cl_n_default(self, field_buffer, stepsize):
        """ Nonlinear part of step. """
        self.prg.cl_nonlinear(self.queue, self.shape, None, field_buffer.data,
                              self.np_float(self.gamma), self.np_float(stepsize))
    
    def cl_nonlinear(self, fieldA_buffer, stepsize, fieldB_buffer):
        """ Nonlinear part of step, exponential term"""
        self.prg.cl_nonlinear_exp(self.queue, self.shape, None, fieldA_buffer.data, fieldB_buffer.data,
                              self.np_float(self.gamma), self.np_float(stepsize))

    def cl_n_with_all_and_ss(self, field_buffer, stepsize):
        """ Nonlinear part of step with self_steepening and raman """
        # |A|^2
        self.cl_copy(self.buf_mod, field_buffer)
        self.prg.cl_square_abs2(self.queue, self.shape, None,
                                self.buf_mod.data)

        # conv = ifft(h_R*(fft(|A|^2)))
        self.cl_copy(self.buf_conv, self.buf_mod)
        self.plan.execute(self.buf_conv.data, inverse = True)
        self.prg.cl_mul(self.queue, self.shape, None,
                        self.buf_conv.data, self.buf_h_R.data)
        self.plan.execute(self.buf_conv.data)

        # p = fft( A*( (1-f_R)*|A|^2 + f_R*conv ) )
        self.prg.cl_nonlinear_with_all_st1(self.queue, self.shape, None,
                                         field_buffer.data, self.buf_mod.data, self.buf_conv.data,
                                         self.np_float(self.f_R), self.np_float(self.f_R_inv))
        self.plan.execute(field_buffer.data, inverse = True)

        # A_out = ifft( factor*(1 + omega*ss_factor)*p)
        self.prg.cl_nonlinear_with_all_st2_with_ss(self.queue, self.shape, None, field_buffer.data,
                              self.buf_nn_factor.data, self.np_float(self.gamma))
        self.plan.execute(field_buffer.data)

        self.prg.cl_step_mul(self.queue, self.shape, None,
                             field_buffer.data, self.np_float(stepsize))

    def cl_n_with_all(self, field_buffer, stepsize):
        """ Nonlinear part of step with self_steepening and raman """
        # |A|^2
        self.cl_copy(self.buf_mod, field_buffer)
        self.prg.cl_square_abs2(self.queue, self.shape, None,
                                self.buf_mod.data)

        # conv = ifft(h_R*(fft(|A|^2)))
        self.cl_copy(self.buf_conv, self.buf_mod)
        self.plan.execute(self.buf_conv.data, inverse = True)
        self.prg.cl_mul(self.queue, self.shape, None,
                        self.buf_conv.data, self.buf_h_R.data)
        self.plan.execute(self.buf_conv.data)

        # p = A*( (1-f_R)*|A|^2 + f_R*conv )
        self.prg.cl_nonlinear_with_all_st1(self.queue, self.shape, None,
                                         field_buffer.data, self.buf_mod.data, self.buf_conv.data,
                                         self.np_float(self.f_R), self.np_float(self.f_R_inv))

        # A_out = factor*p
        self.prg.cl_nonlinear_with_all_st2_without_ss(self.queue, self.shape, None, field_buffer.data,
                                                  self.np_float(self.gamma))

        self.prg.cl_step_mul(self.queue, self.shape, None,
                             field_buffer.data, self.np_float(stepsize))



    def cl_sum(self, first_buffer, first_factor, second_buffer, second_factor):
        """ Calculate weighted summation. """
        self.prg.cl_sum(self.queue, self.shape, None,
                        first_buffer.data, self.np_float(first_factor),
                        second_buffer.data, self.np_float(second_factor))
    
    def cl_ss_symmetric(self, field, field_temp, field_interaction, factor, stepsize):
        """ Symmetric split-step method using OpenCL"""
        half_step = 0.5 * stepsize
        
        self.cl_copy(field_temp, field)

        self.cl_linear(field_temp, half_step, factor)
        self.cl_nonlinear(field, stepsize, field_temp)
        self.cl_linear(field, half_step, factor)
    
    def cl_ss_sym_rk4(self, field, field_temp, field_linear, factor, stepsize):
        """ 
        Runge-Kutta fourth-order method using OpenCL.

            A_L = f.linear(A, hh)
            k0 = h * f.n(A_L, z)
            k1 = h * f.n(A_L + 0.5 * k0, z + hh)
            k2 = h * f.n(A_L + 0.5 * k1, z + hh)
            k3 = h * f.n(A_L + k2, z + h)
            A_N =  A_L + (k0 + 2.0 * (k1 + k2) + k3) / 6.0
            return f.linear(A_N, hh)
        """

        inv_six = 1.0 / 6.0
        inv_three = 1.0 / 3.0
        half_step = 0.5 * stepsize

        self.cl_linear(field, half_step, factor) #A_L

        self.cl_copy(field_temp, field)
        self.cl_copy(field_linear, field)
        self.cl_n(field_temp, stepsize) #k0
        self.cl_sum(field, 1, field_temp, inv_six) #free k0

        self.cl_sum(field_temp, 0.5, field_linear, 1)
        self.cl_n(field_temp, stepsize) #k1
        self.cl_sum(field, 1, field_temp, inv_three) #free k1
        
        self.cl_sum(field_temp, 0.5, field_linear, 1)
        self.cl_n(field_temp, stepsize) #k2
        self.cl_sum(field, 1, field_temp, inv_three) #free k2
        
        self.cl_sum(field_temp, 1, field_linear, 1)
        self.cl_n(field_temp, stepsize) #k3
        self.cl_sum(field, 1, field_temp, inv_six) #free k3
        
        self.cl_linear(field, half_step, factor)
        
    def cl_rk4ip(self, field, field_temp, field_interaction, factor, stepsize):
        """ Runge-Kutta in the interaction picture method using OpenCL. """
        '''
        hh = 0.5 * h
        A_I = f.linear(A, hh)
        k0 = f.linear(h * f.n(A, z), hh)
        k1 = h * f.n(A_I + 0.5 * k0, z + hh)
        k2 = h * f.n(A_I + 0.5 * k1, z + hh)
        k3 = h * f.n(f.linear(A_I + k2, hh), z + h)
        return (k3 / 6.0) + f.linear(A_I + (k0 + 2.0 * (k1 + k2)) / 6.0, hh)
        '''
        inv_six = 1.0 / 6.0
        inv_three = 1.0 / 3.0
        half_step = 0.5 * stepsize

        self.cl_copy(field_temp, field)
        self.cl_linear(field, half_step, factor)

        self.cl_copy(field_interaction, field) #A_I
        self.cl_n(field_temp, stepsize)
        self.cl_linear(field_temp, half_step, factor) #k0

        self.cl_sum(field, 1.0, field_temp, inv_six) #free k0
        self.cl_sum(field_temp, 0.5, field_interaction, 1.0)
        self.cl_n(field_temp, stepsize) #k1

        self.cl_sum(field, 1.0, field_temp, inv_three) #free k1
        self.cl_sum(field_temp, 0.5, field_interaction, 1.0)
        self.cl_n(field_temp, stepsize) #k2

        self.cl_sum(field, 1.0, field_temp, inv_three) #free k2
        self.cl_sum(field_temp, 1.0, field_interaction, 1.0)
        self.cl_linear(field_temp, half_step, factor)
        self.cl_n(field_temp, stepsize) #k3

        self.cl_linear(field, half_step, factor)

        self.cl_sum(field, 1.0, field_temp, inv_six)

if __name__ == "__main__":
    # Compare simulations using Fibre and OpenclFibre modules.
    from pyofss import Domain, System, Gaussian, Sech, Fibre
    from pyofss import power_buffer, multi_plot, labels, lambda_to_nu

    import time
    
    print("*** Test of the default nonlinearity ***")
    # -------------------------------------------------------------------------
    TS = 4096*16
    GAMMA = 20.0
    BETA = [0.0, 0.0, 0.0, 22.0]
    STEPS = 800
    LENGTH = 0.1

    DOMAIN = Domain(bit_width=50.0, samples_per_bit=TS, centre_nu=lambda_to_nu(1050.0))

    SYS = System(DOMAIN)
    SYS.add(Gaussian("gaussian", peak_power=1.0, width=1.0))
    SYS.add(Fibre("fibre", beta=BETA, gamma=GAMMA,
                  length=LENGTH, total_steps=STEPS, method="RK4IP"))

    start = time.time()
    SYS.run()
    stop = time.time()
    NO_OCL_DURATION = (stop - start)
    NO_OCL_OUT = SYS.fields["fibre"]

    sys = System(DOMAIN)
    sys.add(Gaussian("gaussian", peak_power=1.0, width=1.0))
    sys.add(OpenclFibre("ocl_fibre", beta=BETA, gamma=GAMMA,
                        length=LENGTH, total_steps=STEPS))

    start = time.time()
    sys.run()
    stop = time.time()
    OCL_DURATION = (stop - start)
    OCL_OUT = sys.fields["ocl_fibre"]

    NO_OCL_POWER = power_buffer(NO_OCL_OUT)
    OCL_POWER = power_buffer(OCL_OUT)
    DELTA_POWER = NO_OCL_POWER - OCL_POWER

    MEAN_RELATIVE_ERROR = np.mean(np.abs(DELTA_POWER))
    MEAN_RELATIVE_ERROR /= np.max(power_buffer(NO_OCL_OUT))
    
    MAX_RELATIVE_ERROR = np.max(np.abs(DELTA_POWER))
    MAX_RELATIVE_ERROR /= np.max(power_buffer(NO_OCL_OUT))

    print("Run time without OpenCL: %e" % NO_OCL_DURATION)
    print("Run time with OpenCL: %e" % OCL_DURATION)
    print("Mean relative error: %e" % MEAN_RELATIVE_ERROR)
    print("Max relative error: %e" % MAX_RELATIVE_ERROR)

    # Expect both plots to appear identical:
    multi_plot(SYS.domain.t, [NO_OCL_POWER, OCL_POWER], z_labels=['CPU','GPU'],
                x_label=labels["t"], y_label=labels["P_t"], use_fill=False)

    print("*** Test of the Raman response ***")
    # -------------------------------------------------------------------------
    # Dudley_SC, fig 6
    TS = 2**13
    GAMMA = 110.0
    BETA = [0.0, 0.0, -11.830]
    STEPS = 800
    LENGTH = 50*1e-5
    WIDTH = 0.02836

    N_sol = 3.
    L_d = (WIDTH**2)/abs(BETA[2])
    #L_nl = 1/(gamma*P_0)
    L_nl = L_d/(N_sol**2)
    P_0 = 1/(GAMMA*L_nl)

    DOMAIN = Domain(bit_width=10.0, samples_per_bit=TS, centre_nu=lambda_to_nu(835.0))

    SYS = System(DOMAIN)
    SYS.add(Sech(peak_power=P_0, width=WIDTH))
    SYS.add(Fibre("fibre", beta=BETA, gamma=GAMMA, self_steepening=False, use_all='hollenbeck',
                  length=LENGTH, total_steps=STEPS, method="RK4IP"))

    start = time.time()
    SYS.run()
    stop = time.time()
    NO_OCL_DURATION = (stop - start)
    NO_OCL_OUT = SYS.fields["fibre"]

    sys = System(DOMAIN)
    sys.add(Sech(peak_power=P_0, width=WIDTH))
    sys.add(OpenclFibre("ocl_fibre", beta=BETA, gamma=GAMMA, self_steepening=False, use_all='hollenbeck',
                        length=LENGTH, total_steps=STEPS))

    start = time.time()
    sys.run()
    stop = time.time()
    OCL_DURATION = (stop - start)
    OCL_OUT = sys.fields["ocl_fibre"]

    NO_OCL_POWER = power_buffer(NO_OCL_OUT)
    OCL_POWER = power_buffer(OCL_OUT)
    DELTA_POWER = NO_OCL_POWER - OCL_POWER

    MEAN_RELATIVE_ERROR = np.mean(np.abs(DELTA_POWER))
    MEAN_RELATIVE_ERROR /= np.max(power_buffer(NO_OCL_OUT))
    
    MAX_RELATIVE_ERROR = np.max(np.abs(DELTA_POWER))
    MAX_RELATIVE_ERROR /= np.max(power_buffer(NO_OCL_OUT))

    print("Run time without OpenCL: %e" % NO_OCL_DURATION)
    print("Run time with OpenCL: %e" % OCL_DURATION)
    print("Mean relative error: %e" % MEAN_RELATIVE_ERROR)
    print("Max relative error: %e" % MAX_RELATIVE_ERROR)

    # Expect both plots to appear identical:
    multi_plot(SYS.domain.t, [NO_OCL_POWER, OCL_POWER], z_labels=['CPU','GPU'],
                x_label=labels["t"], y_label=labels["P_t"], use_fill=False)

    print("*** Test of the self-steepening contribution ***")
    # -------------------------------------------------------------------------
    # Dudley_SC, fig 8
    TS = 2**13
    GAMMA = 110.0
    BETA = [0.0, 0.0, -11.830, 
           8.1038e-2,  -9.5205e-5,   2.0737e-7,  
          -5.3943e-10,  1.3486e-12, -2.5495e-15, 
           3.0524e-18, -1.7140e-21]
    STEPS = 800
    LENGTH = 50*1e-5
    WIDTH = 0.010
    P_0 = 3480.
    
    try:
        clf = OpenclFibre("ocl_fibre", beta=BETA, gamma=GAMMA, self_steepening=True, use_all=False,
                        dorf="double", length=LENGTH, total_steps=STEPS)
    except NotImplementedError:
        print("Not Implemented, skipped")
    
    print("*** Test of the supercontinuum generation ***")
    # -------------------------------------------------------------------------
    # Dudley_SC, fig 3
    TS = 2**14
    GAMMA = 110.0
    BETA = [0.0, 0.0, -11.830, 
           8.1038e-2,  -9.5205e-5,   2.0737e-7,  
          -5.3943e-10,  1.3486e-12, -2.5495e-15, 
           3.0524e-18, -1.7140e-21]
    STEPS = 10000
    LENGTH = 15*1e-5
    WIDTH = 0.050
    P_0 = 10000.
    TAU_SHOCK = 0.56e-3

    DOMAIN = Domain(bit_width=10.0, samples_per_bit=TS, centre_nu=lambda_to_nu(835.0))

    SYS = System(DOMAIN)
    SYS.add(Sech(peak_power=P_0, width=WIDTH, using_fwhm=True))
    SYS.add(Fibre("fibre", beta=BETA, gamma=GAMMA, self_steepening=TAU_SHOCK, use_all='hollenbeck',
                  length=LENGTH, total_steps=STEPS, method="RK4IP"))

    start = time.time()
    SYS.run()
    stop = time.time()
    NO_OCL_DURATION = (stop - start)
    NO_OCL_OUT = SYS.fields["fibre"]

    sys = System(DOMAIN)
    sys.add(Sech(peak_power=P_0, width=WIDTH, using_fwhm=True))
    sys.add(OpenclFibre("ocl_fibre", beta=BETA, gamma=GAMMA, self_steepening=TAU_SHOCK, use_all='hollenbeck',
                        length=LENGTH, total_steps=STEPS))

    start = time.time()
    sys.run()
    stop = time.time()
    OCL_DURATION = (stop - start)
    OCL_OUT = sys.fields["ocl_fibre"]

    NO_OCL_POWER = power_buffer(NO_OCL_OUT)
    OCL_POWER = power_buffer(OCL_OUT)
    DELTA_POWER = NO_OCL_POWER - OCL_POWER

    MEAN_RELATIVE_ERROR = np.mean(np.abs(DELTA_POWER))
    MEAN_RELATIVE_ERROR /= np.max(power_buffer(NO_OCL_OUT))
    
    MAX_RELATIVE_ERROR = np.max(np.abs(DELTA_POWER))
    MAX_RELATIVE_ERROR /= np.max(power_buffer(NO_OCL_OUT))

    print("Run time without OpenCL: %e" % NO_OCL_DURATION)
    print("Run time with OpenCL: %e" % OCL_DURATION)
    print("Mean relative error: %e" % MEAN_RELATIVE_ERROR)
    print("Max relative error: %e" % MAX_RELATIVE_ERROR)

    # Expect both plots to appear identical:
    multi_plot(SYS.domain.t, [NO_OCL_POWER, OCL_POWER], z_labels=['CPU','GPU'],
                x_label=labels["t"], y_label=labels["P_t"], use_fill=False)


