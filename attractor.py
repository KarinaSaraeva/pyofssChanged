from pyofss import Domain, System, Gaussian, Fibre, OpenclFibre
from pyofss import temporal_power, spectral_power, lambda_to_nu, nu_to_lambda, multi_plot, single_plot, labels
import numpy as np
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import pylab as plt
plt.switch_backend('agg')
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import time
import sys
import os

# peak params 
peak_power = float(sys.argv[0]) # 600 W  
peak_width = float(sys.argv[1]) # 2 ps
output_dir = sys.argv[2]

print("use opencl fibre to model pulse propagation")
print(f"parameters: \n  peak_power = {peak_power}, peak_width = {peak_width}")
# parameters from the article "Nonlinear ultrafast fiber amplifiers beyond the gain-narrowing limit" PAVEL SIDORENKO,* WALTER FU, AND FRANK WISE
result_folder = os.path.join(output_dir, f"PP_{peak_power:.2f}_PW_{peak_width:.2f}")

peak_C = 0.
total_steps = 30001
lamb0 = 1028    
betta_2 = 20  
gamma_active = 5.1  
La_1 = 7*1e-3
Pp_0 = 10               # Ws
N = 4.8
Rr = 80*1e-6            # THz
Tr = 1/Rr    
downsampling = 500

domain = Domain(samples_per_bit=2**14, bit_width=200.0,
                total_bits=1, centre_nu=lambda_to_nu(lamb0))
gaussian = Gaussian(name="initial_pulse", peak_power=peak_power,
                    width=peak_width, C=peak_C, using_fwhm=True)
# input info
lines = []
lines.append(f"output_dir: {output_dir}")
lines.append(f"peak_power: {peak_power}")
lines.append(f"total_steps: {total_steps}")
lines.append(f"lamb0: {lamb0}")
lines.append(f"betta_2: {betta_2}")
lines.append(f"La_1: {La_1}")
lines.append(f"Pp_0: {Pp_0}")
lines.append(f"N: {N}")
lines.append(f"Tr: {Tr}")
lines.append(f"downsampling: {downsampling}")
with open(os.path.join(output_dir, "input_info.txt"), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')

# system calculations
sys1 = System(domain)
sys1.add(gaussian)
sys1.add(OpenclFibre(length=La_1, total_steps=total_steps,
            beta=[0.0, 0.0, betta_2], gamma=gamma_active, self_steepening=False, use_all=True,
            use_Yb_model=True, N=N, Pp_0=Pp_0, Rr=Rr, fast_math=True,
            dorf='double', traces=150, downsampling=500))
start = time.time()
sys1.run()
stop = time.time()
CL_OUT = sys1.field
sys1.df_results
print("Run time with cl is {}".format(stop-start))

# result dataframes
sys1.save_df_to_csv(output_dir, is_temporal=True, save_power=True)
sys1.save_df_to_csv(output_dir, is_temporal=False, save_power=True)
sys1.save_result_df_to_scv(output_dir)

f_t = interp1d(np.arange(len(domain.t)), domain.t)
interpolated_t = f_t(np.linspace(0, len(domain.t) - 1, downsampling))

f_nu = interp1d(np.arange(len(domain.nu)), domain.nu)
interpolated_nu = f_nu(np.linspace(0, len(domain.nu) - 1, downsampling))

# plots
single_plot(interpolated_t, sys1.df_temp.iloc[-1], labels["t"], labels["P_t"], "final_pulse")
plt.savefig(os.path.join(output_dir, 'final_pulse_t'))
single_plot(interpolated_nu, sys1.df_spec.iloc[-1], labels["nu"], labels["P_nu"], "final_pulse")
plt.savefig(os.path.join(output_dir, 'final_pulse_nu'))

# output_info
lines = []
def get_peaks(P):
    peaks, _ = find_peaks(P, height=0, prominence=(np.amax(P)/10))
    return peaks

output_peak_power = sys1.df_results.iloc[-1]["max_value"]
output_energy = sys1.df_results.iloc[-1]["energy"]
output_peaks = get_peaks(temporal_power(sys1.field))

lines.append(f"output_peaks: {output_peaks}")
lines.append(f"num of peaks: {len(output_peaks)}")
lines.append(f"output_peak_power: {output_peak_power}")
lines.append(f"output_energy: {output_energy} nJ")
lines.append(f"L_NL {sys1.modules[1].L_NL*1e3} m")
lines.append(f"L_D {sys1.modules[1].L_D*1e3} m")

with open(os.path.join(output_dir, "output_info.txt"), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')