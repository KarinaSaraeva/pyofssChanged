from pyofss import Domain, System, Gaussian, Fibre, OpenclFibre
from pyofss import temporal_power, spectral_power, lambda_to_nu, nu_to_lambda, multi_plot, labels
import numpy as np
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import pylab as plt
#plt.switch_backend('agg')

import time

domain = Domain(bit_width=200.0, samples_per_bit=2048*16)
gaussian = Gaussian(peak_power=1.0, width=1.0)

fib = Fibre(length=5.0, method='rk4ip', total_steps=200, traces=200, 
            beta=[0.0, 0.0, 0.0, 1.0], gamma=1.0, self_steepening=False, use_all=True)

sys = System(domain)
sys.add(gaussian)
sys.add(fib)

start = time.time()
sys.run()
stop = time.time()
NO_CL_OUT = sys.field

print("Run time without cl is {}".format(stop-start))

#single_plot(sys.domain.t, temporal_power(sys.field), labels["t"], labels["P_t"],
#            x_range=(-20.0, 40.0), use_fill=False)
#plt.savefig('raman_without_cl')

sys1 = System(domain)
sys1.add(gaussian)
sys1.add(OpenclFibre(length=5.0, total_steps=200,
            beta=[0.0, 0.0, 0.0, 1.0], gamma=1.0, self_steepening=False, use_all=True,
            dorf='double'))
start = time.time()
sys1.run()
stop = time.time()
CL_OUT = sys1.field

print("Run time with cl is {}".format(stop-start))

multi_plot(sys1.domain.t, [temporal_power(sys.field), temporal_power(sys1.field)], ["cpu", "gpu"], labels["t"], labels["P_t"],
            x_range=(-20.0, 40.0), use_fill=False)
plt.savefig('raman_cl_compare')


df_results = sys.get_laser_info()
energy_arr = df_results['energy'].values
energy_arr1 = np.array(sys1.modules[1].energy_list)
multi_plot(df_results.index.get_level_values("z [mm]").values, [energy_arr, energy_arr1], ["cpu", "gpu"], labels["z"], "E_t", use_fill=False)
plt.savefig('raman_cl_compare_energy')

NO_CL_POWER = temporal_power(NO_CL_OUT)
CL_POWER = temporal_power(CL_OUT)
DELTA_POWER = NO_CL_POWER - CL_POWER

MEAN_RELATIVE_ERROR = np.mean(np.abs(DELTA_POWER))
MEAN_RELATIVE_ERROR /= np.max(temporal_power(NO_CL_OUT))

MAX_RELATIVE_ERROR = np.max(np.abs(DELTA_POWER))
MAX_RELATIVE_ERROR /= np.max(temporal_power(NO_CL_OUT))

print("Mean relative error: %e" % MEAN_RELATIVE_ERROR)
print("Max relative error: %e" % MAX_RELATIVE_ERROR)
