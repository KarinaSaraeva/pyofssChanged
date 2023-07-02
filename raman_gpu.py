from pyofss import Domain, System, Gaussian, Fibre, OpenclFibre
from pyofss import temporal_power, spectral_power, lambda_to_nu, nu_to_lambda, multi_plot, labels
import numpy as np
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import pylab as plt
plt.switch_backend('agg')
import time

lamb0 = 1028
peak_power = 600   
peak_width = 2    
peak_C = 0.
total_steps = 20000
betta_2 = 20  
gamma_active = 5.1  
La_1 = 5*1e-3

Pp_0 = 10               # Ws
N = 4.8
Rr = 80*1e-6            # THz
Tr = 1/Rr    

domain = Domain(samples_per_bit=2**14, bit_width=200.0,
                total_bits=1, centre_nu=lambda_to_nu(lamb0))
gaussian = Gaussian(name="initial_pulse", peak_power=peak_power,
                    width=peak_width, C=peak_C, using_fwhm=True)


sys1 = System(domain)
sys1.add(gaussian)
sys1.add(OpenclFibre(length=La_1, total_steps=total_steps, method="cl_ss_symmetric",
            beta=[0.0, 0.0, betta_2], gamma=gamma_active, self_steepening=False, use_all=True, 
            use_Yb_model=True, fast_math=True, N=N, Pp_0=Pp_0, Rr=Rr,
            dorf='double'))
start = time.time()
sys1.run()
stop = time.time()
CL_OUT = sys1.field

print("Run time with cl is {}".format(stop-start))


sys = System(domain)
sys.add(gaussian)
sys.add(Fibre(length=La_1, total_steps=total_steps, traces=total_steps, method="ss_symmetric",
            beta=[0.0, 0.0, betta_2], gamma=gamma_active, self_steepening=False, use_all=True,raman_scattering=True,
            use_Yb_model=True, N=N, Pp_0=Pp_0, Rr=Rr))

start = time.time()
sys.run()
stop = time.time()
NO_CL_OUT = sys.field

print("Run time without cl is {}".format(stop-start))

multi_plot(sys1.domain.t, [temporal_power(sys.field), temporal_power(sys1.field)], ["cpu", "gpu"], labels["t"], labels["P_t"],
            x_range=(-50.0, 50.0), use_fill=False)
plt.savefig('raman_cl_compare')
df_results = sys.df_results  
df_results1 = sys1.df_results
energy_arr = df_results['energy'].values
energy_arr1 = df_results1['energy'].values
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
