from scipy.signal import find_peaks
import sys
from pyofss.modules.nonlinearity import calculate_gamma
from pyofss.domain import lambda_to_omega
import os.path
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd

from pyofss import Domain, System, Gaussian, Fibre, Filter, Splitter, FibrePlotter
from pyofss import temporal_power, spectral_power, lambda_to_nu, nu_to_lambda
from pyofss import single_plot, map_plot, waterfall_plot, labels
from pyofss.field import energy, max_peak_params, spectrum_width_params

from pyofss import labels
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams["figure.dpi"] = 100

font = {'family': 'serif',
        'weight': 'normal',
        'size': 8}

matplotlib.rc('font', **font)


def get_caption(folder_name):
    def sorting_func(name, param_string):
        name = name.split(f"_{param_string}_", 1)[1]
        sub = name.split("_", 1)[1] if (len(name.split("_", 1)) > 0) else ""
        name = name.replace(sub, "")
        name = name.replace("_", "")
        return float(name)
    caption = f"Esat_1 = {sorting_func(folder_name, 'Esat_1')}, Esat_2 = {sorting_func(folder_name, 'Esat_2')}, La_1 = {sorting_func(folder_name, 'La_1')}, La_2 = {sorting_func(folder_name, 'La_2')}"
    return caption


def draw_heat_map(df, domain, is_temporal=True, prominence=0.005, title="laser", subplot_spec=None, fig=None):
    if (is_temporal):
        x = domain.t
        x_label = labels["t"]
    else:
        x = domain.nu
        x_label = labels["nu"]
    y = df.values
    z = df.index.get_level_values('z [mm]').values
    d_x = abs(x[1] - x[0])
    plotter = FibrePlotter(x, y, z, x_label, labels["P_t"], labels["z"])
    plotter.draw_heat_map(0, prominence=prominence, is_temporal=is_temporal,
                          subplot_spec=subplot_spec, fig=fig, title=title, vmin=0)


def draw_double_plot(df, subplot_spec=None, fig=None, is_temporal=True):
    if (fig and subplot_spec is not None):
        inner = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                 subplot_spec=subplot_spec, wspace=0.1, hspace=0.1)
        ax1 = plt.subplot(inner[0, 0])
    else:
        fig, ax1 = plt.subplots(figsize=(30, 6))

    color = 'tab:red'
    ax1.set_xlabel(labels["z"])

    if (is_temporal):
        ax1.set_ylabel(labels["t"], color=color)
    else:
        ax1.set_ylabel(labels["nu"], color=color)

    ax1.plot(df.index.get_level_values(
        'z [mm]').values, df["duration"].values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()

    color = 'tab:green'

    if (is_temporal):
        ax2.set_ylabel(labels["P_t"], color=color)
    else:
        ax2.set_ylabel(labels["P_nu"], color=color)

    ax2.plot(df.index.get_level_values(
        'z [mm]').values, df["max_value"].values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()


def get_duration(P, d_x, prominence=0.01):
    heigth_fwhm, fwhm, left_ind, right_ind = max_peak_params(
        P, prominence=prominence)
    return fwhm*d_x


def get_duration_spec(P, d_x, prominence=0.0001):
    heigth_fwhm, fwhm, left_ind, right_ind = spectrum_width_params(
        P, prominence=prominence)
    return abs(fwhm)*d_x


def gaussian(x, a, x0, sigma):
    return a*np.exp(-((x-x0)**2)/(2*sigma**2))


def field_animation(df, domain, path_to_animation, is_temporal=True):
    if (is_temporal):
        x = domain.t
        x_label = labels["t"]
    else:
        x = domain.nu
        x_label = labels["nu"]
    y = df.values
    z = df.index.get_level_values('z [mm]').values

    plotter = FibrePlotter(x, y, z, x_label, labels["P_t"], labels["z"])
    anim = plotter.draw_animation()
    # writervideo = ImageMagickFileWriter(fps=100)
    # anim.save(path_to_animation, writer=writervideo)


def get_laser_info(df_laser, domain, cycle, caption, prominence, path_to_graph, is_temporal):
    if (is_temporal):
        additive = "temp"
        # prominence = 0.1
    else:
        additive = "spec"
        # prominence = 0.0001
    fig = plt.figure(figsize=(20, 10))
    outer = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.2)
    characteristic = ["max_value", "duration", "energy"]
    iterables = [characteristic]
    index = pd.MultiIndex.from_product(iterables,  names=["charact"])
    df_results = pd.DataFrame(index=df_laser.index, columns=index)
    draw_heat_map(df_laser.loc[(cycle)], domain, is_temporal,
                  prominence=prominence, title=caption, subplot_spec=outer[0], fig=fig)

    if (is_temporal):
        x = domain.t
        d_x = abs(x[1] - x[0])
        df_results['max_value'] = df_laser.max(axis=1).values
        df_results['duration'] = df_laser.apply(
            lambda row: get_duration(row, d_x, prominence), axis=1).values
        df_results['energy'] = df_laser.apply(
            lambda row: energy(row, domain.t), axis=1).values
        draw_double_plot(df_results.loc[(cycle)], outer[1], fig)
    else:
        x = domain.nu
        d_x = abs(x[1] - x[0])
        df_results['duration'] = df_laser.apply(
            lambda row: get_duration_spec(row, d_x, prominence), axis=1).values
        draw_double_plot(df_results.loc[(cycle)], outer[1], fig)

    fig.savefig(os.path.join(path_to_graph, f"info_{cycle}_{additive}.png"))
    return df_results


def get_output_field(df_laser, last_cycle):
    z_output = df_laser.index.get_level_values('z [mm]').values[-1]
    last_fibre = df_laser.loc[(last_cycle, "5_active_fibre")]
    z_output = last_fibre.index.get_level_values('z [mm]').values[-1]
    output_field = last_fibre.loc[(z_output)]
    return output_field


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Directory: ", dir, " is created!")


def draw_output_fiiled_graph(domain, temporal_power, spectral_power, path_to_graph):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(domain.nu, spectral_power)
    ax[0].set_yscale('log')
    ax[0].set_title(f"spectrum (log scale)")
    ax[0].set_xlabel(labels['nu'])
    ax[0].set_ylabel(labels['P_nu'])

    ax[1].plot(domain.t, temporal_power)
    ax[1].set_title(f"temporal pulse")
    ax[1].set_xlabel(labels['t'])
    ax[1].set_ylabel(labels['P_t'])

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.1, hspace=0.5)
    fig.savefig(os.path.join(path_to_graph, f"output_field.png"))


def energy_change(result_temp, path_to_graph):
    cycle_output_energy = np.zeros(len(cycle_names))
    cycle_index = np.arange(len(cycle_names))
    for i in cycle_index:
        df_fibre_befor_c = result_temp.loc[(cycle_names[i], "3_passive_fibre")]
        z_before_c = df_fibre_befor_c.index.get_level_values("z [mm]")[-1]
        cycle_output_energy[i] = df_fibre_befor_c.loc[(z_before_c)]["energy"]
    fig, ax = plt.subplots(1, 1)
    ax.scatter(cycle_index, cycle_output_energy)
    fig.savefig(os.path.join(path_to_graph, f"energy change.png"))


F1 = float(sys.argv[1])
F2 = float(sys.argv[2])
FW1 = float(sys.argv[3])
FW2 = float(sys.argv[4])
C1 = 1 -float(sys.argv[5])
C2 = 1 - float(sys.argv[6])
E1 = float(sys.argv[7])
E2 = float(sys.argv[8])
L3 = float(sys.argv[9])
L6 = float(sys.argv[10])
SSG = float(sys.argv[11])
G1 = float(sys.argv[12])
G2 = float(sys.argv[13])
N = int(sys.argv[14])  # use from sh script
local_error = float(sys.argv[15])
step_size = float(sys.argv[16])
ResultDirName = sys.argv[17]
ResultDir = os.path.join(
    sys.argv[18], ResultDirName)
useRaman = bool(float(sys.argv[19]))
useAdaptive = bool(float(sys.argv[20]))

# F1 = lambda_to_nu(1040) - lambda_to_nu(1035)
# F2 = (lambda_to_nu(1030) - lambda_to_nu(1035))
# FW1 = (Domain.vacuum_light_speed/(1040**2))*4
# FW2 = (Domain.vacuum_light_speed/(1030**2))*4
# C1 = 0.4
# C2 = 0.9
# E1 = 9.15
# E2 = 36.6
# L3 = 0.8
# L6 = 0.8
# SSG = 10
# G1 = calculate_gamma(2.7*(10**(-20)), 37.8, lambda_to_omega(1035))
# G2 = calculate_gamma(2.7*(10**(-20)), 29, lambda_to_omega(1035))
# N = 1

Esat_1 = E1                     # 9.15nJ
Esat_2 = E2                   # 36.6nJ
La_1 = 2.5*1e-3                  # 2.5m
La_2 = 2.5*1e-3                  # 2.5m
Lp_1 = 0.8*1e-3                                 # 0.8m
Lp_2 = L3*1e-3                                 # 0.8m
Lp_3 = 0.8*1e-3                                 # 0.8m
Lp_4 = L6*1e-3                                 # 0.8m
# step_size = 5*1e-8

small_signal_gain = SSG
N_cycles = int(N)
gamma_passive = G1
gamma_active = G2
# peak_power = 50.
# peak_width = 3.
# peak_C = 100.

peak_power = 1000.
peak_width = 10.
peak_C = 300.

print(f"parameters: \n FW1 = {FW1}, F1 = {F1}, gamma_passive = {gamma_passive}, FW2 = {FW2}, F2 = {F2}, gamma_active = {gamma_active}, \n Lp_1 = {Lp_1}, Lp_2 = {Lp_2}, Lp_3 = {Lp_3}, Lp_4 = {Lp_4}, La_1 = {La_1}, La_2 = {La_2}, C1 = {C1}, C2 = {C2}, Esat_1 = {Esat_1}, Esat_2 = {Esat_2}, local_error = {local_error}")

useRamanName = ""
useAdaptiveName = ""
if useRaman:
    use_all = "hollenbeck"
    useRamanName = "rmn"
    print(f"use raman gain")
else: 
    use_all = False
    useRamanName = "no_rmn"
    print(f"DO NOT use raman gain")

if useAdaptive:
    method='ass_symmetric'
    useAdaptiveName = "as"
    print(f"use adaptive stepper")
else: 
    method='ss_symmetric'
    useAdaptiveName = "ss"
    print(f"DO NOT use adaptive stepper")

folder_name = f"{useRamanName}_{useAdaptiveName}_{F1:.2f}_F2_{F2:.2f}_FW1_{FW1:.2f}_FW2_{FW2:.2f}_C1_{C1:.2f}_C2_{C2:.2f}_E1_{E1:.2f}_E2_{E2:.2f}_L3_{L3:.2f}_L6_{L6:.2f}_SSG_{SSG:.2f}_G1_{G1:.2f}_G2_{G2:.2f}_N_{N}_LE_{local_error}_SS_{step_size}"

RootDir = os.path.join(
    ResultDir, folder_name)
check_dir(RootDir)
graph_dir = os.path.join(RootDir, "graph")
check_dir(graph_dir)

print(f"result dir: \n {graph_dir}")

# domain = Domain(samples_per_bit=2**15, bit_width=200.0,
#                total_bits=1, centre_nu=lambda_to_nu(1035))
domain = Domain(samples_per_bit=2**15, bit_width=400.0,
                total_bits=1, centre_nu=lambda_to_nu(1035))

gaussian = Gaussian(name="initial_pulse", peak_power=peak_power,
                    width=peak_width, C=peak_C, using_fwhm=True)
A = gaussian.generate(domain.t)
E = energy(A, domain.t)
# print("initial pulse energy = ", E)

Dir = RootDir
GraphDir = os.path.join(Dir, "graph")
plt.clf()
plt.plot(domain.t, temporal_power(A))
check_dir(GraphDir)
plt.savefig(os.path.join(
    GraphDir, f"P_{peak_power}_width_{peak_width}_C_{peak_C}.png"))
plt.show()

GraphDir = os.path.join(Dir, "mamyshev_oscillator")
sys = System(domain, A)

for i in range(N_cycles):
    cycleDir = os.path.join(Dir, f'cycle{int(i)}')
    sys.add(Filter(name="filter_1", width_nu=(FW1), offset_nu=(F1),
                   m=1, channel=0, using_fwhm=True, type_filt="reflected"))
    sys.add(Fibre(name="1_passive_fibre", length=Lp_1, gamma=gamma_passive, beta=np.array(
        [0, 0, 22.2]), total_steps=int(Lp_1/step_size), traces=100, local_error=local_error, use_all=use_all, method=method, cycle=f'cycle{int(i)}', dir=cycleDir, save_represent="both"))
    sys.add(Fibre(name="2_active_fibre", length=La_1, gamma=gamma_active, beta=np.array(
        [0, 0, 24.9]), total_steps=int(La_1/step_size), traces=100, local_error=local_error, use_all=use_all, method=method, cycle=f'cycle{int(i)}', small_signal_gain=small_signal_gain, E_sat=Esat_1, lamb0=1035., bandwidth=40., dir=cycleDir, save_represent="both"))
    sys.add(Fibre(name="3_passive_fibre", length=Lp_2, gamma=gamma_passive, beta=np.array(
        [0, 0, 22.2]), total_steps=int(Lp_2/step_size), traces=100, local_error=local_error, use_all=use_all, method=method, cycle=f'cycle{int(i)}', dir=cycleDir, save_represent="both"))
    sys.add(Splitter(name="splitter", loss=C1))
    sys.add(Filter(name="filter_2", width_nu=(FW2), offset_nu=(F2),
                   m=1, channel=0, using_fwhm=True, type_filt="reflected"))
    sys.add(Fibre(name="4_passive_fibre", length=Lp_3, gamma=gamma_passive, beta=np.array(
        [0, 0, 22.2]),  total_steps=int(Lp_3/step_size), traces=100, local_error=local_error, use_all=use_all, method=method, cycle=f'cycle{int(i)}', dir=cycleDir, save_represent="both"))
    sys.add(Fibre(name="5_active_fibre", length=La_2, gamma=gamma_active, beta=np.array(
        [0, 0, 24.9]), total_steps=int(La_2/step_size), traces=100, local_error=local_error, use_all=use_all, method=method, cycle=f'cycle{int(i)}', small_signal_gain=small_signal_gain, E_sat=Esat_2, lamb0=1035., bandwidth=40., dir=cycleDir, save_represent="both"))
    sys.add(Fibre(name="6_passive_fibre", length=Lp_4, gamma=gamma_passive, beta=np.array(
        [0, 0, 22.2]), total_steps=int(Lp_4/step_size), traces=100, local_error=local_error, use_all=use_all, method=method, cycle=f'cycle{int(i)}', dir=cycleDir, save_represent="both"))
    sys.add(Splitter(name="splitter", loss=C2))  # use from sh script

sys.run()

sys.save_df_to_csv(Dir, is_temporal=True)
sys.save_df_to_csv(Dir, is_temporal=False)

sys.init_df(is_temporal=True)
sys.init_df(is_temporal=False)

cycle_names = list(set(sys.df_temp.index.get_level_values('cycle').values))
cycle_names.sort()

output_field_temp = get_output_field(sys.df_temp, cycle_names[-1])
output_field_spec = get_output_field(sys.df_spec, cycle_names[-1])

draw_output_fiiled_graph(domain, output_field_temp,
                         output_field_spec, graph_dir)

prominence_temp = (np.amax(output_field_temp)/1000)
prominence_spec = (np.amax(output_field_spec)/1000)

result_temp = get_laser_info(
    sys.df_temp, domain, cycle_names[-1], folder_name, prominence_temp, graph_dir, True)
result_temp.to_csv(os.path.join(graph_dir, "laser_info_temp.csv"))

energy_change(result_temp, graph_dir)

# result_spec = get_laser_info(
#     sys.df_spec, domain, cycle_names[-1], folder_name, prominence_spec, graph_dir, False)
# result_spec.to_csv(os.path.join(graph_dir, "laser_info_spec.csv"))

sys.save_last_cycle_df_to_csv(graph_dir, is_temporal=True)
sys.save_last_cycle_df_to_csv(graph_dir, is_temporal=False)

x = domain.t
d_x = x[1] - x[0]
output_field_last = get_output_field(sys.df_temp, cycle_names[-1])
output_field_prev = get_output_field(sys.df_temp, cycle_names[-2])
E_last = np.amax(output_field_last)
E_prev = np.amax(output_field_prev)

delta_E = (abs(E_last - E_prev)/E_prev)
lines = []
lines.append(f"E_last = {E_last}")
lines.append(f"delta_E = {delta_E*100} %")
duration_last = get_duration(output_field_last, d_x, prominence_temp)
duration_prev = get_duration(output_field_prev, d_x, prominence_temp)
delta_duraion = (abs(duration_last - duration_prev)/duration_prev)
lines.append(f"duration_last = {duration_last}")
lines.append(f"delta_duraion = {delta_duraion*100} %")
prominence = E_last/100
peaks, _ = find_peaks(output_field_last, height=0, prominence=prominence)
lines.append(f"peaks = {peaks}")

if delta_E < 0.00001 and delta_duraion < 0.0002:
    lines.append("generation is stable")
else:
    lines.append("generation is Not stable")

with open(os.path.join(graph_dir, "output_info.txt"), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
