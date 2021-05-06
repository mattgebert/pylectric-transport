%matplotlib inline
import os, sys
#To import pylectric package locally, specify relative path from cwd to package.
PACKAGE_PARENT = 'pylectric-transport'
sys.path.append(os.path.normpath(os.path.join(os.getcwd(), PACKAGE_PARENT)))

import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import warnings
from scipy.signal import savgol_filter, argrelextrema
import math

# from pylectric.materials.graphene import RVG_data, fitParamsRvG, fitSet, RvT_data
from pylectric.parsers.RVG import RVG_file
from pylectric.geometries.FET.hallbar import Meas_GatedResistance, Meas_Temp_GatedResistance
from pylectric.materials import graphene, sio2

### ----------------------------------------------------------------------------------------------------------------------------- ###
########################################################################################
####                        Import files and original data.                        #####
########################################################################################
PRELUDE = "\\01 Ga2O3 Devices"
#               Devs4_04, prior to Ga2O3 Deposition                 #
RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\02 Removed Outliers" #Folder for raw data
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V03-V04" # Bare Graphene [Gallium Area]
FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V01-V02" # Bare Graphene

#               Devs4_04, after Ga2O3 Deposition                 #
# RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\03 Data by date\\2020-12-15" #Folder for raw data
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V08-V07" # Gallium Oxide
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V01-V02" # Bare Graphene

#               Devs4_03, after Ga2O3 Deposition                 #
# RAW_DATA_DIR = "\\05 Devs4_03\\01 Outliers Removed\\2020-12-20" #Folder for raw data
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_03_run04_V09-V08" # Gallium Oxide
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_03_run04_V08-V07"   # Bare Graphene
di = [0,len(FILE_DESCRIPTOR)] #descriptor indexes

#Folder for graphical output:
RAW_DATA_DIR = PRELUDE + RAW_DATA_DIR
target = os.getcwd() + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR + "\\"
if not os.path.isdir(target):
    #Create directory.
    os.mkdir(target)

### Processing Properties ###
GEO_FACTOR = (400/200) #Geometry factor of the device - what's the conversion from resistance to resistivity? (\rho = R * (Geo_factor) = R * (W / L))

DIRPATH = os.getcwd() + RAW_DATA_DIR
files = [f for f in os.listdir(DIRPATH) if os.path.isfile(DIRPATH + "\\" + f) and f[di[0]:di[1]] == FILE_DESCRIPTOR]

### REMOVING EXTREMA FILES DUE TO UNFITTABLE
# files = files[0:34] #TODO REMOVE specific for Devs4_03_Run04_V09-V08

####        BEGIN ANALYSIS        ####
# files=[files[0]]
#Parse Data
file_data_objs = []
for file in files:
    data_obj = RVG_file(filepath=DIRPATH + "\\" + file)
    fig = data_obj.plot_all()
    file_data_objs.append(data_obj)


########################################################################################
####                        Setup graphing for Phonon Analysis                     #####
########################################################################################
#Create graphic directory for mobility
target2 = os.getcwd() + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR + "\\" + "02 Phonon Analysis (Fitted)" + "\\"
target3 = os.getcwd() + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR + "\\" + "02 Phonon Analysis (Fitted)" + "\\" + "Dirac Point Fitting" + "\\"
if not os.path.isdir(target2):
    os.mkdir(target2)
if not os.path.isdir(target3):
    os.mkdir(target3)

#Setup graphing
ud_labels = ["→","←"]
plt.rcParams.update({'font.size': 3, "figure.figsize" : [3,2], 'figure.dpi':300})
#Generate graphing colour cycles.
colourCycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
colourCycle2 = []
hexa = colourCycle[0]
for hexa in colourCycle:
    RGB_dark = tuple(int(0.6 * int(hexa.lstrip("#")[i:i+2], 16)) for i in (0,2,4))
    hex_dark = "#" + "".join(["{:02x}".format(RGB_dark[i]) for i in (0,1,2)])
    colourCycle2.append(hex_dark)
cycles = (colourCycle2, colourCycle)
#Calculate SiO2 capacitance for mobility calculation.
SiO2_Properties = sio2.SiO2(t_ox=2.85e-7)
SiO2_Cg = SiO2_Properties.capacitance()

########################################################################################
####                Create measurement objects, aquire temperatures etc.            #####
########################################################################################
temps = [] #Get average temperatures
meas_objs = [] #Setup sets of discritized set of gate voltages and resistances
# obj = file_data_objs[0]
for obj in file_data_objs:
    #Get raw measurement object
    datasets = obj.split_dataset_by_voltage_turning_point()
    t = []
    data_items = []
    #Setup sampling voltages
    i = 0
    for i in range(len(datasets)):
        #Calculate average temperature of dataset
        t.append(np.mean(datasets[i][:,4]))
        #Create gated measurement object.
        meas_obj = Meas_GatedResistance(data=datasets[i], Cg=SiO2_Cg, L=200, W=400)
        data_items.append(meas_obj)
    temps.append(t)
    meas_objs.append(data_items)
temps = np.array(temps)

####################################################################################
####   Get fits for conductivity behaviour of graphene. Get gate voltages.     #####
####################################################################################
plt.rcParams.update({'font.size': 3, "figure.figsize" : [6,1.5], 'figure.dpi':300})
RVG_fits = []
for meas_set_i in range(len(meas_objs)):
    meas_set = meas_objs[meas_set_i]
    fit_set = []
    #Setup graphing.
    gpr = 4 #graphs per row
    if meas_set_i % gpr == 0:
        fig, (axes) = plt.subplots(1, 4)
    for meas_obj_i in range(len(meas_set)):
        meas_obj = meas_set[meas_obj_i]
        try:
            params, covar = graphene.Graphene_Gated.fit_sigma_graphene(meas_obj)
            fit_set.append(params)
            meas_obj.plot_Sigma_vG(ax=axes[meas_set_i%gpr], c=colourCycle[meas_obj_i])
            p0 = tuple(params)
            axes[meas_set_i%gpr].plot(meas_obj.conductivity_data[:,0], graphene.Graphene_Gated.sigma_graphene(meas_obj.conductivity_data[:,0], *p0), c=colourCycle2[meas_obj_i], linewidth=0.5)
            axes[meas_set_i%gpr].set_title("%0.03f K" % temps[meas_set_i, meas_obj_i])
        except RuntimeError: #Caused by not fitting.
            fit_set.append(None)
    RVG_fits.append(fit_set)
    plt.tight_layout()
    if ((meas_set_i+1) % gpr == 0) or (meas_set_i == len(meas_objs)-1):
        plt.savefig(target3 + "01 " + str(meas_set_i) +" Fitting.png", bbox_inches="tight")

fits = np.array(RVG_fits)
fits.shape
fitted_dirac_points = fits[:,:,5]

# Give a quick plot of params
param_labels = ["Puddling", "Mu_e", "Mu_h", "Rho_S_e", "Rho_S_h", "Dirac_V", "Pow"]
for set in range(fits.shape[1]):
    param_list_len = fits.shape[2]
    plt.rcParams.update({'font.size': 3, "figure.figsize" : [param_list_len,1], 'figure.dpi':300})
    fig, (ax) = plt.subplots(1, param_list_len)
    for param in range(param_list_len):
        ax[param].plot(temps[:,set], fits[:,set,param], '.-', markersize=2, linewidth=0.5)
        ax[param].set_xlabel("Temperature (K)")
        ax[param].set_title(param_labels[param])
    plt.tight_layout()
    plt.savefig(target3 + "00 " + str(set) +" Fitting Params.png", bbox_inches="tight")

####################################################################################
####        Get a stack of different gate voltages to sample behaviour.      #####
####################################################################################
plt.rcParams.update({'font.size': 3, "figure.figsize" : [3,2], 'figure.dpi':300})
wide_vgs = np.sort(np.concatenate((np.linspace(-80,80,17), np.linspace(-9,9,10)), axis=0)) #10 voltages, 5-50 includive.
wide_rvg_sets = []      #Axis are [Temperature, Up/Down, Voltage Delta, Voltage | Resistance]
for set_i in range(len(meas_objs)):
    set=meas_objs[set_i]
    rvg_items = []
    for meas_obj_i in range(len(set)):
        meas_obj = set[meas_obj_i]
        #Extract sampled voltages away from dirac point = conductivity minimum.
        # min_i = meas_obj.conductivity_min()
        # min_v = meas_obj.raw_data[min_i,0]
        rvg = meas_obj.discrete_interpolated_voltages(gate_voltages=wide_vgs, center_voltage = fitted_dirac_points[set_i, meas_obj_i])
        rvg_items.append(rvg)
    wide_rvg_sets.append(rvg_items)
wide_rvg_sets = np.array(wide_rvg_sets)
wide_rvg_sets.shape

for i in range(wide_rvg_sets.shape[1]): #Upward and downward runs.
    #Select resisitivities
    res_i = wide_rvg_sets[:,i,:,1]
    #Setup colours
    cmap = cm.get_cmap("coolwarm")
    dims = [1j * a for a in np.shape(res_i)]
    m1, m2 = np.mgrid[0:1:dims[1], 1:1:dims[0]]
    c = cmap(m1)
    # cmat = np.c_[np.ones(len(res_i[:,j])) * wide_vgs[j] for j in len(wide_vgs)]
    #Create temp obj and plot
    rvt_obj = Meas_Temp_GatedResistance(temps=temps[:,i], vg=wide_vgs, resistivity=res_i)
    ax = rvt_obj.plot_Rho_vT(c=c)
    ax.set_yscale("log")
    handles, labels = ax.get_legend_handles_labels()
    ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.0,0.5), loc = "center")
    plt.savefig(target2 + "00 "+ str(i) + ud_labels[i%2] +" RTVg Raw Data.png", bbox_inches="tight")

####################################################################################
###                  Generate linear LA Phonon  fits                   ###
####################################################################################
FILE_DESCRIPTOR
###Take a small subset of gate voltages for good linear behaviour
##Devs4_04_run10_V01-V02     &   Devs4_04_run10_V08-V07
# vgs_linear = np.linspace(34,70,10) #6 voltages, 5-30 includive.
# t1,t2 = (5, 10)

##Devs4_04_Run04_V03-V04    &   Devs4_04_Run04_V01-V02
vgs_linear = np.linspace(34,62,8) #6 voltages, 5-30 includive.
vgs_linear
t1,t2 = (2, 5)

##Devs4_03_run04_V09-V08
# vgs_linear = np.linspace(34,70,10) #6 voltages, 5-30 includive.
# t1,t2 = (3, 10)

##Devs4_03_run04_V08-V07
# vgs_linear = np.linspace(34,70,10) #6 voltages, 5-30 includive.
# t1,t2 = (5, 10)

# t1,t2 = (2,6) #Devs4_4 run04
temps_linear = temps[t1:t2]
temps_linear
fit_dirac_linear = fitted_dirac_points[t1:t2]
linear_rvg_sets = []      #Axis are [Temperature, Up/Down, Voltage Delta, Voltage | Resistance]
for set_i in range(len(meas_objs[t1:t2])):
    set = meas_objs[t1:t2][set_i]
    rvg_items = []
    for meas_obj_i in range(len(set)): #sets based on temperature.
        meas_obj = set[meas_obj_i]
        #Extract sampled voltages away from dirac point = conductivity minimum.
        min_i = meas_obj.conductivity_min()
        min_v = meas_obj.raw_data[min_i,0]
        # rvg = meas_obj.discrete_sample_voltages(gate_voltages=vgs_linear, center_voltage = min_v, tollerance=0.01)
        rvg = meas_obj.discrete_interpolated_voltages(gate_voltages=vgs_linear, center_voltage = fit_dirac_linear[set_i, meas_obj_i])
        rvg_items.append(rvg)
    linear_rvg_sets.append(rvg_items)
linear_rvg_sets = np.array(linear_rvg_sets)
# Check for NaNs within subset
linear_rvg_sets.shape
nan_locs = np.where(np.isnan(linear_rvg_sets))
nan_locs
[np.unique(locs, axis=0) for locs in nan_locs]

##Fit data
rvt_obj_linear_u = Meas_Temp_GatedResistance(temps=temps_linear[:,0], vg=vgs_linear, resistivity=linear_rvg_sets[:,0,:,1])
rvt_obj_linear_d = Meas_Temp_GatedResistance(temps=temps_linear[:,1], vg=vgs_linear, resistivity=linear_rvg_sets[:,1,:,1])
params_lin_u, covar_lin_u = graphene.Graphene_Phonons.fit_Graphene_LA(rvt_obj_linear_u)
params_lin_d, covar_lin_d = graphene.Graphene_Phonons.fit_Graphene_LA(rvt_obj_linear_d)
#Combined fit
temps_linear_comb = np.concatenate((temps_linear[:,1],temps_linear[:,0]), axis=0)
linear_rvg_comb = np.concatenate((linear_rvg_sets[:,1,:,1], linear_rvg_sets[:,0,:,1]),axis=0)
rvt_obj_linear_c = Meas_Temp_GatedResistance(temps=temps_linear_comb, vg=vgs_linear, resistivity=linear_rvg_comb)
params_lin_c, covar_lin_c = graphene.Graphene_Phonons.fit_Graphene_LA(rvt_obj_linear_c)

##Plot data
plt.rcParams.update({'font.size': 3, "figure.figsize" : [3,4], 'figure.dpi':300})
# Generate offsets
sep = 30 #graphing separation in Ohms between gate voltages.
offsets_u = [params_lin_u[-(len(vgs_linear)) + i] + i * sep for i in range(len(vgs_linear))]
# offsets_d = [params_lin_d[-(len(vgs_linear)) + i] - i * sep for i in range(len(vgs_linear))]
# offsets_c = [params_lin_c[-(len(vgs_linear)) + i] - i * sep for i in range(len(vgs_linear))]
# offsets_u, offsets_d, offsets_c = (None, None, None)
# Data
lin_ax = rvt_obj_linear_u.plot_Rho_vT(c = colourCycle, labels=[ud_labels[0] + " " + str(vg) + " V" for vg in vgs_linear], offsets=offsets_u)
rvt_obj_linear_d.plot_Rho_vT(c=colourCycle2, ax=lin_ax, labels=[ud_labels[1] + " " + str(vg) + " V" for vg in vgs_linear], offsets=offsets_u)
# Fits
rvt_obj_linear_u.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_LA, params=tuple(params_lin_u), ax=lin_ax, c=colourCycle2, linewidth=0.3, singleLabel=ud_labels[0] + " fit LA", style="-.", offsets=offsets_u)
rvt_obj_linear_d.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_LA, params=tuple(params_lin_d), ax=lin_ax, c=colourCycle2, linewidth=0.3, singleLabel=ud_labels[1] + " fit LA", style="--", offsets=offsets_u)
rvt_obj_linear_c.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_LA, params=tuple(params_lin_c), ax=lin_ax, c=colourCycle2, linewidth=0.3, singleLabel="fit LA Combined", style="-", offsets=offsets_u)
lin_ax.set_ylabel(r"$\Delta\rho_{xx}$ - $\rho_{xx}^{T=0}$")
handles, labels = lin_ax.get_legend_handles_labels()
lin_ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.1), loc = "lower center")
# lin_ax.set_yscale("log")
#Setup table for params
param_headers = ["D$_a$",]
param_rows = [ud_labels[0] + " Fit", ud_labels[1] + " Fit", "Combined Fit"]
param_list = []
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (params_lin_u[0], math.sqrt(covar_lin_u[0,0]))]))
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (params_lin_d[0], math.sqrt(covar_lin_d[0,0]))]))
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (params_lin_c[0], math.sqrt(covar_lin_c[0,0]))]))
plt.table(cellText=param_list,
        rowLabels=param_rows,
        colLabels=param_headers,
        bbox=(1.175,0.6,0.2,0.4))
plt.savefig(target2 + "01 Phonons LA %0.02f" % temps_linear[0,0] + "K-%0.02f" % temps_linear[-1,0] + "K.png", bbox_inches="tight")

########################################################################################################
####            Take a small subset of gate voltages for good exponential + linear behaviour.     #####
########################################################################################################

FILE_DESCRIPTOR
###Take a large subset of gate voltages for good exponential behaviour
## Devs4_04_run10_V01-V02   &   Devs4_04_run10_V08-V07
# t1,t2 = (5, 21) #Devs4_4 run010, or Devs4_3 run04
# vgs_exp = np.linspace(34,70,10) #6 voltages, 5-30 includive.
# fit_dirac_exp = fitted_dirac_points[t1:t2]
# temps_exp = temps[t1:t2]
# exp_meas_objs = meas_objs[t1:t2]

## Devs4_04_Run04_V03-V04   &   Devs4_04_Run04_V01-V02
t1,t2 = (2, 10) #Devs4_4 run010, or Devs4_3 run04
vgs_exp = np.linspace(34,62,8) #6 voltages, 5-30 includive.
fit_dirac_exp = fitted_dirac_points[t1:t2]
temps_exp = temps[t1:t2]
temps_exp
exp_meas_objs = meas_objs[t1:t2]

## Devs4_03_run04_V09-V08
# t1,t2 = (5, 20) #Devs4_4 run010, or Devs4_3 run04
# vgs_exp = np.linspace(34,70,10) #6 voltages, 5-30 includive.
# fit_dirac_exp = fitted_dirac_points[t1:t2]
# temps_exp = temps[t1:t2]
# exp_meas_objs = meas_objs[t1:t2]

## Devs4_03_run04_V08-V07
# t1,t2 = (5,10)
# t3,t4 = (16, 22)
# vgs_exp = np.linspace(34,70,10) #6 voltages, 5-30 includive.
# fit_dirac_exp = np.concatenate((fitted_dirac_points[t1:t2],fitted_dirac_points[t3:t4]),axis=0)
# temps_exp = np.concatenate((temps[t1:t2], temps[t3:t4]),axis=0)
# exp_meas_objs = meas_objs[t1:t2] + meas_objs[t3:t4]

## Collect/interpolate resistances at voltages:
temps_exp
exp_rvg_sets = []      #Axis are [Temperature, Up/Down, Voltage Delta, Voltage | Resistance]
for set_i in range(len(exp_meas_objs)):
    set = exp_meas_objs[set_i]
    rvg_items = []
    for meas_obj_i in range(len(set)): #sets based on temperature.
        meas_obj = set[meas_obj_i]
        #Extract sampled voltages away from dirac point = conductivity minimum.
        # min_i = meas_obj.conductivity_min()
        # min_v = meas_obj.raw_data[min_i,0]
        # rvg = meas_obj.discrete_sample_voltages(gate_voltages=vgs_exp, center_voltage = min_v, tollerance=0.01)
        rvg = meas_obj.discrete_interpolated_voltages(gate_voltages=vgs_exp, center_voltage = fit_dirac_exp[set_i, meas_obj_i])
        rvg_items.append(rvg)
    exp_rvg_sets.append(rvg_items)
exp_rvg_sets = np.array(exp_rvg_sets)


# Check if any NANs
exp_rvg_sets.shape
# exp_rvg_sets
nan_locs = np.where(np.isnan(exp_rvg_sets))
nan_locs
[np.unique(locs, axis=0) for locs in nan_locs]

## Fit Data:
rvt_obj_u = Meas_Temp_GatedResistance(temps = temps_exp[:,0], vg=vgs_exp, resistivity=exp_rvg_sets[:,0,:,1])
rvt_obj_d = Meas_Temp_GatedResistance(temps = temps_exp[:,1], vg=vgs_exp, resistivity=exp_rvg_sets[:,1,:,1])
params_exp_u, covar_exp_u = graphene.Graphene_Phonons.fit_Graphene_on_SiO2(rvt_obj_u)
params_exp_d, covar_exp_d = graphene.Graphene_Phonons.fit_Graphene_on_SiO2(rvt_obj_d)
# Combined data:
temps_exp_comb = np.concatenate((temps_exp[:,1],temps_exp[:,0]), axis=0)
exp_rvg_comb = np.concatenate((exp_rvg_sets[:,1,:,1], exp_rvg_sets[:,0,:,1]),axis=0)
rvt_obj_comb = Meas_Temp_GatedResistance(temps = temps_exp_comb, vg=vgs_exp, resistivity=exp_rvg_comb)
params_exp_c, covar_exp_c = graphene.Graphene_Phonons.fit_Graphene_on_SiO2(rvt_obj_comb)
params_exp_c2, covar_exp_c2 = graphene.Graphene_Phonons.fit_Graphene_on_Dielectric(rvt_obj_comb)

params_exp_c
params_exp_c2

## Plot the data:
#Generate offsets by using original points and subtracting the same offset from all fits/series.
sep = 80 #graphing separation in Ohms between gate voltages.
offsets_exp_u = [params_exp_u[-(len(vgs_exp)) + i] + i * sep for i in range(len(vgs_exp))]
# offsets_u, offsets_d, offsets_c = (None, None, None)

exp_ax = rvt_obj_u.plot_Rho_vT(c = colourCycle2, labels=[ud_labels[0] + " " + str(vg) + " V" for vg in vgs_exp], offsets=offsets_exp_u)
rvt_obj_d.plot_Rho_vT(c=colourCycle, ax=exp_ax, labels=[ud_labels[1] + " " + str(vg) + " V" for vg in vgs_exp], offsets=offsets_exp_u)
# Plot Fits
rvt_obj_u.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_SiO2, params=tuple(params_exp_u), ax=exp_ax, c=colourCycle2, linewidth=0.3, singleLabel=ud_labels[0] + " fit SiO2", offsets=offsets_exp_u)
rvt_obj_d.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_SiO2, params=tuple(params_exp_d), ax=exp_ax, c=colourCycle, linewidth=0.3, singleLabel=ud_labels[1] + " fit SiO2", offsets=offsets_exp_u)
rvt_obj_comb.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_SiO2, params=tuple(params_exp_c), ax=exp_ax, c=colourCycle, points=30, linewidth=0.3, style="--", singleLabel="Combined Fit SiO2", offsets=offsets_exp_u)
rvt_obj_comb.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_Dielectric, params=tuple(params_exp_c2), ax=exp_ax, c=colourCycle, points=30, linewidth=0.3, style="-.", singleLabel="Combined Fit (E0=%0.02f meV)" % (params_exp_c2[3]*1000), offsets=offsets_exp_u)
# LEGEND
exp_ax.set_ylabel(r"$\Delta\rho_{xx}$ - $\rho_{xx}^{T=0}$ ($\Omega$)")
handles, labels = exp_ax.get_legend_handles_labels()
exp_ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.1), loc = "lower center")
# PARAMETERS:
param_headers = ["D$_a$", r"$\alpha$", "$B_1$", "E_0 (meV)"]
param_rows = [ud_labels[0] + " SiO2", ud_labels[1] + " SiO2", "Combined SiO2", "Combined Dielectric"]
param_list = []
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (params_exp_u[i], math.sqrt(covar_exp_u[i,i])) for i in range(3)] + ["-"]))
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (params_exp_d[i], math.sqrt(covar_exp_d[i,i])) for i in range(3)] + ["-"]))
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (params_exp_c[i], math.sqrt(covar_exp_c[i,i])) for i in range(3)] + ["-"]))
param_list.append(tuple(
    ["%0.03f $\pm$ %0.03f" % (params_exp_c2[i], math.sqrt(covar_exp_c2[i,i])) for i in range(3)] +
    ["%0.03f $\pm$ %0.03f" % (params_exp_c2[3] * 1e3, math.sqrt(covar_exp_c2[3,3])* 1e3)]
))

plt.table(cellText=param_list,
        rowLabels=param_rows,
        colLabels=param_headers,
        # bbox=(1.175,0.6,1.0,0.4))
        bbox=(1.175,0.6,0.7,0.4))
plt.savefig(target2 + "02 Phonons %0.02f" % temps_exp[0,0] + "K-%0.02f" % temps_exp[-1,0] + "K.png", bbox_inches="tight")
