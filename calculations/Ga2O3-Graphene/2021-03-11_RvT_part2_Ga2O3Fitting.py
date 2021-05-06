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
####                        Import files and format data.                          #####
########################################################################################
PRELUDE = "\\01 Ga2O3 Devices"
#               Devs4_04, prior to Ga2O3 Deposition                 #
# RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\02 Removed Outliers" #Folder for raw data
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V03-V04" # Bare Graphene [Gallium Area]
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V01-V02" # Bare Graphene

#               Devs4_04, after Ga2O3 Deposition                 #
RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\03 Data by date\\2020-12-15" #Folder for raw data
FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V08-V07" # Gallium Oxide
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
# files = files[0:34] #TODO REMOVE specific for Devs4_03_Run04_V09-V08 and Devs4_03_Run04_V08-V07

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
target2 = os.getcwd() + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR + "\\" + "05 Phonon Analysis (Ga2O3)" + "\\"
target3 = target2 + "\\" + "Dirac Point Fitting" + "\\"
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
temps_orig = [] #Get average temperatures
temps_err = [] #Get average deviation of temperatures
meas_objs = [] #Setup sets of discritized set of gate voltages and resistances
# obj = file_data_objs[0]
for obj in file_data_objs:
    #Get raw measurement object
    datasets = obj.split_dataset_by_voltage_turning_point()
    t = []
    t_err = []
    data_items = []
    #Setup sampling voltages
    i = 0
    for i in range(len(datasets)):
        #Calculate average temperature of dataset
        t.append(np.mean(datasets[i][:,4]))
        t_err.append(np.std(datasets[i][:,4]))
        #Create gated measurement object.
        meas_obj = Meas_GatedResistance(data=datasets[i], Cg=SiO2_Cg, L=200, W=400)
        data_items.append(meas_obj)
    temps_orig.append(t)
    temps_err.append(t_err)
    meas_objs.append(data_items)
temps_orig = np.array(temps_orig)
temps = temps_orig.copy()
temps_err = np.array(temps_err)
temps.shape
temps_err.shape
#average over up/down
temps = np.average(temps, axis=1)
temps_err =  np.sqrt(np.sum(np.power(temps_err, 2),axis=1)) #propogate uncertainty.


####################################################################################
####   Get fits for conductivity behaviour of graphene. Get gate voltages.     #####
####################################################################################
plt.rcParams.update({'font.size': 3, "figure.figsize" : [6,1.5], 'figure.dpi':300})
RVG_fits = []
# meas_set_i=0
for meas_set_i in range(len(meas_objs)):
    meas_set = meas_objs[meas_set_i]
    fit_set = []
    #Setup graphing.
    gpr = 4 #graphs per row
    if meas_set_i % gpr == 0:
        fig, (axes) = plt.subplots(1, 4)
    for meas_obj_i in range(len(meas_set)):
        # meas_obj_i = 0
        meas_obj = meas_set[meas_obj_i]
        try:
            d_v, quad, fit_voltages = graphene.Graphene_Gated.fit_min_cond_quadratic(meas_obj)
            fit_set.append([d_v,quad])
            meas_obj.plot_Sigma_vG(ax=axes[meas_set_i%gpr], c=colourCycle[meas_obj_i])
            axes[meas_set_i%gpr].plot(fit_voltages, quad(fit_voltages), c=colourCycle2[meas_obj_i], linewidth=0.5)
            axes[meas_set_i%gpr].set_title("%0.03f K" % temps[meas_set_i])
        except RuntimeError: #Caused by not fitting.
            fit_set.append(None)
    axes[meas_set_i%gpr].get_figure()
    RVG_fits.append(fit_set)
    plt.tight_layout()
    if ((meas_set_i+1) % gpr == 0) or (meas_set_i == len(meas_objs)-1):
        plt.savefig(target3 + "01 " + str(meas_set_i) +" Fitting.png", bbox_inches="tight")
RVG_fits
fits = np.array(RVG_fits)
fits.shape
fitted_dirac_points = fits[:,:,0]
fitted_dirac_points

def f(x, *coefs):
    a0, a1, a2 =coefs
    return a0 + a1 * (x) + a2*(x**2)

quad = fits[0,0,1]
quad(-30)
quad.coef

quad2 = quad.convert(domain=[-1,1])

f(-30, *quad2.coef)

#
# # Give a quick plot of params
# param_labels = ["Puddling", "Mu_e", "Mu_h", "Rho_S_e", "Rho_S_h", "Dirac_V", "Pow"]
# for set in range(fits.shape[1]):
#     param_list_len = fits.shape[2]
#     plt.rcParams.update({'font.size': 3, "figure.figsize" : [param_list_len,1], 'figure.dpi':300})
#     fig, (ax) = plt.subplots(1, param_list_len)
#     for param in range(param_list_len):
#         ax[param].plot(temps_orig[:,set], fits[:,set,param], '.-', markersize=2, linewidth=0.5)
#         ax[param].set_xlabel("Temperature (K)")
#         ax[param].set_title(param_labels[param])
#     plt.tight_layout()
#     plt.savefig(target3 + "00 " + str(set) +" Fitting Params.png", bbox_inches="tight")

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
tempSet = np.zeros((wide_rvg_sets.shape[0], 1, wide_rvg_sets.shape[2], wide_rvg_sets.shape[3]))
tempSet[:,0,:,:] = np.average(wide_rvg_sets, axis=1)
tempSet.shape
wide_rvg_sets = tempSet
wide_rvg_sets

# Save data in a formattable way.
len_1D = wide_rvg_sets.shape[0] * wide_rvg_sets.shape[2]
wide_1D_data = np.ones((len_1D,3))
wide_1D_data[:,1:3] = wide_rvg_sets.reshape((len_1D,2))
for temp_i in range(len(temps)):
    temp = temps[temp_i]
    wide_1D_data[temp_i * len(wide_vgs): (temp_i + 1) * len(wide_vgs),0] *= temp
wide_1D_data
np.savetxt(target2 + FILE_DESCRIPTOR + "_RhoVG-Extraction_Wide.txt", wide_1D_data, delimiter=",", header="Temperature (K),Gate Voltage (V),Resisitivity (Ohms)")

#Select resisitivities
res_i = wide_rvg_sets[:,0,:,1]
#Setup colours
cmap = cm.get_cmap("coolwarm")
dims = [1j * a for a in np.shape(res_i)]
m1, m2 = np.mgrid[0:1:dims[1], 1:1:dims[0]]
c = cmap(m1)
# cmat = np.c_[np.ones(len(res_i[:,j])) * wide_vgs[j] for j in len(wide_vgs)]
#Create temp obj and plot
rvt_obj = Meas_Temp_GatedResistance(temps=temps[:], vg=wide_vgs, resistivity=res_i)
ax = rvt_obj.plot_Rho_vT(c=c)
ax.set_yscale("log")
handles, labels = ax.get_legend_handles_labels()
ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.0,0.5), loc = "center")
plt.savefig(target2 + "00 RTVg Ave Data.png", bbox_inches="tight")

####################################################################################
###                  Generate linear LA Phonon  fits                   ###
####################################################################################
FILE_DESCRIPTOR
###Take a small subset of gate voltages for good linear behaviour
##Devs4_04_run10_V01-V02     &   Devs4_04_run10_V08-V07
vgs_linear = np.linspace(42,70,8) #6 voltages, 5-30 includive.
# vgs_linear = np.linspace(34,70,10) #6 voltages, 5-30 includive.
# vgs_linear = np.linspace(10,70,16) #6 voltages, 5-30 includive.
# vgs_linear = np.linspace(18,70,14) #6 voltages, 5-30 includive.
# vgs_linear = np.linspace(26,70,12) #6 voltages, 5-30 includive.
# vgs_linear
t1,t2 = (5, 10)

##Devs4_04_Run04_V03-V04    &   Devs4_04_Run04_V01-V02
# vgs_linear = np.linspace(34,62,8) #6 voltages, 5-30 includive.
# vgs_linear
# t1,t2 = (2, 5)

##Devs4_03_run04_V09-V08
# vgs_linear = np.linspace(34,66,9) #6 voltages, 5-30 includive.
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
tempSet = np.zeros((linear_rvg_sets.shape[0], 1, linear_rvg_sets.shape[2], linear_rvg_sets.shape[3]))
tempSet[:,0,:,:] = np.average(linear_rvg_sets, axis=1)
tempSet.shape
linear_rvg_sets = tempSet


# Save data in a formattable way.
len_1D = linear_rvg_sets.shape[0] * linear_rvg_sets.shape[2]
linear_1D_data = np.ones((len_1D,3))
linear_1D_data[:,1:3] = linear_rvg_sets.reshape((len_1D,2))
for temp_i in range(len(temps_linear)):
    temp = temps_linear[temp_i]
    linear_1D_data[temp_i * len(vgs_linear): (temp_i + 1) * len(vgs_linear),0] *= temp
linear_1D_data
np.savetxt(target2 + FILE_DESCRIPTOR + "_RhoVG-Extraction_Linear.txt", linear_1D_data, delimiter=",", header="Temperature (K),Gate Voltage (V),Resisitivity (Ohms)")

# Check for nans before fitting.
nan_locs = np.where(np.isnan(linear_rvg_sets))
nan_locs
[np.unique(locs, axis=0) for locs in nan_locs]

##Fit data
rvt_obj_linear = Meas_Temp_GatedResistance(temps=temps_linear, vg=vgs_linear, resistivity=linear_rvg_sets[:,0,:,1])
params_lin, covar_lin = graphene.Graphene_Phonons.fit_Graphene_LA(rvt_obj_linear)

##Plot data
plt.rcParams.update({'font.size': 3, "figure.figsize" : [2,4], 'figure.dpi':300})
# Generate offsets
sep = 30 #graphing separation in Ohms between gate voltages.
offsets = [params_lin[-(len(vgs_linear)) + i] + i * sep for i in range(len(vgs_linear))]
# offsets_d = [params_lin_d[-(len(vgs_linear)) + i] - i * sep for i in range(len(vgs_linear))]
# offsets_c = [params_lin_c[-(len(vgs_linear)) + i] - i * sep for i in range(len(vgs_linear))]
# offsets_u, offsets_d, offsets_c = (None, None, None)
# Data
lin_ax = rvt_obj_linear.plot_Rho_vT(c = colourCycle + colourCycle, labels=["" + str(vg) + " V" for vg in vgs_linear], offsets=offsets)
# Fits
rvt_obj_linear.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_LA, params=tuple(params_lin), ax=lin_ax, c=colourCycle2 + colourCycle2, linewidth=0.3, singleLabel="fit LA", style="-.", offsets=offsets)
lin_ax.set_ylabel(r"$\Delta\rho_{xx}$ - $\rho_{xx}^{T=0}$")
handles, labels = lin_ax.get_legend_handles_labels()
lin_ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.1), loc = "lower center")
# lin_ax.set_yscale("log")
#Setup table for params
param_headers = ["D$_a$",]
param_rows = ["Ave Fit"]
param_list = []
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (params_lin[0], math.sqrt(covar_lin[0,0]))]))
plt.table(cellText=param_list,
        rowLabels=param_rows,
        colLabels=param_headers,
        bbox=(1.175,0.6,0.2,0.4))
plt.savefig(target2 + "01 Phonons LA %0.02f" % temps_linear[0] + "K-%0.02f" % temps_linear[-1] + "K %0.01fV" % vgs_linear[0] + "-%0.01fV.png" % vgs_linear[-1], bbox_inches="tight")

########################################################################################################
####            Take a small subset of gate voltages for good exponential + linear behaviour.     #####
########################################################################################################

FILE_DESCRIPTOR
###Take a large subset of gate voltages for good exponential behaviour
## Devs4_04_run10_V01-V02   &

## Devs4_04_run10_V08-V07
t1,t2 = (5, 23) #Devs4_4 run010, or Devs4_3 run04
# t1,t2 = (5, 21) #Devs4_4 run010, or Devs4_3 run04
# vgs_exp = np.linspace(10,70,16) #6 voltages, 5-30 includive.
# vgs_exp = np.linspace(30,70,11) #6 voltages, 5-30 includive.
# vgs_exp = np.linspace(26,70,12) #6 voltages, 5-30 includive.
# vgs_exp = np.linspace(34,70,10) #6 voltages, 5-30 includive.
# vgs_exp = np.linspace(38,70,9) #6 voltages, 5-30 includive.
# vgs_exp = np.linspace(42,70,8) #6 voltages, 5-30 includive.
vgs_exp = np.linspace(10,66,15) #6 voltages, 5-30 includive.
# vgs_exp = np.linspace(30,66,10) #6 voltages, 5-30 includive.
# vgs_exp = np.linspace(26,66,11) #6 voltages, 5-30 includive.
# vgs_exp = np.linspace(34,66,9) #6 voltages, 5-30 includive.
# vgs_exp = np.linspace(38,66,8) #6 voltages, 5-30 includive.
# vgs_exp = np.linspace(42,66,7) #6 voltages, 5-30 includive.
vgs_exp
vgs_exp
fit_dirac_exp = fitted_dirac_points[t1:t2]
temps_exp = temps[t1:t2]
exp_meas_objs = meas_objs[t1:t2]

## Devs4_04_Run04_V03-V04   &   Devs4_04_Run04_V01-V02
# t1,t2 = (2, 10) #Devs4_4 run010, or Devs4_3 run04
# vgs_exp = np.linspace(34,62,8) #6 voltages, 5-30 includive.
# fit_dirac_exp = fitted_dirac_points[t1:t2]
# temps_exp = temps[t1:t2]
# temps_exp
# exp_meas_objs = meas_objs[t1:t2]

# ## Devs4_03_run04_V09-V08
# t1,t2 = (5, 20) #Devs4_4 run010, or Devs4_3 run04
# vgs_exp = np.linspace(34,66,9) #6 voltages, 5-30 includive.
# vgs_exp = np.linspace(34,70,10) #6 voltages, 5-30 includive.
# fit_dirac_exp = fitted_dirac_points[t1:t2]
# temps_exp = temps[t1:t2]
# exp_meas_objs = meas_objs[t1:t2]

## Devs4_03_run04_V08-V07
# t1,t2 = (3,10)
# t3,t4 = (16, 22)
# vgs_exp = np.linspace(34,66,9) #6 voltages, 5-30 includive.
# fit_dirac_exp = np.concatenate((fitted_dirac_points[t1:t2],fitted_dirac_points[t3:t4]),axis=0)
# temps_exp = np.concatenate((temps[t1:t2], temps[t3:t4]),axis=0)
# exp_meas_objs = meas_objs[t1:t2] + meas_objs[t3:t4]

## Collect/interpolate resistances at voltages:
# temps_exp
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
exp_rvg_sets.shape
# Average over hysteresis.
tempSet = np.zeros((exp_rvg_sets.shape[0], 1, exp_rvg_sets.shape[2], exp_rvg_sets.shape[3]))
tempSet[:,0,:,:] = np.average(exp_rvg_sets, axis=1)
tempSet.shape
exp_rvg_sets = tempSet

# Save data in a formattable way.
len_1D = exp_rvg_sets.shape[0] * exp_rvg_sets.shape[2]
exp_1D_data = np.ones((len_1D,3))
exp_1D_data[:,1:3] = exp_rvg_sets.reshape((len_1D,2))
for temp_i in range(len(temps_exp)):
    temp = temps_exp[temp_i]
    exp_1D_data[temp_i * len(vgs_exp): (temp_i + 1) * len(vgs_exp),0] *= temp
exp_1D_data
np.savetxt(target2 + FILE_DESCRIPTOR + "_RhoVG-Extraction_Exp.txt", exp_1D_data, delimiter=",", header="Temperature (K),Gate Voltage (V),Resisitivity (Ohms)")

# Check if any NANs before fitting.
exp_rvg_sets.shape
# exp_rvg_sets
nan_locs = np.where(np.isnan(exp_rvg_sets))
nan_locs
[np.unique(locs, axis=0) for locs in nan_locs]

## Fit Data:
rvt_obj = Meas_Temp_GatedResistance(temps = temps_exp[:], vg=vgs_exp, resistivity=exp_rvg_sets[:,0,:,1])
params_exp, covar_exp = graphene.Graphene_Phonons.fit_Graphene_on_SiO2(rvt_obj)
params_exp_ga2o3, covar_exp_ga2o3 = graphene.Graphene_Phonons.fit_Graphene_between_Sio2_Ga2O3(rvt_obj)
params_exp_e0, covar_exp_e0 = graphene.Graphene_Phonons.fit_Graphene_on_Dielectric(rvt_obj)

## Plot the data:
plt.rcParams.update({'font.size': 3, "figure.figsize" : [2,4], 'figure.dpi':300})
#Generate offsets by using original points and subtracting the same offset from all fits/series.
sep = 80 #graphing separation in Ohms between gate voltages.
offsets_exp = [params_exp[-(len(vgs_exp)) + i] + i * sep for i in range(len(vgs_exp))]
# offsets_u, offsets_d, offsets_c = (None, None, None)
exp_ax = rvt_obj.plot_Rho_vT(c = colourCycle + colourCycle, labels=[str(vg) + " V" for vg in vgs_exp], offsets=offsets_exp)
# Plot Fits
rvt_obj.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_SiO2, params=tuple(params_exp), ax=exp_ax, c=colourCycle2 + colourCycle2, linewidth=0.3, singleLabel="Ave fit SiO2", offsets=offsets_exp, T_max=350)
rvt_obj.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_between_SiO2_Ga2O3, params=tuple(params_exp_ga2o3), ax=exp_ax, c=colourCycle + colourCycle, points=30, linewidth=0.3, style="-.", singleLabel="Ga2O3 Fit", offsets=offsets_exp, T_max=350)
rvt_obj.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_Dielectric, params=tuple(params_exp_e0), ax=exp_ax, c=colourCycle + colourCycle, points=30, linewidth=0.3, style="--", singleLabel="Ave Fit (E0=%0.02f meV)" % (params_exp_e0[3]*1000), offsets=offsets_exp, T_max=350)
# LEGEND
exp_ax.set_ylabel(r"$\Delta\rho_{xx}$ - $\rho_{xx}^{T=0}$ ($\Omega$)")
handles, labels = exp_ax.get_legend_handles_labels()
exp_ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.1,0.1), loc = "lower center")
# PARAMETERS:
param_headers = ["D$_a$", r"$\alpha$", "$B_1$", "E_0 (meV)"]
param_rows = ["Bare SiO2", "Generic Dielectric", "Ga2O3 Fit"]
param_list = []
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (params_exp[i], math.sqrt(covar_exp[i,i])) for i in range(3)] + ["-"]))
param_list.append(tuple(
    ["%0.03f $\pm$ %0.03f" % (params_exp_e0[i], math.sqrt(covar_exp_e0[i,i])) for i in range(3)] +
    ["%0.03f $\pm$ %0.03f" % (params_exp_e0[3] * 1e3, math.sqrt(covar_exp_e0[3,3])* 1e3)]
))
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (params_exp_ga2o3[i], math.sqrt(covar_exp_ga2o3[i,i])) for i in range(3)] + ["-"]))

plt.table(cellText=param_list,
        rowLabels=param_rows,
        colLabels=param_headers,
        # bbox=(1.175,0.6,1.0,0.4))
        bbox=(1.225,0.6,1.0,0.4))
        # bbox=(1.175,0.6,0.7,0.4))
plt.savefig(target2 + "02 Phonons %0.02f" % temps_exp[0] + "K-%0.02f" % temps_exp[-1] + "K %0.01fV" % vgs_exp[0] + "-%0.01fV.png" % vgs_exp[-1], bbox_inches="tight")
