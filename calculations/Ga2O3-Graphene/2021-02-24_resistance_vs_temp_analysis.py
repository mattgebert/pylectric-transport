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
from pylectric.geometries.hallbar_FET import Meas_GatedResistance, Meas_Temp_GatedResistance
from pylectric.materials import graphene, sio2

### ----------------------------------------------------------------------------------------------------------------------------- ###
### FIlE IO Properties ###
PRELUDE = "\\01 Ga2O3 Devices"
#               Devs4_04, prior to Ga2O3 Deposition                 #
# RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\02 Removed Outliers" #Folder for raw data
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V03-V04" # Bare Graphene [Gallium Area]
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V01-V02" # Bare Graphene

#               Devs4_04, after Ga2O3 Deposition                 #
RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\03 Data by date\\2020-12-15" #Folder for raw data
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V08-V07" # Gallium Oxide
FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V01-V02" # Bare Graphene

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
####        BEGIN ANALYSIS        ####
# files=[files[0]]
#Parse Data
file_data_objs = []
for file in files:
    data_obj = RVG_file(filepath=DIRPATH + "\\" + file)
    # fig = data_obj.plot_all()
    file_data_objs.append(data_obj)

###         Phonon RVGVT ANALYSIS           ###
#Create graphic directory for mobility
target2 = os.getcwd() + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR + "\\" + "01 Phonon Analysis" + "\\"
if not os.path.isdir(target2):
    os.mkdir(target2)

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
####        Get a stack of different gate voltages to sample behaviour.      #####
####################################################################################
wide_vgs = np.sort(np.concatenate((np.linspace(-80,80,17), np.linspace(-9,9,10)), axis=0)) #10 voltages, 5-50 includive.
wide_rvg_sets = []      #Axis are [Temperature, Up/Down, Voltage Delta, Voltage | Resistance]
for set in meas_objs:
    rvg_items = []
    for meas_obj in set:
        #Extract sampled voltages away from dirac point = conductivity minimum.
        min_i = meas_obj.conductivity_min()
        min_v = meas_obj.raw_data[min_i,0]
        rvg = meas_obj.discrete_sample_voltages(gate_voltages=wide_vgs, center_voltage = min_v, tollerance=0.01)
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

plt.rcParams.update({'font.size': 3, "figure.figsize" : [3,10], 'figure.dpi':300})

##Take a small subset of gate voltages for good linear behaviour
vgs_linear = np.concatenate((np.linspace(-60,-40,5),np.linspace(40,60,5))) #6 voltages, 5-30 includive.
# vgs_linear = np.linspace(-30,30,6) #6 voltages, 5-30 includive.
t1,t2 = (6, 11) #Devs4_4 run010, or Devs4_3 run04
# t1,t2 = (2,6) #Devs4_4 run04
temps_linear = temps[t1:t2]
linear_rvg_sets = []      #Axis are [Temperature, Up/Down, Voltage Delta, Voltage | Resistance]
for set in meas_objs[t1:t2]:
    rvg_items = []
    for meas_obj in set: #sets based on temperature.
        #Extract sampled voltages away from dirac point = conductivity minimum.
        min_i = meas_obj.conductivity_min()
        min_v = meas_obj.raw_data[min_i,0]
        rvg = meas_obj.discrete_sample_voltages(gate_voltages=vgs_linear, center_voltage = min_v, tollerance=0.01)
        rvg_items.append(rvg)
    linear_rvg_sets.append(rvg_items)
linear_rvg_sets = np.array(linear_rvg_sets)
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
# Data
lin_ax = rvt_obj_linear_u.plot_Rho_vT(c = colourCycle, labels=[ud_labels[0] + " " + str(vg) + " V" for vg in vgs_linear])
rvt_obj_linear_d.plot_Rho_vT(c=colourCycle2, ax=lin_ax, labels=[ud_labels[1] + " " + str(vg) + " V" for vg in vgs_linear])
# Fits
rvt_obj_linear_u.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_LA, params=tuple(params_lin_u), ax=lin_ax, c=colourCycle2, linewidth=0.3, singleLabel=ud_labels[0] + " fit LA", style="-.")
rvt_obj_linear_d.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_LA, params=tuple(params_lin_d), ax=lin_ax, c=colourCycle2, linewidth=0.3, singleLabel=ud_labels[1] + " fit LA", style="--")
rvt_obj_linear_c.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_LA, params=tuple(params_lin_c), ax=lin_ax, c=colourCycle2, linewidth=0.3, singleLabel="fit LA Combined", style="-")
handles, labels = lin_ax.get_legend_handles_labels()
lin_ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0), loc = "lower center")
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
linear_rvg_sets.shape
linear_rvg_sets[0,0,4,1]
linear_rvg_sets[7,0,4,1]


########################################################################################################
####            Take a small subset of gate voltages for good exponential + linear behaviour.     #####
########################################################################################################

##Take a large subset of gate voltages for good exponential behaviour
t1,t2 = (3, 21) #Devs4_4 run010, or Devs4_3 run04
# t1,t2 = (2,8) #Devs4_4 run04
vgs_exp = np.linspace(10,30,6) #6 voltages, 5-30 includive.

#Collect voltages:
temps_exp = temps[t1:t2]
exp_rvg_sets = []      #Axis are [Temperature, Up/Down, Voltage Delta, Voltage | Resistance]
for set in meas_objs[t1:t2]:
    rvg_items = []
    for meas_obj in set: #sets based on temperature.
        #Extract sampled voltages away from dirac point = conductivity minimum.
        min_i = meas_obj.conductivity_min()
        min_v = meas_obj.raw_data[min_i,0]
        rvg = meas_obj.discrete_sample_voltages(gate_voltages=vgs_exp, center_voltage = min_v, tollerance=0.01)
        rvg_items.append(rvg)
    exp_rvg_sets.append(rvg_items)
exp_rvg_sets = np.array(exp_rvg_sets)

# Fit Data:
rvt_obj_u = Meas_Temp_GatedResistance(temps = temps_exp[:,0], vg=vgs_exp, resistivity=exp_rvg_sets[:,0,:,1])
rvt_obj_d = Meas_Temp_GatedResistance(temps = temps_exp[:,1], vg=vgs_exp, resistivity=exp_rvg_sets[:,1,:,1])
paramsu, covaru = graphene.Graphene_Phonons.fit_Graphene_on_SiO2(rvt_obj_u)
paramsd, covard = graphene.Graphene_Phonons.fit_Graphene_on_SiO2(rvt_obj_d)
# Combined data:
temps_exp_comb = np.concatenate((temps_exp[:,1],temps_exp[:,0]), axis=0)
exp_rvg_comb = np.concatenate((exp_rvg_sets[:,1,:,1], exp_rvg_sets[:,0,:,1]),axis=0)
rvt_obj_comb = Meas_Temp_GatedResistance(temps = temps_exp_comb, vg=vgs_exp, resistivity=exp_rvg_comb)
paramsc, covarc = graphene.Graphene_Phonons.fit_Graphene_on_SiO2(rvt_obj_comb)
paramsc2, covarc2 = graphene.Graphene_Phonons.fit_Graphene_on_Dielectric(rvt_obj_comb)

# Plot the data:
exp_ax = rvt_obj_u.plot_Rho_vT(c = colourCycle2, labels=[ud_labels[0] + " " + str(vg) + " V" for vg in vgs_exp])
rvt_obj_d.plot_Rho_vT(c=colourCycle, ax=exp_ax, labels=[ud_labels[1] + " " + str(vg) + " V" for vg in vgs_exp])
# Plot Fits
rvt_obj_u.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_SiO2, params=tuple(paramsu), ax=exp_ax, c=colourCycle2, linewidth=0.3, singleLabel=ud_labels[0] + " fit SiO2")
rvt_obj_d.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_SiO2, params=tuple(paramsd), ax=exp_ax, c=colourCycle, linewidth=0.3, singleLabel=ud_labels[1] + " fit SiO2")
rvt_obj_comb.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_SiO2, params=tuple(paramsc), ax=exp_ax, c=colourCycle, points=30, linewidth=0.3, style="--", singleLabel="Combined Fit SiO2")
rvt_obj_comb.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_Dielectric, params=tuple(paramsc2), ax=exp_ax, c=colourCycle, points=30, linewidth=0.3, style="-.", singleLabel="Combined Fit (E0=%0.02f meV)" % (paramsc2[3]*1000))
# LEGEND
handles, labels = exp_ax.get_legend_handles_labels()
exp_ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0), loc = "lower center")
# PARAMETERS:
param_headers = ["D$_a$", r"$\alpha$", "$B_1$", "E_0 (meV)"]
param_rows = [ud_labels[0] + " SiO2", ud_labels[1] + " SiO2", "Combined SiO2", "Combined Dielectric"]
param_list = []
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (paramsu[i], math.sqrt(covaru[i,i])) for i in range(3)] + ["-"]))
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (paramsd[i], math.sqrt(covard[i,i])) for i in range(3)] + ["-"]))
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (paramsc[i], math.sqrt(covarc[i,i])) for i in range(3)] + ["-"]))
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (paramsc2[i], math.sqrt(covarc2[i,i])) for i in range(4)]))
# param_list.append(tuple(["%0.03f" % i for i in paramsu[0:3]] + ["-"]))
# param_list.append(tuple(["%0.03f" % i for i in paramsd[0:3]] + ["-"]))
# param_list.append(tuple(["%0.03f" % i for i in paramsc[0:3]] + ["-"]))
# param_list.append(tuple(["%0.03f" % i for i in paramsc2[0:3]] + ["%0.03f" % (paramsc2[3]*1000)]))

plt.table(cellText=param_list,
        rowLabels=param_rows,
        colLabels=param_headers,
        bbox=(1.175,0.6,0.6,0.4))
plt.savefig(target2 + "02 Phonons %0.02f" % temps_exp[t1,0] + "K-%0.02f" % temps_exp[t2,0] + "K.png", bbox_inches="tight")
