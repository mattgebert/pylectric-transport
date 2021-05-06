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
####                        Setup graphing for Phonon Analysis                     #####
########################################################################################

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
####                        Import files and format data.                          #####
########################################################################################
PRELUDE = "\\01 Ga2O3 Devices"
RAW_DATA_DIR = "\\99 Sampled Data" #Folder for raw data

#               Devs4_04, after Ga2O3 Deposition                 #
# FILE_DESCRIPTOR1 = "hBN-Gr_Devs4_04_run10_V01-V02_RhoVG-Extraction_Exp 60.00K-210.00K 26.0V-66.0V.txt" # Bare Graphene
# FILE_DESCRIPTOR2 = "hBN-Gr_Devs4_04_run10_V08-V07_RhoVG-Extraction_Exp 60.00K-210.00K 26.0V-66.0V.txt" # Gallium Oxide
FILE_DESCRIPTOR1 = "hBN-Gr_Devs4_04_run10_V01-V02_RhoVG-Extraction_Exp 60.00K-220.00K 26.0V-66.0V.txt" # Bare Graphene
FILE_DESCRIPTOR2 = "hBN-Gr_Devs4_04_run10_V08-V07_RhoVG-Extraction_Exp 60.00K-220.00K 26.0V-66.0V.txt" # Gallium Oxide
# FILE_DESCRIPTOR1 = "hBN-Gr_Devs4_04_run10_V01-V02_RhoVG-Extraction_Exp 60.00K-230.00K 26.0V-66.0V.txt" # Bare Graphene
# FILE_DESCRIPTOR2 = "hBN-Gr_Devs4_04_run10_V08-V07_RhoVG-Extraction_Exp 60.00K-230.00K 26.0V-66.0V.txt" # Gallium Oxide
# FILE_DESCRIPTOR1 = "hBN-Gr_Devs4_04_run10_V01-V02_RhoVG-Extraction_Exp 60.00K-210.00K 30.0V-66.0V.txt" # Bare Graphene
# FILE_DESCRIPTOR2 = "hBN-Gr_Devs4_04_run10_V08-V07_RhoVG-Extraction_Exp 60.00K-210.00K 30.0V-66.0V.txt" # Gallium Oxide
# FILE_DESCRIPTOR1 = "hBN-Gr_Devs4_04_run10_V01-V02_RhoVG-Extraction_Exp 60.00K-220.00K 30.0V-66.0V.txt" # Bare Graphene
# FILE_DESCRIPTOR2 = "hBN-Gr_Devs4_04_run10_V08-V07_RhoVG-Extraction_Exp 60.00K-220.00K 30.0V-66.0V.txt" # Gallium Oxide
# FILE_DESCRIPTOR1 = "hBN-Gr_Devs4_04_run10_V01-V02_RhoVG-Extraction_Exp 60.00K-230.00K 30.0V-66.0V.txt" # Bare Graphene
# FILE_DESCRIPTOR2 = "hBN-Gr_Devs4_04_run10_V08-V07_RhoVG-Extraction_Exp 60.00K-230.00K 30.0V-66.0V.txt" # Gallium Oxide


### Processing Properties ###

### Load datafile and load values.
target1 = os.getcwd() + PRELUDE + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR1
data1 = np.loadtxt(target1,delimiter=",")
data1.shape
temps1 = np.unique(np.round(data1[:,0],decimals=3))
vgs1 = np.unique(np.round(data1[:,1],decimals=3))
temps1
vgs1

target2 = os.getcwd() + PRELUDE + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR2
data2 = np.loadtxt(target2,delimiter=",")
data2.shape
temps2 = np.unique(np.round(data2[:,0],decimals=3))
vgs2 = np.unique(np.round(data2[:,1],decimals=3))

#Reshape data into 2D array, using resistivities & make data objects.
data1_reshaped = np.reshape(data1[:,2], (len(temps1), len(vgs1)))
data1_reshaped.shape
temps1.shape
vgs1.shape
rvt_obj1 = Meas_Temp_GatedResistance(temps = temps1, vg=vgs1, resistivity=data1_reshaped)

data2_reshaped = np.reshape(data2[:,2], (len(temps2), len(vgs2)))
data2_reshaped.shape
temps2.shape
vgs2.shape
rvt_obj2 = Meas_Temp_GatedResistance(temps = temps2, vg=vgs2, resistivity=data2_reshaped)

saveTarget = os.getcwd() + PRELUDE + RAW_DATA_DIR + "\\"
########################################################################################################
####                            Conduct Fitting on subset data                                     #####
########################################################################################################

## Conduct fits
#Bare graphene sample
obj1_params_exp_sio2, obj1_covar_exp_sio2 = graphene.Graphene_Phonons.fit_ROP_Gr_on_Sio2(rvt_obj1)

#Gallium covered sample
obj2_params_exp_sio2, obj2_covar_exp_sio2 = graphene.Graphene_Phonons.fit_ROP_Gr_on_Sio2(rvt_obj2)
obj2_params_exp_ga2o3, obj2_covar_exp_ga2o3 = graphene.Graphene_Phonons.fit_ROP_Gr_between_Sio2_Ga2O3(rvt_obj2)
obj2_params_exp_generic, obj2_covar_exp_generic = graphene.Graphene_Phonons.fit_ROP_Gr_on_Dielectric(rvt_obj2)

## Plot the data:
plt.rcParams.update({'font.size': 3, "figure.figsize" : [2,4], 'figure.dpi':300})

#Generate offsets by using original points and subtracting the same offset from all fits/series.
sep = 80 #graphing separation in Ohms between gate voltages.

#Assuming obj1 and obj2 have the same data lengths:
obj1_offsets_exp_sio2 = [obj1_params_exp_sio2[-(len(vgs1)) + i] + i * sep for i in range(len(vgs1))]
obj2_offsets_exp_sio2 = [obj2_params_exp_sio2[-(len(vgs2)) + i] + i * sep for i in range(len(vgs2))]
kwargs={"marker":"x","s":0.1}
exp_ax = rvt_obj1.plot_Rho_vT(c = colourCycle + colourCycle, labels=[str(vg) + " V" for vg in vgs1], offsets=obj1_offsets_exp_sio2, **kwargs)
kwargs={"marker":"+","s":0.1}
rvt_obj2.plot_Rho_vT(ax=exp_ax, c = colourCycle2 + colourCycle2, singleLabel="Ga$_2$O$_3$/Gr/SiO$_2$ Data", offsets=obj2_offsets_exp_sio2, **kwargs)

# Plot Fits
rvt_obj1.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_ROP_Gr_on_SiO2, params=tuple(obj1_params_exp_sio2), ax=exp_ax, c=colourCycle + colourCycle, points=30, linewidth=0.3, style="--", singleLabel="SiO$_2$/Gr: SiO$_2$ Fit", offsets=obj1_offsets_exp_sio2, T_max=250)
# rvt_obj2.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_ROP_Gr_on_SiO2, params=tuple(obj2_params_exp_sio2), ax=exp_ax, c=colourCycle2 + colourCycle2, points=30, linewidth=0.3, style="-.", singleLabel="SiO$_2$/Gr/Ga$_2$O$_3$: SiO$_2$ Fit", offsets=obj2_offsets_exp_sio2, T_max=250)
rvt_obj2.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_ROP_Gr_between_SiO2_Ga2O3, params=tuple(obj2_params_exp_ga2o3), ax=exp_ax, c=colourCycle2 + colourCycle2, points=30, linewidth=0.3, style="-", singleLabel="SiO$_2$/Gr/Ga$_2$O$_3$: Ga$_2$O$_3$ Fit", offsets=obj2_offsets_exp_sio2, T_max=250)
rvt_obj2.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_ROP_Gr_on_Dielectric, params=tuple(obj2_params_exp_generic), ax=exp_ax, c=colourCycle2 + colourCycle2, points=30, linewidth=0.3, style="-.", singleLabel="SiO$_2$/Gr/Ga$_2$O$_3$: Generic Fit", offsets=obj2_offsets_exp_sio2, T_max=250)

# LEGEND
exp_ax.set_ylabel(r"$\Delta\rho_{xx}$ - $\rho_{xx}^{T=0}$ ($\Omega$)")
handles, labels = exp_ax.get_legend_handles_labels()
exp_ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.1,0.1), loc = "lower center")
# PARAMETERS:
# param_headers = ["$B_1$"]
param_headers = ["$B_1$","$E_0$"]

param_rows= []
param_rows.append("SiO$_2$/Gr: SiO$_2$ Fit")
# param_rows.append("SiO$_2$/Gr/Ga$_2$O$_3$: SiO$_2$ Fit")
param_rows.append("SiO$_2$/Gr/Ga$_2$O$_3$: Ga$_2$O$_3$ Fit")
param_rows.append("SiO$_2$/Gr/Ga$_2$O$_3$: Generic single mode fit")

param_list = []
# param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (params_exp[i], math.sqrt(covar_exp[i,i])) for i in range(1)]))
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (obj1_params_exp_sio2[i], math.sqrt(obj1_covar_exp_sio2[i,i])) for i in range(1)] + ["-"]))
# param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (obj2_params_exp_sio2[i], math.sqrt(obj2_covar_exp_sio2[i,i])) for i in range(1)] + ["-"]))
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (obj2_params_exp_ga2o3[i], math.sqrt(obj2_covar_exp_ga2o3[i,i])) for i in range(1)] + ["-"]))
param_list.append(tuple(["%0.03f $\pm$ %0.03f" % (obj2_params_exp_generic[i], math.sqrt(obj2_covar_exp_generic[i,i])) for i in range(2)]))

exp_ax.set_title(r"Devs4_04 Run10; Bare graphene & Ga$_2$O$_3$ covered graphene")
plt.table(cellText=param_list,
        rowLabels=param_rows,
        colLabels=param_headers,
        # bbox=(1.175,0.6,1.0,0.4))
        # bbox=(1.525,0.6,0.4,0.4))
        bbox=(1.575,0.6,0.6,0.4))

saveLoc = saveTarget + "Phonons %0.02f" % temps1[0] + "K-%0.02f" % temps1[-1] + "K %0.01fV" % vgs1[0] + "-%0.01fV" % vgs1[-1]
integer = 0
if os.path.exists(saveLoc + ".png"):
    saveTarget = saveLoc + str(integer) + ".png"
    while os.path.exists(saveTarget):
        integer += 1
        saveTarget = saveLoc + str(integer) + ".png"
else:
    saveTarget = saveLoc + str(integer) + ".png"
plt.savefig(saveTarget, bbox_inches="tight")
