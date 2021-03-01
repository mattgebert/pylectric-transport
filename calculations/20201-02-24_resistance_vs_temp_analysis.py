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
### FIlE IO Properties ###
PRELUDE = "\\01 Ga2O3 Devices"
#               Devs4_04, prior to Ga2O3 Deposition                 #
# RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\02 Removed Outliers" #Folder for raw data
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V03-V04" # Bare Graphene [Gallium Area]
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V01-V02" # Bare Graphene

#               Devs4_04, after Ga2O3 Deposition                 #
# RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\03 Data by date\\2020-12-15" #Folder for raw data
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V08-V07" # Gallium Oxide
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V01-V02" # Bare Graphene

#               Devs4_03, after Ga2O3 Deposition                 #
RAW_DATA_DIR = "\\05 Devs4_03\\01 Outliers Removed\\2020-12-20" #Folder for raw data
FILE_DESCRIPTOR = "hBN-Gr_Devs4_03_run04_V09-V08" # Gallium Oxide
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
target2 = os.getcwd() + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR + "\\" + "00 Mobility-DTM Aquisition" + "\\"
if not os.path.isdir(target2):
    os.mkdir(target2)

#Setup graphing
ud_labels = ["→","←"]
plt.rcParams.update({'font.size': 3, "figure.figsize" : [3,2], 'figure.dpi':300})
#Calculate SiO2 capacitance for mobility calculation.
SiO2_Cg = sio2.SiO2_Properties.capacitance(thickness=2.85e-7)

temps = [] #Get average temperatures
meas_objs = [] #Setup sets of discritized set of gate voltages and resistances
rvg_sets = [] #Axis are [Temperature, Up/Down, Voltage Delta, Voltage | Resistance]
vgs = np.linspace(10,30,6) #10 voltages, 5-50 includive.
# obj = file_data_objs[0]
for obj in file_data_objs:
    #Get raw measurement object
    datasets = obj.split_dataset_by_voltage_turning_point()
    t = []
    data_items = []
    rvg_items = []
    #Setup sampling voltages
    i = 0
    for i in range(len(datasets)):
        #Calculate average temperature of dataset
        t.append(np.mean(datasets[i][:,4]))
        #Create gated measurement object.
        meas_obj = Meas_GatedResistance(data=datasets[i], Cg=SiO2_Cg, L=200, W=400)
        data_items.append(meas_obj)
        #Extract sampled voltages away from dirac point = conductivity minimum.
        min_i = meas_obj.conductivity_min()
        min_v = meas_obj.raw_data[min_i,0]
        rvg = meas_obj.discrete_sample_voltages(gate_voltages=vgs, center_voltage = min_v, tollerance=0.01)
        rvg_items.append(rvg)
    temps.append(t)
    meas_objs.append(data_items)
    rvg_sets.append(rvg_items)

temps = np.array(temps)
rvg_sets
rvg_sets2 = np.array(rvg_sets)
rvg_sets2.shape
#subset data into relevant temperatures
t1,t2 = (3, 23)
rvg_sets3 = rvg_sets2[t1:t2]
temps3 = temps[t1:t2,:]

#Generate colour cycles.
colourCycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
colourCycle2 = []
hexa = colourCycle[0]
for hexa in colourCycle:
    RGB_dark = tuple(int(0.6 * int(hexa.lstrip("#")[i:i+2], 16)) for i in (0,2,4))
    hex_dark = "#" + "".join(["{:02x}".format(RGB_dark[i]) for i in (0,1,2)])
    colourCycle2.append(hex_dark)
cycles = (colourCycle2, colourCycle)
# Plot the data:
fig, (ax) = plt.subplots(1,1)
for sweep in range(rvg_sets3.shape[1]):
    # vgi = 1
    for vgi in range(rvg_sets3.shape[2]):
        vg = rvg_sets3[0,sweep,vgi,0]
        temps[:,sweep].shape
        temps
        rvg_sets2[:,sweep,vgi,1].shape
        ax.scatter(temps3[:,sweep], rvg_sets3[:,sweep,vgi,1], s=1, c = cycles[sweep][vgi], label=ud_labels[sweep] + " " + str(vg))
ax.set_yscale("log")
handles, labels = ax.get_legend_handles_labels()
ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")

colourCycle

#FIT Data
temps3.shape
vgs.shape
rvg_sets3.shape
rvt_obj_u = Meas_Temp_GatedResistance(temps = temps3[:,0], vg=vgs, resistivity=rvg_sets3[:,0,:,1])
rvt_obj_d = Meas_Temp_GatedResistance(temps = temps3[:,1], vg=vgs, resistivity=rvg_sets3[:,1,:,1])
temps_comb = np.concatenate((temps3[:,1],temps3[:,0]), axis=0)
temps_comb.shape
rvg_sets3_comb = np.concatenate((rvg_sets3[:,1,:,1], rvg_sets3[:,0,:,1]),axis=0)
rvg_sets3_comb.shape
rvt_obj_comb = Meas_Temp_GatedResistance(temps = temps_comb, vg=vgs, resistivity=rvg_sets3_comb)

paramsu, covaru = graphene.Graphene_Phonons.fit_Graphene_on_SiO2(rvt_obj_u)
paramsu

paramsd, covard = graphene.Graphene_Phonons.fit_Graphene_on_SiO2(rvt_obj_d)
paramsd

paramsc, covarc = graphene.Graphene_Phonons.fit_Graphene_on_SiO2(rvt_obj_comb)
paramsc

rvt_obj_u.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_SiO2, params=tuple(paramsu), ax=ax, c=colourCycle2, linewidth=0.3)
rvt_obj_d.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_SiO2, params=tuple(paramsd), ax=ax, c=colourCycle, linewidth=0.3)
rvt_obj_comb.global_RTVg_plot(function=graphene.Graphene_Phonons.rho_Graphene_on_SiO2, params=tuple(paramsc), ax=ax, c=colourCycle, points=30, linewidth=0.3, style="--")


paramsc, covarc = graphene.Graphene_Phonons.fit_Graphene_on_Dielectric(rvt_obj_comb)


graphene.Graphene_Phonons
