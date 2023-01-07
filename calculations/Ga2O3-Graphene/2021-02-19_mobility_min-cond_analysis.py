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
from pylectric.geometries.hallbar_FET import Meas_GatedResistance
from pylectric.materials import graphene, sio2

### ----------------------------------------------------------------------------------------------------------------------------- ###
### FIlE IO Properties ###
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
####        BEGIN ANALYSIS        ####
# files=[files[0]]
#Parse Data
file_data_objs = []
for file in files:
    data_obj = RVG_file(filepath=DIRPATH + "\\" + file)
    fig = data_obj.plot_all()
    file_data_objs.append(data_obj)

###         MOBILITY CVG ANALYSIS           ###
#Create graphic directory for mobility
target2 = os.getcwd() + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR + "\\" + "00 Mobility-DTM Aquisition" + "\\"
if not os.path.isdir(target2):
    os.mkdir(target2)
#Get average temperatures
temps = []
#Setup graphing
ud_labels = ["→","←"]
plt.rcParams.update({'font.size': 3, "figure.figsize" : [3,2], 'figure.dpi':300})
#Calculate SiO2 capacitance for mobility calculation.
SiO2_Cg = sio2.SiO2_Properties.capacitance(thickness=2.85e-7)
#Calculate mobility:
meas_objs = []
mob_elec = []
mob_hole = []
# obj = file_data_objs[36]
for obj in file_data_objs:
    #Get raw measurement object
    datasets = obj.split_dataset_by_voltage_turning_point()

    #Mobility averaging
    window = 101
    mu_elec = np.zeros((len(datasets))) #up and down
    mu_hole = np.zeros((len(datasets))) #up and down

    #Calculate average temperature of dataset
    temp_up = np.mean(datasets[0][:,4])
    temp_down = np.mean(datasets[1][:,4])
    temps.append([temp_up, temp_down])

    data_items = []
    # i = 1
    for i in range(len(datasets)):
        #Smooth resistivity data to avoid digital discritisation.
        data_smooth = savgol_filter(datasets[i], window_length=math.ceil(window/2), polyorder=2, axis=0)
        meas_obj = Meas_GatedResistance(data_smooth, 200, 400, SiO2_Cg)
        data_items.append(meas_obj)
        mobility, _ = meas_obj.mobility_dtm()
        #Smooth mobility data to identify clear maxima.
        mob_smooth = savgol_filter(mobility, window_length=math.ceil(window/2), polyorder=2, axis=0)
        myplot = plt.scatter(mob_smooth[:,0], mob_smooth[:,1], s=1)

        # print((temp_up + temp_down)/2, i)
        # #Find local mobility maxima:
        i_hole, i_elec = graphene.mobility_dtm_peaks(mob_smooth[:,1], 10)

        #Get values of maxima and minima
        mu_elec[i] = mob_smooth[i_elec,1] if i_elec is not None else np.nan
        mu_hole[i] = mob_smooth[i_hole,1] if i_hole is not None else np.nan
        #Plot mobility
        max_vg = np.max(datasets[i][:,0])
        min_vg = np.min(datasets[i][:,0])
        vg_rng = np.abs(max_vg-min_vg)
        if i_hole is not None:
            myplot.axes.plot([mob_smooth[i_hole,0],min_vg], [mu_hole[i], mu_hole[i]], c='r' , linewidth=1)
            plt.text(min_vg+0.0*vg_rng, mu_hole[i]*1.02, "%0.2f" % (mu_hole[i]), c="r", fontsize="large", fontweight = "bold")
        if i_elec is not None:
            myplot.axes.plot([mob_smooth[i_elec,0],max_vg], [mu_elec[i], mu_elec[i]], c=[0.3,1,0.7] , linewidth=1)
            plt.text(max_vg-0.1*vg_rng, mu_elec[i]*1.02, "%0.2f" % (mu_elec[i]), c=[0.3,1,0.7], fontsize="large", fontweight="bold")
        plt.savefig(target2 + "Mu-DTM_Extraction_" + str(i) + ud_labels[i % 2] + "_%000.0f K" % (temps[-1][i]))

    mob_elec.append(mu_elec)
    mob_hole.append(mu_hole)
    meas_objs.append(data_items)

mob_elec = np.array(mob_elec)
mob_hole = np.array(mob_hole)
temps = np.array(temps)
#Save data:
np.savetxt(target2 + FILE_DESCRIPTOR + "Mu-DTM_Extraction_0" + ud_labels[0] + ".txt", np.c_[temps[:,0], mob_elec[:,0], mob_hole[:,0]], delimiter=",", header="Temp[Up](K),Elec Mob [Up] (cm^2/Vs),Hole Mob [Up] (cm^2/Vs)")
np.savetxt(target2 + FILE_DESCRIPTOR + "Mu-DTM_Extraction_1" + ud_labels[1] + ".txt", np.c_[temps[:,1], mob_elec[:,1], mob_hole[:,1]], delimiter=",", header="Temp[Down](K),Elec Mob [Down] (cm^2/Vs),Hole Mob [Down] (cm^2/Vs)")

fig, (ax) = plt.subplots(1,1)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Mobility (cm$^2$V$^{-1}$s${-1}$)")
ax.set_title("Electron and Hole Mobilities")
for i in range(2):
    # mob_elec
    ax.scatter(temps[:,0], mob_elec[:,i], label="Electron Mob " + ud_labels[i], s=2)
    # mob_hole
    ax.scatter(temps[:,1], mob_hole[:,i], label="Hole Mob " + ud_labels[i], s=2)
handles, labels = ax.get_legend_handles_labels()
ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(0.8,0.77), loc = "center")
plt.savefig(target + "00 Mobiltiy-DTM.png", bbox_inches="tight")

###         Minimum Conductivity Analysis           ###
min_cond = []
dirac_vg = []
# dataset = meas_objs[0]
for dataset in meas_objs:
    conds = []
    vg = []
    # data = dataset[0]
    for data in dataset:
        min_index = data.conductivity_min()
        vg.append(data.conductivity_data[min_index,0])
        conds.append(data.conductivity_data[min_index,1])
    min_cond.append(conds)
min_cond = np.array(min_cond)
#calculate resistual carrier density.

fig, (ax) = plt.subplots(1,1)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Conductivity (x$10^{-3}$ S)")
ax.set_title("Puddling")
for i in range(2):
    ax.scatter(temps[:,i], 1000*min_cond[:,i], label=str(ud_labels[i]), s=2)
handles, labels = ax.get_legend_handles_labels()
ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(0.85,0.8), loc = "center")
plt.savefig(target + "01 Conductivity-Puddling.png", bbox_inches="tight")


# plt.savefig(target + "02 Dirac-Point.png", bbox_inches="tight")
