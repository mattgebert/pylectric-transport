%matplotlib inline
import os, sys

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
PACKAGE_PARENT = 'pylectric-transport'
sys.path.append(os.path.normpath(os.path.join(os.getcwd(), PACKAGE_PARENT)))
from pylectric.parsers.RVG import RVG_file
from pylectric.geometries.FET.hallbar import Meas_GatedResistance, Meas_Temp_GatedResistance
from pylectric.materials import graphene, sio2

#################################################################################
#################################################################################
###################              DATA IMPORT           ##########################
#################################################################################
#################################################################################
PRELUDE = "\\01 Ga2O3 Devices"
TEMPSTR = "_300K"
#               Devs4_04, prior to Ga2O3 Deposition                 #
RAW_DATA_DIR1 = "\\04 Devs4_04\\PPMS Data\\02 Removed Outliers" #Folder for raw data
FILE_B41_Ga2O3 = "hBN-Gr_Devs4_04_run04_V03-V04" + TEMPSTR + ".txt"# Bare Graphene [Gallium Area]
FILE_B41_Bare = "hBN-Gr_Devs4_04_run04_V01-V02" + TEMPSTR + ".txt" # Bare Graphene

#               Devs4_04, after Ga2O3 Deposition                 #
RAW_DATA_DIR2 = "\\04 Devs4_04\\PPMS Data\\03 Data by date\\2020-12-15" #Folder for raw data
FILE_AF1_Ga2O3 = "hBN-Gr_Devs4_04_run10_V08-V07" + TEMPSTR + ".txt" # Gallium Oxide
FILE_AF1_Bare = "hBN-Gr_Devs4_04_run10_V01-V02" + TEMPSTR + ".txt" # Bare Graphene

# #               Devs4_03, before Ga2O3 Deposition                 #
# RAW_DATA_DIR = "\\05 Devs4_03\00 Original Data\2020-09-30" #Folder for raw data
# FILE_B42_Ga2O3 = "hBN-Gr_Devs4_03_run04_V09-V08" # Gallium Oxide
# FILE_B42_Bare = "hBN-Gr_Devs4_03_run04_V08-V07"   # Bare Graphene

#               Devs4_03, after Ga2O3 Deposition                 #
RAW_DATA_DIR3 = "\\05 Devs4_03\\01 Outliers Removed\\2020-12-20" #Folder for raw data
FILE_AF2_Ga2O3 = "hBN-Gr_Devs4_03_run04_V09-V08" + TEMPSTR + ".txt"# Gallium Oxide
FILE_AF2_Bare = "hBN-Gr_Devs4_03_run04_V08-V07" + TEMPSTR + ".txt"  # Bare Graphene



#Add all filenames to a list
file_names = []
file_names.append(FILE_B41_Bare)
file_names.append(FILE_B41_Ga2O3)
file_names.append(FILE_AF1_Bare)
file_names.append(FILE_AF1_Ga2O3)
file_names.append(FILE_AF2_Bare)
file_names.append(FILE_AF2_Ga2O3)

titles = [
    "00 Devs4_04 V03-V04 (Ga2O3 Area) Before Ga2O3",
    "01 Devs4_04 V01-V02 (Bare Area) Before Ga2O3",
    "02 Devs4_04 V08-V07 (Ga2O3 Area) After Ga2O3",
    "03 Devs4_04 V01-V02 (Bare Area) After Ga2O3",
    "04 Devs3_04 V09-V08 (Ga2O3 Area) After Ga2O3"
    "05 Devs3_04 V08-V07 (Bare Area) After Ga2O3"
]


file_paths = []
file_paths.append(os.getcwd() + PRELUDE + RAW_DATA_DIR1 + "\\" + FILE_B41_Bare)
file_paths.append(os.getcwd() + PRELUDE + RAW_DATA_DIR1 + "\\" + FILE_B41_Ga2O3)
file_paths.append(os.getcwd() + PRELUDE + RAW_DATA_DIR2 + "\\" + FILE_AF1_Bare)
file_paths.append(os.getcwd() + PRELUDE + RAW_DATA_DIR2 + "\\" + FILE_AF1_Ga2O3)
file_paths.append(os.getcwd() + PRELUDE + RAW_DATA_DIR3 + "\\" + FILE_AF2_Bare)
file_paths.append(os.getcwd() + PRELUDE + RAW_DATA_DIR3 + "\\" + FILE_AF2_Ga2O3)

### Processing Properties ###
L = 256.75
W = 408.9
GEO_FACTOR = (W/L) #Geometry factor of the device - what's the conversion from resistance to resistivity? (\rho = R * (Geo_factor) = R * (W / L))

####        BEGIN ANALYSIS        ###
#Parse Data
file_data_objs = []
for file in file_paths:
    data_obj = RVG_file(filepath=file)
    fig = data_obj.plot_all()
    file_data_objs.append(data_obj)

###         MOBILITY CVG ANALYSIS           ###
#Create graphic directory for mobility
target = os.getcwd() + "\\02 Ga2O3 Paper"
if not os.path.isdir(target):
    os.mkdir(target)
#Get average temperatures
temps = []
#Setup graphing
ud_labels = ["→","←"]
plt.rcParams.update({'font.size': 3, "figure.figsize" : [3,2], 'figure.dpi':300})
#Calculate SiO2 capacitance for mobility calculation.
SiO2_Properties = sio2.SiO2(t_ox=2.85e-7)
SiO2_Cg = SiO2_Properties.capacitance()
#Calculate mobility:
meas_objs = []
mob_elec = []
mob_hole = []
# obj = file_data_objs[36]
for j in range(len(file_data_objs)):
    obj = file_data_objs[j]
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
        meas_obj = Meas_GatedResistance(data_smooth, L, W, SiO2_Cg)
        data_items.append(meas_obj)
        mobility, _ = meas_obj.mobility_dtm_2D()
        #Smooth mobility data to identify clear maxima.
        mob_smooth = savgol_filter(mobility, window_length=math.ceil(window/2), polyorder=2, axis=0)
        myplot = plt.scatter(mob_smooth[:,0], mob_smooth[:,1], s=1)

        # print((temp_up + temp_down)/2, i)
        # #Find local mobility maxima:
        i_hole, i_elec = graphene.Graphene_Gated.mobility_dtm_peaks(mob_smooth[:,1], 10)

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
        plt.savefig(target + "\\" + file_names[j] +  "Mu-DTM_Extraction_" + str(i) + ud_labels[i % 2] + "_%000.0f K.png" % (temps[-1][i]))

    mob_elec.append(mu_elec)
    mob_hole.append(mu_hole)
    meas_objs.append(data_items)

mob_elec = np.array(mob_elec)
mob_hole = np.array(mob_hole)
temps = np.array(temps)


#Generate Mobility Figures:
# plt.rcParams.update({'font.size': 3, "figure.figsize" : [4,1], 'figure.dpi':300})
# kwargs = {"markersize":0.1, "linewidth":0.5}
# fig, (axes) = plt.subplots(1,2)
