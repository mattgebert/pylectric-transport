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
from scipy.signal import savgol_filter

# from pylectric.materials.graphene import RVG_data, fitParamsRvG, fitSet, RvT_data
from pylectric.parsers.RVG import RVG_file
from pylectric.geometries.FET.hallbar import Meas_GatedResistance
from pylectric.materials.sio2 import SiO2_Properties

### ----------------------------------------------------------------------------------------------------------------------------- ###
### FIlE IO Properties ###
PRELUDE = "\\01 Ga2O3 Devices"
# RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\01 Original Data" #Folder for raw data
RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\02 Removed Outliers" #Folder for raw data
# RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\03 Data by date\\2020-12-15" #Folder for raw data
# RAW_DATA_DIR = "\\05 Devs4_03\\01 Outliers Removed\\2020-12-20" #Folder for raw data
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V01-V02" #Files have to include this descriptor to be processed.
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V08-V07" #Files have to include this descriptor to be processed.
FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V03-V04" #Files have to include this descriptor to be processed.
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V01-V02" #Files have to include this descriptor to be processed.
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_03_run04_V08-V07" #Files have to include this descriptor to be processed.
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_03_run04_V09-V08" #Files have to include this descriptor to be processed.
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
files=[files[-1]]
#Parse Data
file_data_objs = []
for file in files:
    data_obj = RVG_file(filepath=DIRPATH + "\\" + file)
    fig = data_obj.plot_all()
    file_data_objs.append(data_obj)

plt.rcParams.update({'font.size': 3, "figure.figsize" : [3,2], 'figure.dpi':300})
#Calculate mobility:
meas_objs = []
mobilities = []
SiO2_Cg = SiO2_Properties.capacitance(thickness=2.85e-7)
obj = file_data_objs[0]
for obj in file_data_objs:
    meas_obj = Meas_GatedResistance(obj.RAW_DATA, 200, 400, SiO2_Cg)

    #Calculate mobility using default method
    (mobility, fig) = meas_obj.mobility_dtm()




    spacing=10
    dd = np.diff(meas_obj.conductivity_data[::spacing,:].copy(), axis=0)
    grad = np.abs(dd[:,1] / dd[:,0])
    mu_dtm = 1.0 / meas_obj.Cg * grad * 1.0E4
    vgs = meas_obj.conductivity_data[:-1:spacing,0] + dd[:,0]

    (dd[:,1]==0)[20:30]
    dd[20:30, 1]
    meas_obj.conductivity_data[20:30,1]

    plt.scatter(vgs, mu_dtm, s=1)


    mobility.shape
    #Smooth mobility and also plot:
    mobility_smoothed = savgol_filter(mobility, 101, 2, axis=0)
    fig.axes[0].plot(mobility_smoothed[:,0], mobility_smoothed[:,1], 'r', label="Smoothed")
    fig
    mobilities.append(mobility)
    meas_objs.append(meas_obj)

meas_objs[0].conductivity_data.shape
dd[:,0]

dd = np.diff(meas_objs[0].conductivity_data[:,:])
grad = dd[:,1] / dd[:,0]
mu_dtm = 1.0 / SiO2_Cg * grad * 1.0E4
grad
mu_dtm
np.mean(mu_dtm)


a = np.array([[1,2,3],[4,5,6]])
a[:-1,:]
a[:,:]
