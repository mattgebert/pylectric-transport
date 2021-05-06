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
# RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\03 Data by date\\2020-12-15" #Folder for raw data
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V08-V07" # Gallium Oxide
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V01-V02" # Bare Graphene

#               Devs4_03, after Ga2O3 Deposition                 #
# RAW_DATA_DIR = "\\05 Devs4_03\\01 Outliers Removed\\2020-12-20" #Folder for raw data
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_03_run04_V09-V08" # Gallium Oxide
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_03_run04_V08-V07"   # Bare Graphene

# Can't check this one, because of the gate going beyond 100V as temp warms up. Starts at 60V.
# RAW_DATA_DIR = "\\02 CVD_hBN-Gr_Dev02 (W2D01)\\PPMS Data\\01 Original Copy"
# FILE_DESCRIPTOR = "CVDGraphene_W2D01_Run04_V10&9"



### Processing Properties ###
L = 256.75
W = 408.9
# GEO_FACTOR = (W/L) #Geometry factor of the device - what's the conversion from resistance to resistivity? (\rho = R * (Geo_factor) = R * (W / L))

RAW_DATA_DIR = "\\03 SSS1 Dev06\\PPMS Data\\01 Original Copy"
FILE_DESCRIPTOR = "SSS1_Dev06_Run03_V3&V4"
### Processing Properties ###
L = 256.75 * 10 #10 added because of incorrect resistance in datafile.
W = 408.9

GEO_FACTOR = (W/L) #Geometry factor of the device - what's the conversion from resistance to resistivity? (\rho = R * (Geo_factor) = R * (W / L))

di = [0,len(FILE_DESCRIPTOR)] #descriptor indexes

#Folder for graphical output:
RAW_DATA_DIR = PRELUDE + RAW_DATA_DIR
target = os.getcwd() + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR + "\\"
if not os.path.isdir(target):
    #Create directory.

    os.mkdir(target)

DIRPATH = os.getcwd() + RAW_DATA_DIR
files = [f for f in os.listdir(DIRPATH) if os.path.isfile(DIRPATH + "\\" + f) and f[di[0]:di[1]] == FILE_DESCRIPTOR]
files
### REMOVING EXTREMA FILES DUE TO UNFITTABLE
# files = files[0:34] #TODO REMOVE specific for Devs4_03_Run04_V09-V08 and Devs4_03_Run04_V08-V07
files.__len__()
files = files[0:34] #TODO REMOVE specific for W2D01


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
target2 = os.getcwd() + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR + "\\" + "08 actually Corrected Geom" + "\\"
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

# Isolate up and down sweep data
dataObjsUp = []
dataObjsDown = []
for i in range(len(file_data_objs)):
    (dataUp, dataDown) = file_data_objs[i].split_dataset_by_voltage_turning_point()
    dataObjsUp.append(dataUp.copy())
    dataObjsDown.append(dataDown.copy())
dataObjsUp.__len__()

dataObjsUp[0][0,0:4]
dataObjsDown[0][0,0:4]
# Change to conductivity and scale:
for i in range(len(dataObjsUp)):
    dataObjsUp[i][:,1] = 1.0/(dataObjsUp[i][:,1] * W/L)
for i in range(len(dataObjsDown)):
    dataObjsDown[i][:,1] = 1.0/(dataObjsDown[i][:,1] * W/L)
dataObjsUp[0][0,0:4]
dataObjsDown[0][0,0:4]

# Minimum conductivity fitting.
def minCondFit(data):
    ###Initial Values
    #Get initial values for minimum conductivity and gate voltage.
    min_sigma_i = np.where(data[:,1] == np.min(data[:,1]))[0][0] #Find min index for conductivity
    #Fit Variables:
    vg_dirac = data[min_sigma_i,0]
    sigma_pud_e = data[min_sigma_i,1]

    ## Find data window within range:
    # Get values greater than
    gt_factor_min_indexes = np.where(data[:,1] > 2*np.min(data[:,1]))[0]
    # take first below min_index
    search = np.where(gt_factor_min_indexes <= min_sigma_i)[0]
    if len(search) > 0:
        i1 = search[-1] #Last index will be closest to beginning of the range.
        sigma_i1 = gt_factor_min_indexes[i1]
        try:
            sigma_i2 = gt_factor_min_indexes[i1 + 1] #next index will be closest after
            # Fit to data range:
            subset = data[sigma_i1:sigma_i2,:]
        except IndexError:
            subset = data[sigma_i1:]
    else:
        # No point less than the minimum. But maybe something after!
        sigma_i2 = gt_factor_min_indexes[0] #next index will be closest after
        subset = data[0:sigma_i2,:]
    quad = np.polynomial.polynomial.Polynomial.fit(x=subset[:,0],y=subset[:,1], deg=2).convert(domain=[-1,1])

    # Calculate
    c,b,a = quad.coef.tolist()
    x_tp = - b / (2*a) #turning point, in otherwords dirac point voltage.
    return x_tp, quad, subset[:,0]

minCondU = []
for obj in dataObjsUp:
    vg, poly, data = minCondFit(obj)
    minCondU.append(vg)
minCondD = []
for obj in dataObjsDown:
    vg, poly, data = minCondFit(obj)
    minCondD.append(vg)
minCondU

# Get average temperatures
tempU = [np.mean(obj[:,4]) for obj in dataObjsUp]
tempD = [np.mean(obj[:,4]) for obj in dataObjsDown]


plt.rcParams.update({'font.size': 3, "figure.figsize" : [1,1], 'figure.dpi':300})
plt.plot(tempU, minCondU)

def interpolateResistances(data,center_voltage=0,gate_voltages=None):
    cv = center_voltage
    #Default sampling if no list provided.
    if gate_voltages is None:
        gate_voltages = [-50,-40,-30,-20,-15,-10,-5,-3,-1,0,1,3,5,10,15,20,30,40,50]

    #Resistsances and gate voltages
    rvg_sampled = []

    #Find
    for gv in gate_voltages:
        vraw = data[:,0] - center_voltage - gv
        vdelta = np.abs(vraw)
        v_i = np.where(vdelta == np.min(vdelta))[0][0]
        if vdelta[v_i] == 0:
            #Offset center voltage, and take reciporical of conductivity for resistivity.
            rvg_sampled.append([data[v_i,0]-center_voltage, 1.0/data[v_i,1]])
        else:
            if not (v_i < 1 or v_i > len(data)-2): #check endpoint condition
                # Interpolate data if not endpoints:
                B1 = data[v_i,:]
                if vdelta[v_i + 1] < vdelta[v_i - 1]: #smaller is better for interpolation, closer to dirac point.
                    B2 = data[v_i + 1,:]
                else:
                    B2 = data[v_i - 1,:]
                #re-arranged gv = (alpha * (B1-cv) + (1-alpha) * (B2-cv)), finding linear interpolation.
                alpha = (gv - (B2[0] - cv)) / (B1[0] - B2[0])
                # Saftey check for result consistency.
                if alpha < 0 or alpha > 1:
                    raise(ValueError("Calculation of linear interpolation factor (alpha = " + str(alpha) + ") is outside (0,1)."))
                #Interolate
                inter_v = (alpha * (B1[0] - cv)) + ((1-alpha) * (B2[0] - cv))
                inter_resistivity = 1.0/(alpha * (B1[1]) + (1-alpha) * (B2[1]))
                #append
                rvg_sampled.append([inter_v, inter_resistivity]) #Add a NAN value to the array because value out of range.
            else:
                rvg_sampled.append([gv, np.nan]) #Add a NAN value to the array because value out of range.

    rvg_sampled = np.array(rvg_sampled)
    return rvg_sampled

gate_voltages=[50]

rvgInterpolatedUp = []
for i in range(len(dataObjsUp)):
    rvgInterpolatedUp.append(interpolateResistances(dataObjsUp[i],gate_voltages=gate_voltages))
rvgInterpolatedDown = []
for i in range(len(dataObjsDown)):
    rvgInterpolatedDown.append(interpolateResistances(dataObjsDown[i],gate_voltages=gate_voltages))

rvgInterpolatedUp[0]

kwargs = {"markersize":1}

fig, (ax) = plt.subplots(1,1)
ax.plot(tempU, np.array(rvgInterpolatedUp)[:,0,1], ".", label="Up", **kwargs)
ax.plot(tempD, np.array(rvgInterpolatedDown)[:,0,1], ".", label="Down", **kwargs)
plt.legend(loc="upper left")
# ax.set_ylim([500,800])

rvgInterpolationDelta = [obj1 - obj2 for obj1,obj2 in zip(rvgInterpolatedUp, rvgInterpolatedDown)]
plt.plot(tempU, np.abs(rvgInterpolationDelta)[:,0,1], ".", label="Delta", **kwargs)

# i=19
# tempU[i]
# rvgInterpolationDelta[i][0,1]
#
# i = 19
# tempU[i]
# np.array(rvgInterpolatedUp)[i,0,1]
savedata = np.c_[tempU,np.array(rvgInterpolatedUp)[:,0,1]]

np.savetxt(os.getcwd() + "\\Hello.txt", savedata, delimiter=",", header="Temperature (K),Gate Voltage (V),Resisitivity (Ohms)")

tempU
