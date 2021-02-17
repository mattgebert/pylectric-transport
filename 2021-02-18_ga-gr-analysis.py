%matplotlib inline
import os, sys
PACKAGE_PARENT = 'pylectric-transport/'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser("2021-02-18_ga-gr-analysis.py"))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import warnings

from pylectric.materials.graphene import RVG_data, fitParamsRvG, fitSet, RvT_data

### ----------------------------------------------------------------------------------------------------------------------------- ###
### FIlE IO Properties ###
PRELUDE = "..\\01 Ga2O3 Devices"
# RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\01 Original Data" #Folder for raw data
RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\02 Removed Outliers" #Folder for raw data
# RAW_DATA_DIR = "\\04 Devs4_04\\PPMS Data\\03 Data by date\\2020-12-15" #Folder for raw data
# RAW_DATA_DIR = "\\05 Devs4_03\\01 Outliers Removed\\2020-12-20" #Folder for raw data
RAW_DATA_DIR = PRELUDE + RAW_DATA_DIR
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V01-V02" #Files have to include this descriptor to be processed.
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run10_V08-V07" #Files have to include this descriptor to be processed.
FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V03-V04" #Files have to include this descriptor to be processed.
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_04_run04_V01-V02" #Files have to include this descriptor to be processed.
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_03_run04_V08-V07" #Files have to include this descriptor to be processed.
# FILE_DESCRIPTOR = "hBN-Gr_Devs4_03_run04_V09-V08" #Files have to include this descriptor to be processed.
di = [0,len(FILE_DESCRIPTOR)] #descriptor indexes

#Folder for graphical output:
target = os.getcwd() + RAW_DATA_DIR + "\\" + FILE_DESCRIPTOR + "\\"
if not os.path.isdir(target):
    #Create directory.
    os.mkdir(target)

### Processing Properties ###
GEO_FACTOR = (400/200) #Geometry factor of the device - what's the conversion from resistance to resistivity? (\rho = R * (Geo_factor) = R * (W / L))

DIRPATH = os.getcwd() + RAW_DATA_DIR
files = [f for f in os.listdir(DIRPATH) if os.path.isfile(DIRPATH + "\\" + f) and f[di[0]:di[1]] == FILE_DESCRIPTOR]

rvg_analysis = []
temps = []
u_fit_params = []
d_fit_params = []
# Initial Graphing Properties:
plt.rcParams.update({'font.size': 10, "figure.figsize" : [2,1], 'figure.dpi':300})
# plt.rcParams.update({'font.size': 16, "figure.figsize" : [10,8]})
for file in files:
    #Interpret file
    analysis = RVG_data(filepath=DIRPATH + "\\" + file, geo_factor=GEO_FACTOR)
    rvg_analysis.append(analysis)
    temps.append(analysis.TEMP_MEAN)
    #Fit up sweep
    ufit = analysis.fitDataU(p0=(4e-4, 4000.0, 4000.0, 100.0, 100.0, None, 0.0, 2.85))
    # ufit, ufit_covar = analysis.fitDataU_gauss(p0=(4e-4, 4000.0, 4000.0, 100.0, 100.0, None, 0.0, 2.85,2))
    u_fit_params.append(ufit)
    #Fit down sweep
    dfit = analysis.fitDataD()
    # dfit, dfit_covar = analysis.fitDataD_gauss(p0=(4e-4, 4000.0, 4000.0, 100.0, 100.0, None, 0.0, 2.85,2))
    d_fit_params.append(dfit)
    #Plot fits on top of data
    x = np.linspace(-80,80,161)
    ax = analysis.plotCvG(s=1, c1="Navy", c2="Red")        #Show data sampling of RvG points
    ufit.plotFit(x1 = x, ax = ax, label="Fit →", c="Blue")
    dfit.plotFit(x1 = x, ax = ax, label="Fit ←", c="Orange")
    #Generate legend
    handles, labels = ax.get_legend_handles_labels()
    ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.2,0.5), loc = "center")

# ax = rvg_analysis[0].plotCvG()
# fitParamsRvG.plotParamsP(x1=np.linspace(-80,80,2*161), ax = ax, params = (5.5e-4, 7700.0, 5000.0, 80.0, 10.0, -40, 0.0, 2.8))
# fitParamsRvG.plotParamsP_gauss(x1=np.linspace(-80,80,2*161), params = (5.5e-4, 3700.0, 5000.0, 500.0, 500.0, -40, 0.0, 2.5, 4))

#Turn off matplotlib warnings
warnings.filterwarnings("ignore")

# Generate plot for all fits and data:
#UPDWARD
cols = cm.rainbow(np.linspace(0,1,len(files)))
plt.rcParams.update({'font.size': 16, "figure.figsize" : [10,8]})
x = np.linspace(-80,80,161)
#Initialize plot:
ax = rvg_analysis[0].plotCvG(c1=cols[0], down = False, s=0.5)
u_fit_params[0].plotFit(x1 = x, ax = ax, label="10K Fit →", c=cols[0])
for i in range(1,len(files)):
    rvg_analysis[i].plotCvG(ax=ax, c1=cols[i], down = False, s=0.5)
    u_fit_params[i].plotFit(x1 = x, ax = ax, label="10K Fit →", c=cols[i])
ax.set_ylim([0,0.002])
plt.legend(['{:.2f} K'.format(rvg_analysis[i].TEMP_MEAN) for i in range(len(files))], loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=11)
plt.savefig(target + "00 C_Vg-Fits_Up.png", bbox_inches="tight")
# DOWNWARD
ax = rvg_analysis[0].plotCvG(c2=cols[0], up = False, s=0.5)
d_fit_params[0].plotFit(x1 = x, ax = ax, label="10K Fit →", c=cols[0])
for i in range(1,len(files)):
    rvg_analysis[i].plotCvG(ax=ax, c2=cols[i], up = False, s=0.5)
    d_fit_params[i].plotFit(x1 = x, ax = ax, label="10K Fit →", c=cols[i])
ax.set_ylim([0,0.002])
plt.legend(['{:.2f} K'.format(rvg_analysis[i].TEMP_MEAN) for i in range(len(files))], loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=11)
plt.savefig(target + "00 C_Vg-Fits_Down.png", bbox_inches="tight")
# handles, labels = ax.get_legend_handles_labels()
# ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.2,0.5), loc = "center")

# Turn on warnings
warnings.filterwarnings("always")

###Plot parameters for each set of fits:
#Extract params
paramSetU = fitSet(temps,u_fit_params)
paramSetD = fitSet(temps,d_fit_params)

paramSetU.mob_e.__len__()
paramSetU.mob_e_err.__len__()

#MOBILITY:
plt.rcParams.update({'font.size': 16, "figure.figsize" : [10,8]})
mob_ax = paramSetU.plotMu(label="→")
paramSetD.plotMu(label="←", ax = mob_ax)
handles, labels = mob_ax.get_legend_handles_labels()
mob_ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")
# mob_ax.set_ylim([1200,5500])
plt.savefig(target + "01 FitParams-Mu.png", bbox_inches="tight")

#V_DIRAC:
plt.rcParams.update({'font.size': 16, "figure.figsize" : [10,8]})
v_dir_ax = paramSetU.plotVDirac(label="→", plot_errors=True)
paramSetD.plotVDirac(label="←", ax=v_dir_ax, plot_errors=True)
handles, labels = v_dir_ax.get_legend_handles_labels()
v_dir_ax.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")
# v_dir_ax.set_ylim([-70,-20])
plt.savefig(target + "02 FitParams-Vg.png", bbox_inches="tight")

#Sigma_Const:
plt.rcParams.update({'font.size': 16, "figure.figsize" : [10,8]})
sig_c = paramSetU.plotSigmaConst(label="→", plot_errors=False)
paramSetD.plotSigmaConst(label="←", ax=sig_c, plot_errors=False)
handles, labels = sig_c.get_legend_handles_labels()
sig_c.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")
plt.savefig(target + "03 FitParams-Sig_c.png", bbox_inches="tight")

#Pow:
plt.rcParams.update({'font.size': 16, "figure.figsize" : [10,8]})
pow = paramSetU.plotPower(label="→", plot_errors=False)
paramSetD.plotPower(label="←", ax=pow, plot_errors=False)
handles, labels = pow.get_legend_handles_labels()
pow.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")
plt.savefig(target + "04 FitParams-Pow.png", bbox_inches="tight")

#Sigma_Pudd:
plt.rcParams.update({'font.size': 16, "figure.figsize" : [10,8]})
sig_pud = paramSetU.plotSigmaPud(label="→")
paramSetD.plotSigmaPud(label="←", ax=sig_pud)
handles, labels = sig_pud.get_legend_handles_labels()
sig_pud.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")
plt.savefig(target + "05 FitParams-Sig_pud.png", bbox_inches="tight")

#Rho_S:
plt.rcParams.update({'font.size': 16, "figure.figsize" : [10,8]})
RhoS = paramSetU.plotRhoS(label="→")
paramSetD.plotRhoS(label="←", ax=RhoS)
handles, labels = RhoS.get_legend_handles_labels()
RhoS.get_figure().legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")
# RhoS.set_ylim([0,1000])
plt.savefig(target + "06 FitParams-Rho_S.png", bbox_inches="tight")

# Generate data for
plt.rcParams.update({'font.size': 16, "figure.figsize" : [4,4]})
rvt_obj = RvT_data(rvg_analysis)
np.savetxt(target + FILE_DESCRIPTOR + "_RvT_data_U" + ".txt", rvt_obj.resistancesU, delimiter=",")
rvt_obj.resistancesU.shape
rvt_obj.gate_voltages
rvt1 = rvt_obj.graphU()                                #Show data of sampled RvG points vs Temp sweep-up
rvt_obj.graphD()                                #Show data of sampled RvG points vs Temp sweep-down

rvt_obj.resistancesU.shape
rvt_obj.resistancesU

for i in range(rvt_obj.resistancesU.shape[0]):
    for j  in range(rvt_obj.resistancesU.shape[1]):
        if str(rvt_obj.resistancesU[i,j]) == "nan":
            print(i,j)

rvt_obj.temps.shape
#################################################################
#Devs4_3 Run04 V09-08 has #1,2,3
#Devs4_3 Run04 V08-07 has #4
#Devs4_4 Run10 V08-07 has #5
#Devs4_4 Run10 V01-02 has #6
#Devs4_4 Run04 V01-02 has #7
#Devs4_4 Run04 V03-04 has #8
#################################################################

# t1,t2 = [3,34] #40K to 350K  #1
# t1,t2 = [3,23] #40K to 230K #2
# t1,t2 = [3,23] #40K to 230K #3
# t1,t2 = [3,23] #40K to 230K #4
# t1,t2 = [3,23] #40K to 230K #5
# t1,t2 = [3,23] #40K to 230K #6
# t1,t2 = [2,11] #40K to 230K #7
t1,t2 = [2,11] #40K to 230K #8
temps = rvt_obj.temps[t1:t2] #Set temps from between 10K - 370K
temps[-1]
full_temps = rvt_obj.temps.copy()
# vg1, vg2, vg3, vg4 = [4,9,14,23] #1
# vg1, vg2, vg3, vg4 = [4,7,16,23] #2
# vg1, vg2, vg3, vg4 = [4,8,13,23] #3
# vg1, vg2, vg3, vg4 = [4,4,12,23] #4
# vg1, vg2, vg3, vg4 = [5,11,12,23] #5
# vg1, vg2, vg3, vg4 = [5,5,17,23] #6
vg1, vg2, vg3, vg4 = [1,1,16,23] #7
rvt_obj.gate_voltages.__len__()
vg = rvt_obj.gate_voltages[vg1:vg2] + rvt_obj.gate_voltages[vg3:vg4] #Exclude 0, +-1, +-2 VG
vg
full_vg = rvt_obj.gate_voltages.copy()
# data = rvt_obj.resistancesU
rvt_obj.resistancesU.shape
data = np.concatenate((rvt_obj.resistancesU[t1:t2,vg1:vg2],rvt_obj.resistancesU[t1:t2,vg3:vg4]), axis=1) #
full_data = rvt_obj.resistancesU.copy()
initialR0s = data[0,:]
initialR0s
# initialR0s[-1] = 200.00
data.shape

fitObj = RvT_data.global_fit_RvT(temp=temps, vg=vg, data=data, params = (35, 1, 2.3, 50), R0s_guess = initialR0s-20)
fitObj.R0
fitObj.Da
fitObj.a1
fitObj.B1

np.sqrt(np.diag(fitObj.fitcovar))


### Generate general plot points -------------------------------------
# Custom RvT Plot
plt.rcParams.update({'font.size': 16, "figure.figsize" : [6,4]})
fig, (ax1) = plt.subplots(1,1)
# cmap = cm.get_cmap("inferno")
# cmap = cm.get_cmap("viridis")
# cmap = cm.get_cmap("plasma")
cmap = cm.get_cmap("coolwarm")
full_data.shape
dims = [1j * a for a in np.shape(full_data)]
m1, m2 = np.mgrid[0:1:dims[1], 1:1:dims[0]]
c = cmap(m1)
c.shape
for i in range(len(full_vg)):
    # cmat = np.ones(len(full_data[:,i])) * full_vg[i]
    ax1.scatter(full_temps, full_data[:,i], label="{:0.3g}".format(full_vg[i]),s=7,c=c[i])
handles, labels = ax1.get_legend_handles_labels()
# fig.set_size_inches(8, 8)
fig.legend(handles, labels, title="Legend", bbox_to_anchor=(1.2,0.85))#, loc = "best")
ax1.set_title("Phonon dependent electronic transport")
ax1.set_xlabel("Temperature (K)")
# ax1.set_ylabel("Conductivity (cm$^2V^{-1}s^{-1}$)")
ax1.set_ylabel(r"Resitivity ($\Omega$)")
ax1.tick_params(direction="in")
### Generate fitted plot lines --------------------------------------
plot_temps = np.linspace(10,400,290)
c_data=np.concatenate((c[vg1:vg2],c[vg3:vg4]),axis=0)
c_data.shape
c_data=c_data[:,0,:]
c_data2 = np.zeros((c_data.shape[0], len(plot_temps), c_data.shape[1]))
for i in range(c_data.shape[0]): #For each gate voltage
    for j in range(len(plot_temps)): #for each temp point
        c_data2[i,j,:] = c_data[i,:]
fitObj.plotParams(ax=ax1, temps=plot_temps, c=c_data2)
plt.savefig(target + "07 RvT_Fit_Linear.png", bbox_inches="tight")
ax1.set_yscale("log")
plt.savefig(target + "07 RvT_Fit_Log.png", bbox_inches="tight")

# # Plot Fitting: Da, A, B
# params = [35,1,2.3] + list(initialR0s-20)
# test_params=RvT_data.fitParamsRvT1(p=tuple(params),vg=vg,temps=temps)
# test_params.R0
# test_params.plotParams(ax=ax1)
