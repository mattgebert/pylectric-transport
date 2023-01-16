from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np

class electronic_graph_base():
    def __init__(self, *ax) -> None:
        """_summary_

        Args:
            ax (Axes): Multiple Matplotlib Axes object.
        """
        
        self.fig = None
        self.ax = []
        for obj in ax:
            if isinstance(obj, plt.Axes):
                self.ax.append(obj)
                if self.fig is None:
                    self.fig = obj.get_figure()
                else:
                    if self.fig != obj.get_figure():
                        raise AttributeError("Provided axis don't belong to the same figure").
        return

    defaultParams = {
        
    }
    
    def setDefaultParams(self, i=None):
        if not i:
            for axis in self.ax:
                axis.yaxis.set_major_locator(ticker.MaxNLocator(5))
                axis.xaxis.set_major_locator(ticker.MaxNLocator(5))
        else:
            self.ax[0].yaxis.set_major_locator(ticker.MaxNLocator(5))
            self.ax[0].xaxis.set_major_locator(ticker.MaxNLocator(5))
        self.fig.setParams(acsJournals.acsNano_1Col_Params)
        return

    def setParams(self, rcParams):
        self.fig.setParams(rcParams)
        return
    
    set setDefaultTicks(self):
        
    
    def yMobility(self):
        self.ax.set_ylabel(r"Mobility ($^2$/Vs)")

    def yMobilityCm(self):
        self.ax.set_ylabel(r"Mobility (cm$^2$/Vs)")

ax = plt.subplot(1, 2)
ax

class acsJournals:
    # Parameters shared between single and double column figures.
    acsNanoLetters_Shared_Params = {
        'figure.dpi': 300,
        "font.family":
            'arial',
            # 'helvetica',
        'font.size': 7,  # 4.5 Minimum!
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.framealpha": 0,
        "axes.titlesize": 7,
        "axes.linewidth": 0.5,
    }
    
    acsNano_2Col_Params = acsJournals.acsNano_Shared_Params.copy()
    acsNano_2Col_Params.update({
        # 4.167in (300pt) to 7in (504pt) wide maximum, 9.167 in (660pt) high.
        "figure.figsize": [7, 3.33],
    })

    acsNano_1Col_Params = acsJournals.acsNano_Shared_Params.copy()
    acsNano_1Col_Params.update({
        # 3.33 in (240pt) wide, maximum 9.167 in (660pt) high.
        "figure.figsize": [3.33, 3.33],
    })
