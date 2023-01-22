import numpy as np
import pandas as pd
from io import TextIOWrapper
from pylectric.geometries import hallbar
from pylectric.signals import importing, conversion


class OxfordCryoNanonisFile():
    def __init__(self, file) -> None:
        self.fname = None
        if type(file) == str:
            self.fname = file
            filebuffer = open(file)
        elif type(file) == TextIOWrapper:
            filebuffer = file
            self.fname = file.name
        else:
            raise IOError("Object passed was not a valid file/filename.")
        
        self.data, self.labels, self.params = OxfordCryoNanonisFile.filereader(
            filebuffer)
        return
    
    def filename(self):
        if self.fname is None:
            raise AttributeError("No file specified")
        else:        
            return self.fname.split("\\")[-1]
    
    def to_DataFrame(self):
        return pd.DataFrame(self.data, columns=self.labels)
    
    def filereader(filebuffer):
        assert type(filebuffer) == TextIOWrapper
        
        params = {}
        data = []
        data_labels = None
        atdata = False # Does data exist in the file?
        indata = False # To grab data titles
        
        #iterate through file to acquire data and info.
        for line in filebuffer:
            if atdata == False:
                if "[DATA]" in line:
                    atdata = True
                if "\t" in line:
                    parts = line.split('\t')
                    if len(parts) > 0 and type(parts) == list:
                        params[parts[0].strip()] = [a.strip() for a in parts[1:]]
            else:
                if indata == False:
                    data_labels = line.replace("\n","").split("\t")
                    indata = True
                elif len(line.strip()) > 0:
                    data.append([float(a.strip()) for a in line.replace("\n","").split("\t")])
        return np.array(data), data_labels, params
    
    def to_Hallbar(self, rxx_label="X-Value (V)", rxy_label="X-Value 2 (V)", field_label="Magnetic Field (T)", geom=1):
        # Check if labels exist within data labels for correct identification of data.
        for x in [rxx_label, rxy_label, field_label]:
            if x not in self.labels:
                raise AttributeError(str(x) + " does not exists within data labels.")        
        
        ### Construct new object.
        # Find indexes of field, rxx, rxy in data.
        
        print(self.data)
        
        arranged_data, arranged_labels = importing.arrange_by_label(
            A=self.data, labels=self.labels, labels_ref=[field_label, rxx_label, rxy_label])
        
        print(arranged_data)
        
        field = arranged_data[:,0]
        rxx = arranged_data[:,1]
        rxy = arranged_data[:,2]
        otherdata = arranged_data[:,3:]
        
        dataseries = {}
        for i in range(len(arranged_labels)-3):
            dataseries[arranged_labels[i+3]] = otherdata[:,i]
        
        if geom != 1:
            hb = hallbar.hallbar_measurement(
                field=field, rxx=rxx, rxy=rxy, dataseries=dataseries, params=self.params, geom=geom)
        else:
            hb = hallbar.hallbar_measurement(
                field=field, rxx=rxx, rxy=rxy, dataseries=dataseries, params=self.params, geom=geom)
        
        return hb
                
    def _applyToLabelledData(self, fn, labels, *args):
        
        assert isinstance(self.data, np.ndarray)
        assert isinstance(labels, list)
        assert callable(fn)
        
        # Check if labels exist within data labels for correct identification of data.
        for x in set(labels):  # use set to avoid multiple conversions.
            if x not in self.labels:
                raise AttributeError(
                    str(x) + " does not exists within data labels.")
            # Find list index
            i = np.where(np.array(self.labels) == x)[0]
            for j in i:
                self.data[:,j] = fn(self.data[:,j], *args)
                
    def v2r_cc(self, current, labels):
        args = (current,)
        self._applyToLabelledData(conversion.voltageProbes.V2R_ConstCurrent, labels, *args)
        
    def v2r_vc(self, v_source, r_series, labels = []):
        args = [v_source, r_series]
        self._applyToLabelledData(conversion.voltageProbes.V2R_VarCurrent, labels, *args)
    
    def remove_gain(self, gain, labels=[]):
        args = [gain]
        self._applyToLabelledData(conversion.preamplifier.removeConstGain, labels, *args)