from pylectric.geometries import hallbar
from pylectric.signals import importing, conversion
from io import TextIOWrapper
import numpy as np
import pandas as pd
import abc

class parserFile(metaclass=abc.ABCMeta):
    
    def __init__(self,file=None) -> None:
        super().__init__()
        if file == None:
            return
        if type(file) == str:
            self.fname = file
            filebuffer = open(file)
        elif type(file) == TextIOWrapper:
            filebuffer = file
            self.fname = file.name
        else:
            raise IOError("Object passed was not a valid file/filename.")

        self.data, self.labels, self.params = self.filereader(filebuffer)
        return

    @classmethod
    @abc.abstractmethod
    def filereader(cls, filepath):
        data = None
        labels = None
        params = None
        return data, labels, params

    @abc.abstractmethod
    def copy(self, newobj):
        assert(isinstance(self.data,np.ndarray))
        newobj.data = self.data.copy()
        newobj.labels = self.labels
        newobj.fname = self.fname
        newobj.params = self.params
        return

    def to_DataFrame(self):
        return pd.DataFrame(self.data, columns=self.labels)

    def filename(self):
        if self.fname is None:
            raise AttributeError("No file specified")
        else:
            return self.fname.split("\\")[-1]

    def to_Hallbar(self, rxx_label="X-Value (V)", rxy_label="X-Value 2 (V)", field_label="Magnetic Field (T)", geom=1):
        # Check if labels exist within data labels for correct identification of data.
        for x in [rxx_label, rxy_label, field_label]:
            if x not in self.labels:
                raise AttributeError(
                    str(x) + " does not exists within data labels.")

        # Construct new object.
        # Find indexes of field, rxx, rxy in data.

        arranged_data, arranged_labels = importing.arrange_by_label(
            A=self.data, labels=self.labels, labels_ref=[field_label, rxx_label, rxy_label])

        field = arranged_data[:, 0]
        rxx = arranged_data[:, 1]
        rxy = arranged_data[:, 2]
        otherdata = arranged_data[:, 3:]

        dataseries = {}
        for i in range(len(arranged_labels)-3):
            dataseries[arranged_labels[i+3]] = otherdata[:, i]

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
                self.data[:, j] = fn(self.data[:,j], *args)

    def v2r_cc(self, current, labels=[]):
        args = (current,)
        self._applyToLabelledData(
            conversion.voltageProbes.V2R_ConstCurrent, labels, *args)

    def v2r_vc(self, v_source, r_series, labels=[]):
        args = [v_source, r_series]
        self._applyToLabelledData(
            conversion.voltageProbes.V2R_VarCurrent, labels, *args)

    def remove_gain(self, gain, labels=[]):
        args = [gain]
        self._applyToLabelledData(conversion.preamplifier.removeConstGain, labels, *args)
