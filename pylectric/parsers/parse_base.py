from pylectric.geometries import fourprobe, hallbar
from pylectric.signals import importing, conversion
from pylectric.signals.importing import remove_duplicate_columns, arrange_by_label
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
        if 'filename' not in self.params:
            self.params['filename'] = self.filename()
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

    def to_Hallbar(self, rxx_label="X-Value (V)", rxy_label="X-Value 2 (V)", field_label="Magnetic Field (T)", geom=1) -> hallbar.hallbar_measurement:
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
            
        hb = hallbar.hallbar_measurement(
            field=field, rxx=rxx, rxy=rxy, dataseries=dataseries, params=self.params, geom=geom)

        return hb
    
    def to_Fourprobe(self, src_type=fourprobe.fourprobe_measurement.iv_types.UNDEFINED, src_label="", rxx_label="", geom=1):
        # Ensure src_type is defined by graphable_base enumerator.
        assert src_type in fourprobe.fourprobe_measurement.iv_types
        for x in [src_label, rxx_label]:
            if x not in self.labels:
                raise AttributeError(
                    str(x) + " does not exists within data labels.")

        # Construct new object.
        # Find indexes of src, rxx in data.

        arranged_data, arranged_labels = importing.arrange_by_label(
            A=self.data, labels=self.labels, labels_ref=[src_label, rxx_label])

        src = arranged_data[:, 0]
        rxx = arranged_data[:, 1]
        otherdata = arranged_data[:, 2:]

        undef = fourprobe.fourprobe_measurement.iv_types.UNDEFINED
        if src_type is undef:
            src_type = fourprobe.fourprobe_measurement.iv_types.match_str_to_enum(src_label) #remains undefined if no match
            if src_type != undef:
                print("Type identified as " + str(src_type))

        dataseries = {}
        for i in range(len(arranged_labels)-2):
            dataseries[arranged_labels[i+2]] = otherdata[:, i]

        fp = fourprobe.fourprobe_measurement(
            src=src, rxx=rxx, src_type=src_type, dataseries=dataseries, params=self.params, geom=geom)
        return fp
        

    def _applyToLabelledData(self, fn, labels, *args):
        """Applys function to each column of data, with args.

        Args:
            fn (function): A function applied to an np.ndarray, returning the same dimension.
            labels (list of strings): List of labels to apply function to.

        Raises:
            AttributeError: Listed label doesn't exist in parser labels.
        """

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

    def _applyToDataAndLabels(self, fn, *args):
        assert isinstance(self.data, np.ndarray)
        assert isinstance(self.labels, list)
        assert callable(fn)
        self.data, self.labels = fn(self.data, self.labels, *args)
        return

    def v2r_cc(self, current, labels=[], updated_labels = False):
        """Converts a voltage to resistance, by knowing the constant current amount. Returns updated label."""
        args = [current]
        self._applyToLabelledData(
            conversion.voltageProbes.V2R_ConstCurrent, labels, *args)
        
        # Generate updated label?
        if updated_labels:
            new_labels = []
            for label in labels:
                new_labels.append(label.replace("(V)","(Ohms)"))
            # Return new labels.
            return new_labels

    def v2r_vc(self, v_source, r_series, labels=[]):
        """Converts a measured voltage to resistance, by knowing the voltage source and the series resistance to the measurement.
        Can use a single value or array of values for v_source.
        """
        args = [v_source, r_series]
        self._applyToLabelledData(
            conversion.voltageProbes.V2R_VarCurrent, labels, *args)
        
    def v2c(self, circuit_res, labels=[]):
        args = [circuit_res]
        self._applyToLabelledData(
            conversion.voltageProbes.V2R_ConstCurrent, labels, *args)
    
    def c2v(self, circuit_res, labels=[]):
        args = [circuit_res]
        self._applyToLabelledData(
            conversion.voltageProbes.V2R_ConstCurrent, labels, *args)

    def remove_gain(self, gain, labels=[]):
        """Removes a factor of gain to the listed labels."""
        args = [gain]
        self._applyToLabelledData(conversion.preamplifier.removeConstGain, labels, *args)

    def remove_duplicate_cols(self):
        self._applyToDataAndLabels(remove_duplicate_columns)
        
    def arrange_by_label(self, labels_ref):
        args = [labels_ref]
        self._applyToDataAndLabels(arrange_by_label, *args)
