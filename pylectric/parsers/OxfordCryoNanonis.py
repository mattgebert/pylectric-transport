import numpy as np
from io import TextIOWrapper
from pylectric.geometries import hallbar

class OxfordCryoNanonisFile():
    def __init__(self, file) -> None:
        self.data, self.labels, self.params = OxfordCryoNanonisFile.filereader(file)
        return
    
    def filereader(filename):
        
        if type(filename) == str:
            filereader = open(filename)
        elif type(filename) == TextIOWrapper:
            filereader = filename
        else:
            raise IOError("Object passed was not a valid file/filename.")
        
        params = {}
        data = []
        data_labels = None
        atdata = False # Does data exist in the file?
        indata = False # To grab data titles
        
        #iterate through file to acquire data and info.
        for line in filereader:
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
    
    def to_Hallbar(self):
        return 