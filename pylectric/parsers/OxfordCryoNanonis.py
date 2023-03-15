import numpy as np
from io import TextIOWrapper
from pylectric.parsers import parse_base

class OxfordCryoNanonisFile(parse_base.parserFile):
    def __init__(self, file=None) -> None:
        return super().__init__(file)

    @classmethod
    def filereader(cls, filebuffer):
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
    
    def copy(self):
        newobj = OxfordCryoNanonisFile()
        super().copy(newobj)
        return  newobj #no extra parameters needed.