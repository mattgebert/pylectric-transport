import numpy as np
from io import TextIOWrapper
from pylectric.parsers import parse_base

class SR830RecorderFile(parse_base.parserFile):
    def __init__(self, file=None) -> None:
        return super().__init__(file)

    @classmethod
    def filereader(cls, filebuffer):
        assert type(filebuffer) == TextIOWrapper
        
        params = {}
        data = []
        data_labels = None
        unit_labels = None
        atdata = False # Does data exist in the file?
        attitles = False # To grab data titles
        atunits = False #To grab units

        #iterate through file to acquire data and info.
        for line in filebuffer:
            if atdata == False:
                if "[Data]" in line:
                    atdata = True
                # Move params into dictionary
                if "\t" in line:
                    parts = line.split('\t')
                    if len(parts) > 0 and type(parts) == list:
                        params[parts[0].strip()] = [a.strip() for a in parts[1:]]
            else:
                if attitles == False:
                    data_labels = line.replace("\n","").split("\t")
                    attitles = True
                elif atunits == False:
                    unit_labels = line.replace("\n","").split("\t")
                    atunits = True
                elif len(line.strip()) > 0:
                    data.append([float(a.strip()) for a in line.replace("\n","").split("\t")])

        if not "units" in params.keys():
            params["units"] = unit_labels
        else: 
            params["unit-labels"] = unit_labels
        return np.array(data), data_labels, params
    
    def copy(self):
        newobj = SR830RecorderFile()
        super().copy(newobj)
        return  newobj #no extra parameters needed.