import numpy as np
from io import TextIOWrapper
from pylectric.parsers import parse_base

class CSVFile(parse_base.parserFile):
    
    delimeter = "," #default but can be adjusted via init.
    
    def __init__(self, file=None, delimeter=",") -> None:
        CSVFile.delimeter = delimeter
        return super().__init__(file)

    @classmethod
    def filereader(cls, filebuffer):
        assert type(filebuffer) == TextIOWrapper
        
        data_labels = None
        units = None
        data = []
        
        params = {} #this will be empty... returned for consistency.
        
        #iterate through file to acquire data and info.
        i = 0
        for line in filebuffer:
            if i==0:
                data_labels = line.replace("\n","").split(cls.delimeter)
            elif i==1:
                units = line.replace("\n", "").split(cls.delimeter)
            else:
                data.append([float(a.strip())
                            for a in line.replace("\n", "").split(cls.delimeter)])
            i+=1
        
        params["units"]=units
            
        return np.array(data), data_labels, params
    
    def copy(self):
        newobj = CSVFile()
        super().copy(newobj)
        return  newobj #no extra parameters needed.
    
    @classmethod
    def reset_delimeter(cls):
        cls.delimeter=","
        CSVFile.delimeter=","