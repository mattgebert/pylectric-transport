import overrides
import abc

class journalBase(metaclass=abc.ABCMeta):
    defaultParams = {
        # Scatterplots
        "scatter.marker": ".",
        "lines.linewidth": 1,
        "lines.markersize": 1
        # Lineplots
    }
        
    #All to be specified in inches!
    #unspecified
    maxwidth = None
    maxheight = None 
    minwidth = None 
    minheight = None
    #1Col
    maxwidth_1col = None 
    maxheight_1col = None
    minwidth_1col = None
    minheight_1col = None
    #2Col
    minwidth_2col = None
    minheight_2col = None
    maxwidth_2col = None
    maxheight_2col = None
    
    def _verify(l, low=None, up=None):  # lower and upper bounds
        if up:
            if l > up:
                raise AttributeError(
                    "Specified figure size ", l,
                    " is larger than Journal limit: ", up)
                return False
        if low:
            if l < low:
                raise AttributeError(
                    "Specified figure size ", l,
                    " is smaller than Journal limit: ", low)
                return False
        return True
    
    @classmethod
    def setHeightIn(cls, height_in):
        if cls._verify(height_in, cls.minheight, cls.maxheight):
            if "figure.figsize" in cls.defaultParams:
                cls.defaultParams.update({
                    "figure.figsize": [cls.defaultParams["figure.figsize"][0], height_in]
                })
            else:
                cls.defaultParams.update({
                    "figure.figsize": [cls.maxwidth if cls.maxwidth is not None else height_in, height_in]
                })
        else:
            raise AttributeError("Specified figure height ", height_in, 
                                 " is outside of the limits ", cls.minheight, " to ", cls.maxheight, ".")

    @classmethod
    def setHeightIn_1Col(cls, height_in):
        if cls._verify(height_in, cls.minheight_1col, cls.maxheight_1col):
            if "figure.figsize" in cls.defaultParams:
                cls.defaultParams.update({
                    "figure.figsize": [cls.defaultParams["figure.figsize"][0], height_in]
                })
            else:
                cls.defaultParams.update({
                    "figure.figsize": [cls.maxwidth_1col if cls.maxwidth_1col is not None else height_in, height_in]
                })
        else:
            raise AttributeError("Specified 1-column figure height ", height_in,
                                 " is outside of the limits ", cls.minheight_1col, " to ", cls.maxheight_1col, ".")

    @classmethod
    def setHeightIn_2Col(cls, height_in):
        if cls._verify(height_in, cls.minheight_2col, cls.maxheight_2col):
            if "figure.figsize" in cls.defaultParams:
                cls.defaultParams.update({
                    "figure.figsize": [cls.defaultParams["figure.figsize"][0], height_in]
                })
            else:
                cls.defaultParams.update({
                    "figure.figsize": [cls.maxwidth_2col if cls.maxwidth_2col is not None else height_in, height_in]
                })
        else:
            raise AttributeError("Specified 2-column figure height ", height_in,
                                 " is outside of the limits ", cls.minheight_2col, " to ", cls.maxheight_2col, ".")

    @classmethod
    def setWidthIn(cls, width_in):
        if cls._verify(width_in, cls.minwidth, cls.maxwidtht):
            if "figure.figsize" in cls.defaultParams:
                cls.defaultParams.update({
                    "figure.figsize": [cls.defaultParams["figure.figsize"][0], width_in]
                })
            else:
                cls.defaultParams.update({
                    "figure.figsize": [cls.maxwidth if cls.maxwidth is not None else width_in, width_in]
                })
        else:
            raise AttributeError("Specified figure height ", width_in,
                                 " is outside of the limits ", cls.minwidtht, " to ", cls.maxwidtht, ".")

    @classmethod
    def setWidthIn_1Col(cls, width_in):
        if cls._verify(width_in, cls.minwidtht_1col, cls.maxwidtht_1col):
            if "figure.figsize" in cls.defaultParams:
                cls.defaultParams.update({
                    "figure.figsize": [cls.defaultParams["figure.figsize"][0], width_in]
                })
            else:
                cls.defaultParams.update({
                    "figure.figsize": [cls.maxwidth_1col if cls.maxwidth_1col is not None else width_in, width_in]
                })
        else:
            raise AttributeError("Specified 1-column figure height ", width_in,
                                 " is outside of the limits ", cls.minwidtht_1col, " to ", cls.maxwidtht_1col, ".")

    @classmethod
    def setWidthIn_2Col(cls, width_in):
        if cls._verify(width_in, cls.minwidtht_2col, cls.maxwidtht_2col):
            if "figure.figsize" in cls.defaultParams:
                cls.defaultParams.update({
                    "figure.figsize": [cls.defaultParams["figure.figsize"][0], width_in]
                })
            else:
                cls.defaultParams.update({
                    "figure.figsize": [cls.maxwidth_2col if cls.maxwidth_2col is not None else width_in, width_in]
                })
        else:
            raise AttributeError("Specified 2-column figure height ", width_in,
                                 " is outside of the limits ", cls.minwidtht_2col, " to ", cls.maxwidtht_2col, ".")


class acsNanoLetters(journalBase):
    
    # Parameters shared between single and double column figures.
    defaultParams = dict(journalBase.defaultParams,**{
        # Graphic
        'figure.dpi': 300,
        # Fonts
        "font.family":
            'arial',
            # 'helvetica',
        'font.size': 7,  # 4.5 Minimum!
        # Axes
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.5,
        # Legends
        "legend.fontsize": 7,
        "legend.framealpha": 0,
    })

    maxwidth_1col = 3.33  # in
    maxwidth_2col = 7  # in
    maxwidth = maxwidth_2col
    minwidth_2col = 4.167  # in
    maxheight = 9.167  # in
    maxheight_1col = maxheight
    maxheight_2col = maxheight
