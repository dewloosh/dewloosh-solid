# -*- coding: utf-8 -*-
from pyoneer.core.types import Hierarchy
from pyoneer.tools.kwargtools import getasany, allinkwargs, anyinkwargs
import numpy as np
from abc import abstractmethod


class MetaSurface(Hierarchy):
    """
    Base object implementing methods that both a folder (a shell) and a
    file (a layer) can posess.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._angle = getasany(['angle', 'a'], 0, **kwargs)
        self.t = None

    @property
    def angle(self):
        if self._angle is None:
            if self.parent is not None:
                return self.parent.angle
            else:
                return self._angle
        else:
            return self._angle

    @angle.setter
    def angle(self, value):
        if self.has_file():
            for layer in self.iterfiles(inclusive=True):
                layer._angle = value
        else:
            self._angle = value


class Surface(MetaSurface):

    def __init__(self, *args, layertype=None, **kwargs):
        assert layertype is not None, "Type of layer must be specified!"
        self._layertype = layertype
        super().__init__(*args, **kwargs)

    def Layer(self, *args, **kwargs):
        return self._layertype(*args, **kwargs)

    @property
    def numLayers(self):
        return len(self.layers())

    @property
    def layertype(self):
        return self._layertype

    def layers(self):
        return [layer for layer in self.iterlayers()]

    def iterlayers(self):
        return self.iterfiles()

    def stiffness_matrix(self):
        res = np.zeros(self._layertype.__shape__)
        for layer in self.layers():
            res += layer.stiffness_matrix()
        return res

    def add_layer(self, layer: 'Layer' = None, *args,
                  freeze=False, **kwargs):
        layer = self.new_file(layer)
        if not freeze:
            self.zip()
        return layer

    def add_layers(self, *args, **kwargs):
        try:
            [self.add_layer(layer, freeze=True) for layer in args]
            self.zip()
            return True
        except Exception:
            return False

    def zip(self):
        """
        Sets thickness ranges for the layers.
        """
        layers = self.layers()
        t = sum([layer.t for layer in layers])
        tmin, tmax = -t/2, t/2
        layers[0].tmin = tmin
        nLayers = len(layers)
        for i in range(nLayers-1):
            layers[i].tmax = layers[i].tmin + layers[i].t
            layers[i+1].tmin = layers[i].tmax
        layers[-1].tmax = tmax

        for layer in layers:
            layer.zi = [layer.loc_to_z(l_) for l_ in layer.__loc__]

        self.t = t
        self.tmin = tmin
        self.tmax = tmax
        return True


class Layer(MetaSurface):
    """
    Helper class for early binding.
    """

    __loc__ = [-1., 0., 1.]
    __shape__ = (8, 8)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.material = getasany(['material', 'm'], **kwargs)
        # set thickness
        self.tmin = None
        self.tmax = None
        self.t = None
        if allinkwargs(['tmin', 'tmax'], **kwargs):
            self.tmin = kwargs.get('tmin', None)
            self.tmax = kwargs.get('tmax', None)
            self.t = self.tmax-self.tmin
        elif anyinkwargs(['t', 'thickness'], **kwargs):
            self.t = getasany(['t', 'thickness'], **kwargs)
            if 'tmin' in kwargs:
                self.tmin = kwargs['tmin']
                self.tmax = self.tmin + self.t
            elif 'tmax' in kwargs:
                self.tmax = kwargs['tmax']
                self.tmin = self.tmax - self.t
            else:
                self.tmin = (-1) * self.t / 2
                self.tmax = self.t / 2

    def loc_to_z(self, loc):
        """
        Returns height of a local point by linear interpolation.
        Local coordinate is expected between -1 and 1.
        """
        return 0.5 * ((self.tmax + self.tmin) + loc * (self.tmax - self.tmin))

    @abstractmethod
    def stiffness_matrix(self):
        pass
