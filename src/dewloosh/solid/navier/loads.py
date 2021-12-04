# -*- coding: utf-8 -*-
from dewloosh.core.types import Hierarchy
from dewloosh.core.tools.kwargtools import allinkwargs, popfromdict
from dewloosh.core.tools import float_to_str_sig
import json
import numpy as np
from numpy import sin, cos
from numba import njit, prange


class LoadGroup(Hierarchy):
    _typestr_ = None

    def __init__(self, *args, Navier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._Navier = Navier

    def Navier(self):
        return self.root()._Navier

    @staticmethod
    def string_to_type(string : str = None):
        if string == 'group':
            return LoadGroup
        elif string == 'rect':
            return RectLoad
        elif string == 'point':
            return PointLoad
        else:
            return None

    def dump(self, path, *args, mode='w', indent=4, **kwargs):
        with open(path, mode) as f:
            json.dump(self.to_dict(), f, indent = indent)

    @classmethod
    def load(cls, path : str = None, **kwargs):
        if path is not None:
            with open(path, 'r') as f:
                d = json.load(f)
            return cls.from_dict(d, **kwargs)
        
    def encode(self, *args, **kwargs) -> dict:
        """
        Overwrite this in child implementations.
        """
        res = {}
        cls = type(self)
        res = {
                'type' : cls._typestr_,
                'key' : self.key,
                }
        return res

    @classmethod
    def decode(cls, d : dict=None, *args, **kwargs):
        """
        Overwrite this in child implementations.
        """
        if d is None:
            d = kwargs
            kwargs = None
        if kwargs is not None:
            d.update(kwargs)
        clskwargs = {
            'key' : d.pop('key', None),
            }
        clskwargs.update(d)
        return cls(**clskwargs)

    def to_dict(self):
        res = self.encode()
        for key, value in self.items():
            if isinstance(value, LoadGroup):
                res[key] = value.to_dict()
            else:
                res[key] = value
        return res

    @staticmethod
    def from_dict(d : dict = None, **kwargs) -> 'LoadGroup':
        d, subd = separate_typed_subdicts(d)
        cls = LoadGroup.string_to_type(d.pop('type', None))
        obj = cls.decode(d, **kwargs)
        for key, value in subd.items():
            f = LoadGroup.from_dict(value, **kwargs)
            obj.new_file(f, key = key)
        return obj


class RectLoad(LoadGroup):
    _typestr_ = 'rect'

    def __init__(self, *args, value=None, **kwargs):
        self.points = RectLoad.get_coords(kwargs)
        self.value = value
        super().__init__(*args, **kwargs)
        
    def encode(self, *args, **kwargs) -> dict:
        res = {}
        cls = type(self)
        res = {
                'type' : cls._typestr_,
                'key' : self.key,
                'region' : float_to_str_sig(self.region(), sig=6),
                'value' : float_to_str_sig(self.value, sig=6),
                }
        return res

    @classmethod
    def decode(cls, d : dict = None, *args, **kwargs):
        if d is None:
            d = kwargs
            kwargs = None
        if kwargs is not None:
            d.update(kwargs)
        points = RectLoad.get_coords(d)
        clskwargs = {
            'key' : d.pop('key', None),
            'points' : points,
            'value' : np.array(d.pop('value'), dtype=float)
            }
        clskwargs.update(d)
        return cls(**clskwargs)

    @staticmethod
    def get_coords(d : dict=None, *args, **kwargs):
        points = None
        if d is None:
            d = kwargs
        try:
            if 'points' in d:
                points = np.array(d.pop('points'))
            elif 'region' in d:
                x0, y0, w, h = np.array(d.pop('region'))
                points = np.array([[x0, y0], [x0 + w, y0 + h]])
            elif allinkwargs(['xy', 'w', 'h'], **d):
                (x0, y0), w, h = popfromdict(['xy', 'w', 'h'], d)
                points = np.array([[x0, y0], [x0 + w, y0 + h]])
            elif allinkwargs(['center', 'w', 'h'], **d):
                (xc, yc), w, h = popfromdict(['center', 'w', 'h'], d)
                points = np.array([[xc - w/2, yc - h/2],
                                   [xc + w/2, yc + h/2]])
        except Exception as e:
            print(e)
            return None
        return points

    def region(self):
        if self.points is not None:
            xmin = self.points[:, 0].min()
            ymin = self.points[:, 1].min()
            xmax = self.points[:, 0].max()
            ymax = self.points[:, 1].max()
            return xmin, ymin, xmax - xmin, ymax - ymin
        return None, None, None, None

    def rhs(self, *args, **kwargs):
        Navier = kwargs.get('Navier', self.Navier())
        return rhs_rect_const(Navier.size, Navier.shape, self.value, self.points)


class PointLoad(LoadGroup):
    _typestr_ = 'point'

    def __init__(self, *args, point=None, value=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.point = point
        self.value = value

    @classmethod
    def decode(cls, d : dict=None, *args, **kwargs):
        if d is None:
            d = kwargs
            kwargs = None
        if kwargs is not None:
            d.update(kwargs)
        clskwargs = {
            'key' : d.pop('key', None),
            'point' : np.array(d.pop('point')),
            'value' : np.array(d.pop('value')),
            }
        clskwargs.update(d)
        return cls(**clskwargs)

    def encode(self, *args, **kwargs) -> dict:
        res = {}
        cls = type(self)
        res = {
                'type' : cls._typestr_,
                'key' : self.key,
                'point' : float_to_str_sig(self.point, sig=6),
                'value' : float_to_str_sig(self.value, sig=6),
                }
        return res

    def rhs(self, *args, **kwargs):
        Navier = kwargs.get('Navier', self.Navier())
        return rhs_conc(Navier.size, Navier.shape, self.value, self.point)


def rhs_rect_const(size : tuple, shape : tuple, values : np.ndarray,
                   points : np.ndarray, *args, **kwargs):
    """
    Produces coefficients for continous loads in the order [mx, my, fz].
        mx : bending moment aroung global x
        my : bending moment aroung global y
        fz : force along global z
    """
    nD = len(values.shape)
    if nD > 1:
        return rhs_rect_const_njit(size, shape, values, points)
    else:
        return rhs_rect_const_njit(size, shape, values.reshape(1, *values.shape),
                                   points.reshape(1, *points.shape))[0]


@njit(nogil=True, parallel=True, cache=True)
def rhs_rect_const_njit(size : tuple, shape : tuple, values : np.ndarray,
                        points : np.ndarray):
    nRHS = values.shape[0]
    Lx, Ly = size
    nX, nY = shape
    rhs = np.zeros((nRHS, nX * nY, 3), dtype=points.dtype)
    PI = np.pi
    PI2 = PI**2

    for iRHS in prange(nRHS):
        xmin, ymin = points[iRHS, 0]
        xmax, ymax = points[iRHS, 1]
        xc = (xmin + xmax)/2
        yc = (ymin + ymax)/2
        lx = np.abs(xmax - xmin)
        ly = np.abs(ymax - ymin)
        Sm1 = PI * xc / Lx
        Sm2 = PI * lx / Lx / 2
        Sn1 = PI * yc / Ly
        Sn2 = PI * ly / Ly / 2
        for iN in prange(1, nX + 1):
            for iM in prange(1, nY + 1):
                iNM = (iN - 1) * nY + iM - 1
                rhs[iRHS, iNM, :] = 16 / (iM * iN * PI2)
                rhs[iRHS, iNM, 0] *= values[iRHS, 0] * \
                    sin(iM*Sm1) * cos(iN*Sn1) * sin(iM*Sm2) * sin(iN*Sn2)
                rhs[iRHS, iNM, 1] *= values[iRHS, 1] * \
                    cos(iM*Sm1) * sin(iN*Sn1) * sin(iM*Sm2) * sin(iN*Sn2)
                rhs[iRHS, iNM, 2] *= values[iRHS, 2] * \
                    sin(iM*Sm1) * sin(iN*Sn1) * sin(iM*Sm2) * sin(iN*Sn2)
    return rhs


def rhs_conc(size : tuple, shape : tuple, values : np.ndarray,
             points : np.ndarray, *args, **kwargs):
    nD = len(values.shape)
    if nD > 1:
        return rhs_conc_njit(size, shape, values, points)
    else:
        return rhs_conc_njit(size, shape, values.reshape(1, *values.shape),
                             points.reshape(1, *points.shape))[0]


@njit(nogil=True, parallel=True, cache=True)
def rhs_conc_njit(size : tuple, shape : tuple, values : np.ndarray,
                  points : np.ndarray):
    nRHS = values.shape[0]
    Lx, Ly = size
    nX, nY = shape
    rhs = np.zeros((nRHS, nX * nY, 3), dtype=points.dtype)
    PI = np.pi

    for iRHS in prange(nRHS):
        x, y = points[iRHS]
        Sx = PI * x / Lx
        Sy = PI * y / Ly
        c = 4 / Lx / Ly
        for iN in prange(1, nX + 1):
            for iM in prange(1, nY + 1):
                iNM = (iN - 1) * nY + iM - 1
                rhs[iRHS, iNM, :] = c
                rhs[iRHS, iNM, 0] *= values[iRHS, 0] * sin(iM * Sx) * cos(iN * Sy)
                rhs[iRHS, iNM, 1] *= values[iRHS, 1] * cos(iM * Sx) * sin(iN * Sy)
                rhs[iRHS, iNM, 2] *= values[iRHS, 2] * sin(iM * Sx) * sin(iN * Sy)
    return rhs


def has_typed_subdict(d : dict):
    isdict = lambda x : isinstance(x, dict)
    istyped = lambda x : 'type' in x
    children = list(filter(istyped, filter(isdict, d.values())))
    return len(children) > 0


def separate_typed_subdicts(d : dict):
    isdict = lambda item : isinstance(item[1], dict)
    istyped = lambda item : 'type' in item[1]
    subdicts = filter(istyped, filter(isdict, d.items()))
    key = lambda item : item[0]
    keys_of_subdicts = list(map(key, subdicts))
    subdicts = {k : d.pop(k) for k in keys_of_subdicts}
    return d, subdicts


def points_to_region(points : np.ndarray):
    xmin = points[:, 0].min()
    ymin = points[:, 1].min()
    xmax = points[:, 0].max()
    ymax = points[:, 1].max()
    return xmin, ymin, xmax - xmin, ymax - ymin


if __name__ == '__main__':

    loads = {
        'LG1' : {
            'LC1' : {
                'type' : 'rect',
                'points' : [[0, 0], [10, 10]],
                'value' : [0, 0, -10],
                    },
            'LC2' : {
                'type' : 'rect',
                'region' : [5., 6., 12., 10.],
                'value' : [0, 0, -2],
                    }
                },
        'LG2' : {
            'LC3' : {
                'type' : 'point',
                'point' : [10, 10],
                'value' : [0, 0, -10],
                    }
                },
        'dummy1' : 10
            }

    LC = LoadGroup.from_dict(loads)
