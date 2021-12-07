# -*- coding: utf-8 -*-
from dewloosh.core.types import Hierarchy
from dewloosh.core.types.defaultdict import parsedicts_addr
from dewloosh.core.tools import allinkwargs, popfromdict
from dewloosh.core.tools import float_to_str_sig
from dewloosh.math.array import atleast2d, atleast3d
from dewloosh.core.squeeze import squeeze
import json
import numpy as np
from numpy import sin, cos, ndarray, pi as PI
from numba import njit, prange


class LoadGroup(Hierarchy):
    _typestr_ = None

    def __init__(self, *args, Navier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._Navier = Navier
    
    def Navier(self):
        return self.root()._Navier
    
    def blocks(self, *args, inclusive=False, blocktype=None, 
               deep=True, **kwargs):
        dtype = LoadGroup if blocktype is None else blocktype
        return self.containers(self, inclusive=inclusive, 
                               dtype=dtype, deep=deep)

    def load_cases(self, *args, **kwargs):
        return filter(lambda i: i.__class__._typestr_ is not None, 
                      self.blocks(*args, **kwargs))

    @staticmethod
    def string_to_type(string : str = None):
        if string == 'group':
            return LoadGroup
        elif string == 'rect':
            return RectLoad
        elif string == 'point':
            return PointLoad
        else:
            raise NotImplementedError

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
    def from_dict(d: dict=None, **kwargs) -> 'LoadGroup':
        res = LoadGroup()
        for addr, value in parsedicts_addr(d):
            if len(addr) == 0:
                continue
            if 'type' in value:
                cls = LoadGroup.string_to_type(value['type'])
                value['key'] = addr[-1]
                res[addr] = cls(**value)
        return res
        
    def __repr__(self):
        return 'LoadGroup(%s)' % (dict.__repr__(self)) 
    

class RectLoad(LoadGroup):
    _typestr_ = 'rect'

    def __init__(self, *args, value=None, **kwargs):
        self.points = RectLoad.get_coords(kwargs)
        self.value = np.array(value, dtype=float)
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
            return points_to_region(self.points)
        return None, None, None, None

    def rhs(self, *args, **kwargs):
        Navier = kwargs.get('Navier', self.Navier())
        return rect_const(Navier.size, Navier.shape, self.value, 
                          self.points, model=Navier.model)
    
    def __repr__(self):
        return 'RectLoad(%s)' % (dict.__repr__(self)) 


@squeeze(True)
def rect_const(size : tuple, shape : tuple, values : np.ndarray,
               points : np.ndarray, *args, model: str='mindlin', **kwargs):
    """
    Produces coefficients for continous loads in the order [mx, my, fz].
        mx : bending moment aroung global x
        my : bending moment aroung global y
        fz : force along global z
    """
    if model.lower() in ['mindlin', 'm']:
        rhs = rect_const_M(size, shape, atleast2d(values), atleast3d(points))
    elif model.lower() in ['kirchhoff', 'k']:
        rhs = rect_const_K(size, shape, atleast2d(values), atleast3d(points))
    return rhs


@njit(nogil=True, cache=True)
def _rect_const_M(size : tuple, m: int, n:int, xc: float, yc:float, 
                  w: float, h: float, values: ndarray):
    Lx, Ly = size
    mx, my, fz = values
    return np.array([16*mx*sin((1/2)*PI*m*w/Lx)*
                     sin((1/2)*PI*h*n/Ly)*sin(PI*n*yc/Ly)*
                     cos(PI*m*xc/Lx)/(PI**2*m*n), 
                     16*my*sin((1/2)*PI*m*w/Lx)*
                     sin(PI*m*xc/Lx)*sin((1/2)*PI*h*n/Ly)*
                     cos(PI*n*yc/Ly)/(PI**2*m*n), 
                     16*fz*sin((1/2)*PI*m*w/Lx)*sin(PI*m*xc/Lx)*
                     sin((1/2)*PI*h*n/Ly)*sin(PI*n*yc/Ly)/(PI**2*m*n)])
    

@njit(nogil=True, parallel=True, cache=True)
def rect_const_M(size : tuple, shape : tuple, values : np.ndarray,
                 points : np.ndarray):
    nR = values.shape[0]
    M, N = shape
    rhs = np.zeros((nR, M * N, 3), dtype=points.dtype)
    for iR in prange(nR):
        xmin, ymin = points[iR, 0]
        xmax, ymax = points[iR, 1]
        xc = (xmin + xmax)/2
        yc = (ymin + ymax)/2
        w = np.abs(xmax - xmin)
        h = np.abs(ymax - ymin)  
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                mn = (m - 1) * N + n - 1
                rhs[iR, mn, :] = \
                    _rect_const_M(size, m, n, xc, yc, w, h, values[iR])  
    return rhs


@njit(nogil=True, cache=True)
def _rect_const_K(size : tuple, m: int, n:int, xc: float, yc:float, 
                  w: float, h: float, values: ndarray):
    Lx, Ly = size
    mx, my, fz = values
    return np.array([
        -16*mx*sin((1/2)*PI*m*w/Lx)*sin(PI*m*xc/Lx)*sin((1/2)*PI*h*n/Ly)*\
        sin(PI*n*yc/Ly)/(PI*Lx*n),
        -16*my*sin((1/2)*PI*m*w/Lx)*sin(PI*m*xc/Lx)*sin((1/2)*PI*h*n/Ly)*\
        sin(PI*n*yc/Ly)/(PI*Ly*m),
        16*fz*sin((1/2)*PI*m*w/Lx)*sin(PI*m*xc/Lx)
    ])
    

@njit(nogil=True, parallel=True, cache=True)
def rect_const_K(size : tuple, shape : tuple, values : np.ndarray,
                 points : np.ndarray):
    nR = values.shape[0]
    M, N = shape
    rhs = np.zeros((nR, M * N, 3), dtype=points.dtype)
    for iR in prange(nR):
        xmin, ymin = points[iR, 0]
        xmax, ymax = points[iR, 1]
        xc = (xmin + xmax)/2
        yc = (ymin + ymax)/2
        w = np.abs(xmax - xmin)
        h = np.abs(ymax - ymin)  
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                mn = (m - 1) * N + n - 1
                rhs[iR, mn] = \
                    _rect_const_K(size, m, n, xc, yc, w, h, values[iR])  
    return rhs


class PointLoad(LoadGroup):
    _typestr_ = 'point'

    def __init__(self, *args, point=None, value=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.point = np.array(point, dtype=float)
        self.value = np.array(value, dtype=float)

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
        return rhs_conc(Navier.size, Navier.shape, self.value, 
                        self.point, model=Navier.model)
    
    def __repr__(self):
        return 'PointLoad(%s)' % (dict.__repr__(self)) 
    

@squeeze(True)
def rhs_conc(size : tuple, shape : tuple, values : np.ndarray,
             points : np.ndarray, *args, model: str='mindlin', **kwargs):
    if model.lower() in ['mindlin', 'm']:
        rhs = _conc_M(size, shape, atleast2d(values), atleast2d(points))
    elif model.lower() in ['kirchhoff', 'k']:
        rhs = _conc_K(size, shape, atleast2d(values), atleast2d(points))
    return rhs


@njit(nogil=True, parallel=True, cache=True)
def _conc_K(size: tuple, shape: tuple, values: ndarray, points : ndarray):
    nRHS = values.shape[0]
    Lx, Ly = size
    c = 4 / Lx / Ly
    Sx = PI / Lx
    Sy = PI / Ly
    M, N = shape
    rhs = np.zeros((nRHS, M * N, 3), dtype=points.dtype)
    for iRHS in prange(nRHS):
        x, y = points[iRHS]
        mx, my, fz = values[iRHS]
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                mn = (m - 1) * N + n - 1
                rhs[iRHS, mn, :] = c
                rhs[iRHS, mn, 0] *= -mx * sin(m * x * Sx) * sin(n * y * Sy)
                rhs[iRHS, mn, 1] *= -my * sin(m * x * Sx) * sin(n * y* Sy)
                rhs[iRHS, mn, 2] *= fz * sin(m * x * Sx) * sin(n * y * Sy)
    return rhs


@njit(nogil=True, parallel=True, cache=True)
def _conc_M(size: tuple, shape: tuple, values: ndarray, points : ndarray):
    nRHS = values.shape[0]
    Lx, Ly = size
    c = 4 / Lx / Ly
    M, N = shape
    rhs = np.zeros((nRHS, M * N, 3), dtype=points.dtype)
    PI = np.pi
    for iRHS in prange(nRHS):
        x, y = points[iRHS]
        mx, my, fz = values[iRHS]
        Sx = PI * x / Lx
        Sy = PI * y / Ly
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                mn = (m - 1) * N + n - 1
                rhs[iRHS, mn, :] = c
                rhs[iRHS, mn, 0] *= mx * cos(m * Sx) * sin(n * Sy)
                rhs[iRHS, mn, 1] *= my * sin(m * Sx) * cos(n * Sy)
                rhs[iRHS, mn, 2] *= fz * sin(m * Sx) * sin(n * Sy)
    return rhs


def points_to_region(points : ndarray):
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