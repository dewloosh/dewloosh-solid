# -*- coding: utf-8 -*-
import numpy as np
from typing import Iterable
from numpy.linalg import solve as linsolve
from dewloosh.solid.navier.loads import LoadGroup, RectLoad, PointLoad
from dewloosh.solid.navier.preproc import lhs_Navier
from dewloosh.solid.navier.mindlin.postproc import pproc_Mindlin_Navier
from numpy import swapaxes as swap


class NavierProblem:

    def __init__(self, size : tuple, shape : tuple, *args,
                 D : np.ndarray = None, S : np.ndarray = None,
                 Winkler = None, Pasternak = None,
                 Hetenyi = None, loads = None, **kwargs):
        self.size = np.array(size, dtype=float)
        self.shape = np.array(shape, dtype=int)
        self.D = D
        self.S = S
        if isinstance(loads, LoadGroup):
            self.loads = loads
        elif isinstance(loads, dict):
            self.add_loads_from_dict(loads)
        elif isinstance(loads, np.ndarray):
            raise NotImplementedError
        else:
            self.loads = LoadGroup(Navier=self)

    def add_point_load(self, name : str, pos : Iterable, value : Iterable,
                       **kwargs):
        file = PointLoad(key = name, point = pos, value = value, **kwargs)
        return self.loads.new_file(file)

    def add_rect_load(self, name : str, **kwargs):
        file = RectLoad(key = name, **kwargs)
        return self.loads.new_file(file)

    def add_loads_from_dict(self, d : dict, *args, **kwargs):
        try:
            self.loads = LoadGroup.from_dict(d)
            self.loads._Navier = self
        except Exception as e:
            print(e)
            self.loads = LoadGroup(Navier = self)
        return self.loads

    def linsolve(self, *args, **kwargs):
        Lx, Ly = size = self.size
        Nx, Ny = shape = self.shape
        LC = list(self.loads.load_cases())
        if self.S is not None:
            LHS = lhs_Navier(size, shape, D=self.D, S=self.S)
        else:
            raise NotImplementedError
        RHS = list(lc.rhs() for lc in LC)
        coeffs = list(map(lambda rhs : linsolve(LHS, rhs), RHS))
        [setattr(lc, '_coeffs', c) for lc, c in zip(LC, coeffs)]

    def postproc(self, points : np.ndarray, *args, **kwargs):
        Lx, Ly = size = self.size
        Nx, Ny = shape = self.shape
        LC = list(self.loads.load_cases())
        coeffs = np.stack([getattr(lc, '_coeffs') for lc in LC])
        if self.S is not None:
            ABDS = np.zeros((5, 5))
            ABDS[:3, :3] = self.D
            ABDS[3:, 3:] = self.S
        else:
            raise NotImplementedError
        res = pproc_Mindlin_Navier(ABDS, points, size=size, 
                                   shape=shape, solution=coeffs)
        [setattr(lc, 'res2d', swap(r2d, 0, 1)) for lc, r2d in zip(LC, res)]
        #[delattr(lc, '_coeffs') for lc, r2d in zip(LC, res)]


if __name__ == '__main__':
    pass
