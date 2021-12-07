# -*- coding: utf-8 -*-
import numpy as np
from typing import Iterable
from dewloosh.solid.navier.loads import LoadGroup, RectLoad, PointLoad
from dewloosh.solid.navier.preproc import lhs_Navier
from dewloosh.solid.navier.postproc import postproc
from dewloosh.solid.navier.proc import linsolve
from numpy import swapaxes as swap


class NavierProblem:

    def __init__(self, size : tuple, shape : tuple, *args,
                 D : np.ndarray = None, S : np.ndarray = None,
                 Winkler = None, Pasternak = None, loads = None,
                 Hetenyi = None, model='mindlin', mesh=None, **kwargs):
        self.size = np.array(size, dtype=float)
        self.shape = np.array(shape, dtype=int)
        self.D = D
        self.S = S
        self.model = model.lower()
        self.mesh = mesh
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

    def solve(self, *args, key='_coeffs', squeeze=True, 
              postproc=False, **kwargs):
        self._key_coeff = key
        LC = list(self.loads.load_cases())
        LHS = lhs_Navier(self.size, self.shape, D=self.D, 
                         S=self.S, model=self.model, squeeze=False)
        RHS = list(lc.rhs() for lc in LC)
        [setattr(lc, '_rhs', rhs) for lc, rhs in zip(LC, RHS)]
        if self.model in ['k', 'kirchhoff']:
            RHS = [np.sum(rhs, axis=-1) for rhs in RHS]
        coeffs = linsolve(LHS, np.stack(RHS), squeeze=False)
        #coeffs = np.squeeze(coeffs) if squeeze else coeffs
        [setattr(lc, self._key_coeff, c) for lc, c in zip(LC, coeffs)]
        if postproc:
            pass
        
    def postproc(self, points : np.ndarray, *args, cleanup=True, 
                 key='res2d', **kwargs):
        self._key_res2d = key
        size=self.size 
        shape=self.shape
        LC = list(self.loads.load_cases())
        RHS = np.stack(list(lc._rhs for lc in LC))
        coeffs = np.stack([getattr(lc, self._key_coeff) for lc in LC])
        ABDS = np.zeros((5, 5))
        ABDS[:3, :3] = self.D
        if self.S is not None:
            ABDS[3:, 3:] = self.S
        else:
            ABDS[3:, 3:] = 1e12
        res = postproc(ABDS, points, size=size, shape=shape, loads=RHS, 
                       solution=coeffs, model=self.model)
        [setattr(lc, self._key_res2d, swap(r2d, 0, 1)) for lc, r2d in zip(LC, res)]
        if cleanup:
            [delattr(lc, self._key_coeff) for lc in LC]


if __name__ == '__main__':
    pass
