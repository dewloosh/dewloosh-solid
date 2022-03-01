# -*- coding: utf-8 -*-
from numba import njit, prange
import numpy as np
from numpy import ndarray
from typing import Union

from .solid import Solid

__cache = True

ArrayOrFloat = Union[ndarray, float]


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix(gdshp: ndarray):
    nE, nP = gdshp.shape[:2]
    B = np.zeros((nE, nP, 4, 12), dtype=gdshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            # 
            B[iE, iP, 0, 0] = gdshp[iE, iP, 0, 0]
            B[iE, iP, 0, 6] = gdshp[iE, iP, 1, 0]
            #
            B[iE, iP, 1, 3] = gdshp[iE, iP, 0, 0]
            B[iE, iP, 1, 9] = gdshp[iE, iP, 1, 0]
            #
            B[iE, iP, 2, 2] = -gdshp[iE, iP, 6, 1]
            B[iE, iP, 2, 4] = -gdshp[iE, iP, 7, 1]
            B[iE, iP, 2, 8] = -gdshp[iE, iP, 8, 1]
            B[iE, iP, 2, 10] = -gdshp[iE, iP, 9, 1]
            #
            B[iE, iP, 3, 1] = gdshp[iE, iP, 2, 1]
            B[iE, iP, 3, 5] = gdshp[iE, iP, 3, 1]
            B[iE, iP, 3, 7] = gdshp[iE, iP, 4, 1]
            B[iE, iP, 3, 11] = gdshp[iE, iP, 5, 1]
    return B


@njit(nogil=True, parallel=True, cache=__cache)
def element_dcm_matrices(frames: ndarray, nNE: int=2):
    nE = frames.shape[0]
    nEVAB = nNE * 6
    res = np.zeros((nE, nEVAB, nEVAB), dtype=frames.dtype)
    for iE in prange(nE):
        for j in prange(2*nNE):
            res[iE, 3*j : 3*(j+1), 3*j : 3*(j+1)] = frames[iE]
    return res


class BernoulliBeam(Solid):
    
    NDOFN = 6
    NSTRE = 4    
           
    def model_stiffness_matrix(self, *args, **kwargs):
        return self.material_stiffness_matrix()
    
    def element_dcm_matrices(self, *args, frames=None, source=None, 
                             target=None, **kwargs):
        frames = self.frames.to_numpy() if frames is None else frames
        if source is not None:
            frames = frames @ source.dcm().T
        elif target is not None:
            frames = target.dcm() @ frames.T
        return element_dcm_matrices(frames, self.__class__.NNODE)
        
    @classmethod
    def strain_displacement_matrix(cls, pcoords: ArrayOrFloat=None, *args, 
                                   jac=None, rng=None, dshp=None, **kwargs):
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        gdshp = cls.shape_function_derivatives(pcoords, rng=rng, jac=jac, dshp=dshp)
        return strain_displacement_matrix(gdshp)
    
