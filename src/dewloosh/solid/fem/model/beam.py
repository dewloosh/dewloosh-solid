# -*- coding: utf-8 -*-
from dewloosh.solid.fem.model.solid import Solid
from dewloosh.geom.utils import cells_coords
from numba import njit, prange
import numpy as np
from numpy import ndarray
__cache = True


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
    
    def element_dcm_matrices(self, *args, frames=None, **kwargs):
        frames = self.frames.to_numpy() if frames is None else frames
        return element_dcm_matrices(frames)
    
    
