# -*- coding: utf-8 -*-
from numba import njit, prange
import numpy as np
from numpy import ndarray
from typing import Union

from dewloosh.math.linalg import ReferenceFrame, Vector

from .solid import Solid


__cache = True

ArrayOrFloat = Union[ndarray, float]


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix(gdshp: ndarray):
    """
    Returns the matrix expressing the relationship of generalized strains
        0 : strain along x
        1 : curvature around x
        2 : curvature around y
        3 : curvature around z
    and generalized displacements of a Bernoulli beam element.

    Parameters
    ----------
    gdshp : numpy float array of shape (N, 6, 3)  
        First, second and third derivatives for every dof(6) of every node(N).

    Returns
    -------
    numpy array of shape (4, N * 6) 
        Apprpximation coefficients for every generalized strain and every 
        shape function.
    """
    nNE = gdshp.shape[0]
    B = np.zeros((4, nNE * 6), dtype=gdshp.dtype)
    for i in prange(nNE):
        di = i * 6
        # \epsilon_x
        B[0, 0 + di] = gdshp[i, 0, 0]
        # \kappa_x
        B[1, 3 + di] = gdshp[i, 3, 0]
        # \kappa_y
        B[2, 2 + di] = -gdshp[i, 2, 1]
        B[2, 4 + di] = -gdshp[i, 4, 1]
        # \kappa_z
        B[3, 1 + di] = gdshp[i, 1, 1]
        B[3, 5 + di] = gdshp[i, 5, 1]
    return B


@njit(nogil=True, parallel=True, cache=__cache)
def strain_displacement_matrix_bulk(gdshp: ndarray):
    """
    gdshp (nE, nP, nNE, nDOF=6, 3)
    ---
    (nE, nP, 4, nNODE * 6)
    """
    nE, nP, nNE = gdshp.shape[:3]
    B = np.zeros((nE, nP, 4, nNE * 6), dtype=gdshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            B[iE, iP] = strain_displacement_matrix(gdshp[iE, iP])
    return B


"""@njit(nogil=True, cache=__cache)
def nodal_dcm(dcm: ndarray):
    nE = dcm.shape[0]
    res = np.zeros((nE, 6, 6), dtype=dcm.dtype)
    res[:, :3, :3] = dcm
    res[:, 3:, 3:] = dcm
    return res"""


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_dcm(dcm: ndarray, N=2):
    nE = dcm.shape[0]
    res = np.zeros((nE, 3 * N, 3 * N), dtype=dcm.dtype)
    for i in prange(N):
        _i = i * 3
        i_ = _i + 3
        res[:, _i:i_, _i:i_] = dcm
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def element_dcm(nodal_dcm: ndarray, nNE: int = 2):
    nE = nodal_dcm.shape[0]
    nEVAB = nNE * 6
    res = np.zeros((nE, nEVAB, nEVAB), dtype=nodal_dcm.dtype)
    for iNE in prange(nNE):
        i0, i1 = 6*iNE, 6 * (iNE+1)
        for iE in prange(nE):
            res[iE, i0: i1, i0: i1] = nodal_dcm[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def calculate_shear_forces(dofsol: ndarray, forces: ndarray,
                           D: ndarray, gdshp: ndarray):
    # dofsol (nE, nNE, nDOF=6, nRHS)
    # forces (nE, nP, 4, nRHS)
    # gdshp (nE, nP, nNE=2, nDOF=6, 3)
    nE, nP, _, nRHS = forces.shape
    nNE = dofsol.shape[1]
    res = np.zeros((nE, nP, 6, nRHS), dtype=forces.dtype)
    res[:, :, 0, :] = forces[:, :, 0, :]
    res[:, :, 3, :] = forces[:, :, 1, :]
    res[:, :, 4, :] = forces[:, :, 2, :]
    res[:, :, 5, :] = forces[:, :, 3, :]
    for i in prange(nE):
        for j in prange(nP):
            for k in prange(nRHS):
                for m in range(nNE):
                    # Vy
                    res[i, j, 1, k] += - D[i, 3, 3] * (
                        gdshp[i, j, m, 1, 2] * dofsol[i, m, 1, k] +
                        gdshp[i, j, m, 5, 2] * dofsol[i, m, 5, k])
                    # Vz
                    res[i, j, 2, k] += - D[i, 2, 2] * (
                        gdshp[i, j, m, 2, 2] * dofsol[i, m, 2, k] +
                        gdshp[i, j, m, 4, 2] * dofsol[i, m, 4, k])
    return res


class BernoulliBeam(Solid):

    __dofs__ = ('UX', 'UY', 'UZ', 'ROTX', 'ROTY', 'ROTZ')
    
    NDOFN = 6
    NSTRE = 4
    
    def model_stiffness_matrix(self, *args, **kwargs):
        return self.material_stiffness_matrix()

    def direction_cosine_matrix(self, *args, source=None, target=None, N=None, **kwargs):
        N = self.__class__.NNODE if N is None else N
        frames = self.frames.to_numpy()  # dcm_G_L
        dcm = element_dcm(nodal_dcm(frames), N)
        if source is not None:
            if isinstance(source, str):
                if source == 'global':
                    S = ReferenceFrame(dim = dcm.shape[-1]) 
                    return ReferenceFrame(dcm).dcm(source=S)
            elif isinstance(source, ReferenceFrame):
                assert source.dim == 2
                return ReferenceFrame(dcm).dcm(source=source)
            else:
                raise NotImplementedError
        elif target is not None:
            if isinstance(target, str):
                if target == 'global':
                    return ReferenceFrame(dcm)
            elif isinstance(target, ReferenceFrame):
                assert target.dim == 2
                return ReferenceFrame(dcm).dcm(target=target)
            else:
                raise NotImplementedError
        return dcm

    def strain_displacement_matrix(self, pcoords: ArrayOrFloat = None, *args,
                                   jac=None, rng=None, dshp=None, **kwargs):
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        gdshp = self.shape_function_derivatives(
            pcoords, rng=rng, jac=jac, dshp=dshp)
        return strain_displacement_matrix_bulk(gdshp)
    
    def masses(self, *args, values=None, **kwargs):
        if isinstance(values, np.ndarray):
            dens = values
        else:
            dens = self.db.densities.to_numpy()
        try:
            areas = self.areas()
        except Exception:
            areas = np.ones_like(dens)
        lengths = self.lengths()
        return areas * dens * lengths
    
    def mass(self, *args, **kwargs):
        return np.sum(self.masses(*args, **kwargs))

    


    
