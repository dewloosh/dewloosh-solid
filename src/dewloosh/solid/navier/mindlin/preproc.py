# -*- coding: utf-8 -*-
import numpy as np
from dewloosh.math.linalg import linspace1d
from dewloosh.math.array import atleast1d, atleast2d, atleast3d, \
    atleast4d, ascont, clip1d
from numpy.linalg import inv
from numba import njit, prange
from collections import Iterable
from dewloosh.math.array import atleast3d
__cache = True


def lhs_Navier_Mindlin(size : tuple, shape : tuple, *args, D : np.ndarray,
                       S : np.ndarray, squeeze=True, **kwargs):
    """
    Returns coefficient matrix for a Navier solution, for a single or 
    multiple left-hand sides.
    
    Parameters
    ----------
    size : tuple
        Tuple of floats, containing the sizes of the rectagle.
    
    shape : tuple
        Tuple of integers, containing the number of harmonic terms
        included in both directions.
        
    D : numpy.ndarray
        2d or 3d float array of bending stiffnesses.
    
    S : numpy.ndarray
        2d or 3d float array of shear stiffnesses.
        
    squeeze : boolean, optional
        Removes single-dimensional entries from the shape of the 
        resulting array. Default is True.
        
    Returns
    -------
    numpy.ndarray
        3d or 4d float array of coefficients. The shape depends on
        the shape of the input parameters.
    """
    res = lhs_Navier_Mindlin_njit(size, shape, atleast3d(D), atleast3d(S))
    return res if not squeeze else np.squeeze(res)


@njit(nogil=True, parallel=True, cache=True)
def lhs_Navier_Mindlin_njit(size : tuple, shape : tuple, D : np.ndarray,
                            S : np.ndarray):
    """
    JIT compiled function, that returns coefficient matrices for a Navier 
    solution for multiple left-hand sides.
    
    Parameters
    ----------
    size : tuple
        Tuple of floats, containing the sizes of the rectagle.
    
    shape : tuple
        Tuple of integers, containing the number of harmonic terms
        included in both directions.
        
    D : numpy.ndarray
        3d float array of bending stiffnesses.
    
    S : numpy.ndarray
        3d float array of shear stiffnesses.
        
    Returns
    -------
    numpy.ndarray
        4d float array of coefficients.
    """
    nLHS = D.shape[0]
    Lx, Ly = size
    nX, nY = shape
    A = np.zeros((nLHS, nX * nY, 3, 3), dtype=D.dtype)
    PI = np.pi

    for iR in prange(nLHS):
        for m in prange(1, nX + 1):
            for n in prange(1, nY + 1):
                # auxiliary variables
                iMN = (m - 1) * nY + n - 1
                cx = m * PI / Lx
                cy = n * PI / Ly
                cxy = cx * cy
                # sum_mx
                A[iR, iMN, 0, 0] = - cxy * (D[iR, 1, 0] + D[iR, 2, 2])
                A[iR, iMN, 0, 1] = - cy**2 * D[iR, 1, 1] - \
                    cx**2 * D[iR, 2, 2] - S[iR, 1, 1]
                A[iR, iMN, 0, 2] = S[iR, 1, 1] * cy
                # sum_my
                A[iR, iMN, 1, 0] = cx**2 * D[iR, 0, 0] + \
                    cy**2 * D[iR, 2, 2] + S[iR, 0, 0]
                A[iR, iMN, 1, 1] = cxy * (D[iR, 0, 1] + D[iR, 2, 2])
                A[iR, iMN, 1, 2] = -S[iR, 0, 0] * cx
                # sum_fz
                A[iR, iMN, 2, 0] = S[iR, 0, 0] * cx
                A[iR, iMN, 2, 1] = S[iR, 1, 1] * cy
                A[iR, iMN, 2, 2] = -S[iR, 0, 0] * cx**2 - S[iR, 1, 1] * cy**2
    return A