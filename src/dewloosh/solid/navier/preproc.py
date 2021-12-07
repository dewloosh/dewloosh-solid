# -*- coding: utf-8 -*-
import numpy as np
from dewloosh.math.array import atleast3d
from dewloosh.core.squeeze import squeeze
from numba import njit, prange
from numpy import ndarray, pi as PI


@squeeze(True)
def lhs_Navier(size: tuple, shape: tuple, *args, D : ndarray,
               S : ndarray=None, model: str='mindlin', **kwargs):
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
    
    S : numpy.ndarray, Optional
        2d or 3d float array of shear stiffnesses. Default is None.
        
    squeeze : boolean, optional
        Removes single-dimensional entries from the shape of the 
        resulting array. Default is True.
        
    Returns
    -------
    numpy.ndarray
        3d or 4d float array of coefficients. The shape depends on
        the shape of the input parameters.
    """
    if model.lower() in ['mindlin', 'm']:
        return lhs_Mindlin(size, shape, atleast3d(D), atleast3d(S))
    elif model.lower() in ['kirchhoff', 'k']:
        return lhs_Kirchhoff(size, shape, atleast3d(D))
        

@njit(nogil=True, cache=True)
def _lhs_Mindlin(size : tuple, m: int, n:int, D: ndarray, S: ndarray):
    Lx, Ly = size
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    S44, S55 = S[0, 0], S[1, 1]
    return np.array(([[-PI**2*D22*n**2/Ly**2 - PI**2*D66*m**2/Lx**2 - S44, 
                       PI**2*D12*m*n/(Lx*Ly) + PI**2*D66*m*n/(Lx*Ly), 
                       PI*S44*n/Ly], 
                      [-PI**2*D12*m*n/(Lx*Ly) - PI**2*D66*m*n/(Lx*Ly), 
                       PI**2*D11*m**2/Lx**2 + PI**2*D66*n**2/Ly**2 + S55, 
                       PI*S55*m/Lx], 
                      [-PI*S44*n/Ly, PI*S55*m/Lx, 
                       PI**2*S44*n**2/Ly**2 + PI**2*S55*m**2/Lx**2]
                      ]))


@njit(nogil=True, parallel=True, cache=True)
def lhs_Mindlin(size : tuple, shape : tuple, D : np.ndarray, S : np.ndarray):
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
    M, N = shape
    res = np.zeros((nLHS, M * N, 3, 3), dtype=D.dtype)
    for iR in prange(nLHS):
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                iMN = (m - 1) * N + n - 1
                res[iR, iMN] = _lhs_Mindlin(size, m, n, D[iR], S[iR])
    return res


@njit(nogil=True, cache=True)
def _lhs_Kirchhoff(size : tuple, m: int, n:int, D: ndarray):
    Lx, Ly = size
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    return PI**4*D11*m**4/Lx**4 + 2*PI**4*D12*m**2*n**2/(Lx**2*Ly**2) + \
        PI**4*D22*n**4/Ly**4 + 4*PI**4*D66*m**2*n**2/(Lx**2*Ly**2)


@njit(nogil=True, parallel=True, cache=True)
def lhs_Kirchhoff(size : tuple, shape : tuple, D : np.ndarray):
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
            
    Returns
    -------
    numpy.ndarray
        2d float array of coefficients.
    """
    nLHS = D.shape[0]
    M, N = shape
    res = np.zeros((nLHS, M * N), dtype=D.dtype)
    for iR in prange(nLHS):
        for m in prange(1, M + 1):
            for n in prange(1, N + 1):
                iMN = (m - 1) * N + n - 1
                res[iR, iMN] = _lhs_Kirchhoff(size, m, n, D[iR])
    return res

