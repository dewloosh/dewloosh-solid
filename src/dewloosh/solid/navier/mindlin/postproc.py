# -*- coding: utf-8 -*-
import numpy as np
from dewloosh.solid.model.mindlin.utils import stiffness_data_Mindlin, \
    pproc_Mindlin_3D
from numpy import sin, cos, pi, ndarray as nparray
from numba import njit, prange
from dewloosh.math.array import atleast2d, atleast3d, itype_of_ftype


def pproc_Mindlin_Navier(ABDS : nparray, points : nparray, *args,
                         size : tuple=None, shape : tuple=None,
                         squeeze=True, angles=None,
                         bounds=None, shear_factors=None,
                         C_126=None, C_45=None, separate=True,
                         res2d=None, solution=None, **kwargs):
    nD = points.shape[1]
    ABDS = atleast3d(ABDS)
    dtype=ABDS.dtype
    itype = itype_of_ftype(ABDS.dtype)

    # 2d postproc
    if res2d is None and solution is not None:
        assert size is not None
        assert shape is not None
        assert points is not None
        res2d = \
            pproc_Mindlin_Navier_2D(np.array(size).astype(dtype),
                                    np.array(shape).astype(itype),
                                    atleast2d(points)[:, :2].astype(dtype),
                                    atleast3d(solution).astype(dtype),
                                    ABDS[:, 3:6, 3:6], ABDS[:, 6:, 6:], dtype)

    # 3d postproc
    assert res2d is not None
    assert points is not None
    assert points.shape[0] == res2d.shape[2]
    if nD == 2 and C_126 is not None:
        assert bounds is not None
        raise NotImplementedError
    elif nD == 3 and C_126 is not None:
        res3d = pproc_Mindlin_Navier_3D(ABDS, C_126, C_45, bounds, points,
                                        res2d, shear_factors=shear_factors,
                                        squeeze=squeeze, angles=angles, 
                                        separate=separate)
        return res3d if not squeeze else np.squeeze(res3d)
    else:
        return res2d if not squeeze else np.squeeze(res2d)


@njit(nogil=True, parallel=True, cache=True)
def pproc_Mindlin_Navier_2D(size : nparray, shape : nparray, points : nparray,
                            solution : nparray, D : nparray, S : nparray):
    """
    JIT-compiled function that calculates post-processing quantities 
    at selected ponts for multiple left- and right-hand sides.       

    Parameters
    ----------
    size : tuple 
        (Lx, Ly) : size of domain
        
    shape : tuple 
        (nX, nY) : number of harmonic terms involved in x and y
                   directions
    
    points : numpy.ndarray[nP, 2] 
        2d array of point coordinates
    
    solution : numpy.ndarray[nRHS, nX * nY, 3] 
        results of a Navier solution as a 3d array
    
    D : numpy.ndarray[nLHS, 3, 3] 
        3d array of bending stiffness terms
    
    S : numpy.ndarray[nLHS, 2, 2]
        3d array of shear stiffness terms

    Returns
    -------
    numpy.ndarray[nLHS, nRHS, nP, ...] 
        numpy array of post-processing items. The indices along 
        the last axis denote the following quantities:
        
            0 : displacement z
            1 : rotation x
            2 : rotation y
            3 : curvature x
            4 : curvature y
            5 : curvature xy
            6 : shear strain xz
            7 : shear strain yz
            8 : moment x
            9 : moment y
            10 : moment xy
            11 : shear force x
            12 : shear force y
    """
    Lx, Ly = size
    nX, nY = shape
    nLHS = D.shape[0]
    nP = points.shape[0]
    nRHS = solution.shape[0]
    res2d = np.zeros((nLHS, nRHS, nP, 11), dtype=D.dtype)
    scmn = gen_sincos_mn(size, shape, points)
    for m in prange(nX):
        cx = pi * (m + 1) / Lx
        for n in prange(nY):
            cy = pi * (n + 1) / Ly
            iMN = m * nY + n
            for iRHS in prange(nRHS):
                Amn, Bmn, Cmn = solution[iRHS, iMN, :]
                for iP in prange(nP):
                    Sm = scmn[iMN, iP, 0]
                    Cm = scmn[iMN, iP, 1]
                    Sn = scmn[iMN, iP, 2]
                    Cn = scmn[iMN, iP, 3]
                    res2d[:, iRHS, iP, 0] += Cmn * Sm * Sn
                    res2d[:, iRHS, iP, 1] += Amn * Cm * Sn
                    res2d[:, iRHS, iP, 2] += Bmn * Sm * Cn
                    res2d[:, iRHS, iP, 3] += cx * Amn * Sm * Sn
                    res2d[:, iRHS, iP, 4] += cy * Bmn * Sm * Sn
                    res2d[:, iRHS, iP, 5] -= (Amn * cx + Bmn + cy) * Cm * Cn
                    res2d[:, iRHS, iP, 6] += (Cmn * cx - Amn) * Sn * Cm
                    res2d[:, iRHS, iP, 7] += (Cmn * cy - Bmn) * Sm * Cn
                    for iLHS in prange(nLHS):
                        res2d[iLHS, iRHS, iP, 8] += Sm * Sn * \
                            (D[iLHS, 0, 0] * Amn * cx +
                             D[iLHS, 0, 1] * Bmn * cy)
                        res2d[iLHS, iRHS, iP, 9] += Sm * Sn * \
                            (D[iLHS, 0, 1] * Amn * cx +
                             D[iLHS, 1, 1] * Bmn * cy)
                        res2d[iLHS, iRHS, iP, 10] -= Cm * Cn * \
                            D[iLHS, 2, 2] * (Amn * cy + Bmn * cx)
                        res2d[iLHS, iRHS, iP, 11] += Sn * Cm * \
                            S[iLHS, 0, 0] * (Cmn * cx - Amn)
                        res2d[iLHS, iRHS, iP, 12] += Sm * Cn * \
                            S[iLHS, 1, 1] * (Cmn * cy - Bmn)
    return res2d


def pproc_Mindlin_Navier_3D(ABDS : nparray,
                            C_126 : nparray, C_45 : nparray,
                            bounds : nparray, points : nparray,
                            res2d : nparray, *args,
                            angles=None, separate=True,
                            squeeze=False, shear_factors : nparray=None,
                            **kwargs):
    nLHS, nRHS, nP, _ = res2d.shape
    strains2d = np.zeros((nLHS, nRHS, nP, 8), dtype=ABDS.dtype)
    strains2d[:, :, :, 3:8] = res2d[:, :, :, 3:8]
    ABDS = atleast3d(ABDS)
    res3d = pproc_Mindlin_3D(ABDS, C_126, C_45, bounds, points, strains2d,
                             separate = separate,
                             squeeze = squeeze, angles = angles,
                             shear_factors = shear_factors)
    return res3d


@njit(nogil=True, parallel=True, cache=True)
def gen_sincos_mn(size : nparray, shape : nparray, points : nparray):
    """
    JIT compiled convinience function to avoid repeated evaluation of 
    intermediate quantities.
    """
    Lx, Ly = size
    nX, nY = shape
    nP = points.shape[0]
    sincosmn = np.zeros((nX * nY, nP, 4), dtype=points.dtype)
    for m in prange(1, nX + 1):
        argm = m * pi / Lx
        for n in prange(1, nY + 1):
            argn = n * pi / Ly
            iMN = (m - 1) * nY + n - 1
            for iP in prange(nP):
                sincosmn[iMN, iP, 0] = sin(argm * points[iP, 0])
                sincosmn[iMN, iP, 1] = cos(argm * points[iP, 0])
                sincosmn[iMN, iP, 2] = sin(argn * points[iP, 1])
                sincosmn[iMN, iP, 3] = cos(argn * points[iP, 1])
    return sincosmn


def shell_stiffness_data(shell):
    shell.stiffness_matrix()
    layers = shell.layers()
    C_126, C_45 = [], []
    for layer in layers:
        Cm = layer.material.stiffness_matrix()
        C_126.append(Cm[0:3, 0:3])
        C_45.append(Cm[3:, 3:])
    C_126 = np.stack(C_126)
    C_45 = np.stack(C_45)
    angles = np.stack([layer.angle for layer in layers])
    bounds = np.stack([[layer.tmin, layer.tmax]
                       for layer in layers])
    ABDS, shear_corrections, shear_factors = \
        stiffness_data_Mindlin(C_126, C_45, angles, bounds)
    ABDS[-2:, -2:] *= shear_corrections
    return C_126, C_45, bounds, angles, ABDS, shear_factors


if __name__ == '__main__':
    pass
