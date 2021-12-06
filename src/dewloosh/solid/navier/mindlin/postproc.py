# -*- coding: utf-8 -*-
import numpy as np
from dewloosh.solid.model.mindlin.utils import stiffness_data_Mindlin, \
    pproc_Mindlin_3D
from numpy import sin, cos, ndarray as nparray, pi as PI
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
                                    ABDS[:, :3, :3], ABDS[:, 3:, 3:])

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


@njit(nogil=True, cache=True)
def _pproc_Mindlin_2D(size, m: int, n:int, 
                      xp: float, yp:float, solution : nparray, 
                      D: nparray, S: nparray):
    Lx, Ly = size
    Amn, Bmn, Cmn = solution
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    S44, S55 = S[0, 0], S[1, 1]
    return np.array(
        [Cmn*sin(PI*m*xp/Lx)*sin(PI*n*yp/Ly), 
         Amn*sin(PI*m*xp/Lx)*cos(PI*n*yp/Ly), 
         Bmn*sin(PI*n*yp/Ly)*cos(PI*m*xp/Lx), 
         -PI*Bmn*m*sin(PI*m*xp/Lx)*sin(PI*n*yp/Ly)/Lx, 
         PI*Amn*n*sin(PI*m*xp/Lx)*sin(PI*n*yp/Ly)/Ly, 
         -PI*Amn*m*cos(PI*m*xp/Lx)*cos(PI*n*yp/Ly)/Lx + 
         PI*Bmn*n*cos(PI*m*xp/Lx)*cos(PI*n*yp/Ly)/Ly, 
         Bmn*sin(PI*n*yp/Ly)*cos(PI*m*xp/Lx) + 
         PI*Cmn*m*sin(PI*n*yp/Ly)*cos(PI*m*xp/Lx)/Lx, 
         -Amn*sin(PI*m*xp/Lx)*cos(PI*n*yp/Ly) + 
         PI*Cmn*n*sin(PI*m*xp/Lx)*cos(PI*n*yp/Ly)/Ly, 
         PI*Amn*D12*n*sin(PI*m*xp/Lx)*sin(PI*n*yp/Ly)/Ly - 
         PI*Bmn*D11*m*sin(PI*m*xp/Lx)*sin(PI*n*yp/Ly)/Lx, 
         PI*Amn*D22*n*sin(PI*m*xp/Lx)*sin(PI*n*yp/Ly)/Ly - 
         PI*Bmn*D12*m*sin(PI*m*xp/Lx)*sin(PI*n*yp/Ly)/Lx, 
         -PI*Amn*D66*m*cos(PI*m*xp/Lx)*cos(PI*n*yp/Ly)/Lx + 
         PI*Bmn*D66*n*cos(PI*m*xp/Lx)*cos(PI*n*yp/Ly)/Ly, 
         Bmn*S55*sin(PI*n*yp/Ly)*cos(PI*m*xp/Lx) + 
         PI*Cmn*S55*m*sin(PI*n*yp/Ly)*cos(PI*m*xp/Lx)/Lx, 
         -Amn*S44*sin(PI*m*xp/Lx)*cos(PI*n*yp/Ly) + 
         PI*Cmn*S44*n*sin(PI*m*xp/Lx)*cos(PI*n*yp/Ly)/Ly]
    )


@njit(nogil=True, parallel=False, fastmath=True, cache=True)
def pproc_Mindlin_Navier_2D(size, shape : nparray, points : nparray,
                            solution : nparray, D : nparray, S : nparray):
    """
    JIT-compiled function that calculates post-processing quantities 
    at selected ponts for multiple left- and right-hand sides.       

    Parameters
    ----------
    size : tuple 
        (Lx, Ly) : size of domain
        
    shape : tuple 
        (nxp, nY) : number of harmonic terms involved in xp and yp
                   directions
    
    points : numpy.ndarray[nP, 2] 
        2d array of point coordinates
    
    solution : numpy.ndarray[nRHS, nxp * nY, 3] 
        results of a Navier solution as a 3d array
    
    D : numpy.ndarray[nLHS, 3, 3] 
        3d array of bending stiffness terms
    
    S : numpy.ndarray[nLHS, 2, 2]
        3d array of shear stiffness terms

    Returns
    -------
    numpy.ndarray[nLHS, nRHS, nP, ...] 
        numpy array of post-processing items. The indices along 
        the last axpis denote the following quantities:
        
            0 : displacement z
            1 : rotation xp
            2 : rotation yp
            3 : curvature xp
            4 : curvature yp
            5 : curvature xpy
            6 : shear strain xpz
            7 : shear strain yz
            8 : moment xp
            9 : moment yp
            10 : moment xpy
            11 : shear force xp
            12 : shear force yp
    """
    M, N = shape
    nLHS = D.shape[0]
    nP = points.shape[0]
    nRHS = solution.shape[0]
    res2d = np.zeros((nLHS, nRHS, nP, 13), dtype=D.dtype)
    for m in range(M):
        for n in range(N):
            iMN = m * N + n
            for iRHS in prange(nRHS):
                for iP in prange(nP):
                    xp, yp = points[iP, :2]            
                    for iLHS in prange(nLHS):
                        res2d[iLHS, iRHS, iP, :] += \
                            _pproc_Mindlin_2D(size, m+1, n+1, xp, yp, 
                                              solution[iRHS, iMN], D[iLHS], 
                                              S[iLHS])
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


def shell_stiffness_data(shell):
    shell.stiffness_matrixp()
    layers = shell.layers()
    C_126, C_45 = [], []
    for layer in layers:
        Cm = layer.material.stiffness_matrixp()
        C_126.append(Cm[0:3, 0:3])
        C_45.append(Cm[3:, 3:])
    C_126 = np.stack(C_126)
    C_45 = np.stack(C_45)
    angles = np.stack([layer.angle for layer in layers])
    bounds = np.stack([[layer.tmin, layer.tmaxp]
                       for layer in layers])
    ABDS, shear_corrections, shear_factors = \
        stiffness_data_Mindlin(C_126, C_45, angles, bounds)
    ABDS[-2:, -2:] *= shear_corrections
    return C_126, C_45, bounds, angles, ABDS, shear_factors


if __name__ == '__main__':
    pass
