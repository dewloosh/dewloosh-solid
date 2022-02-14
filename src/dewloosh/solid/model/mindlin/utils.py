# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import inv
from numba import njit, prange
from collections import Iterable

from dewloosh.math.linalg import linspace1d
from dewloosh.math.array import ascont, clip1d, \
    atleast1d, atleast2d, atleast3d, atleast4d

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def rotation_matrices(angles: np.ndarray):
    """
    Returns transformation matrices T_126 and T_45 for each angle.
    Angles are expected in radians.
    """
    nL = len(angles)
    T_126 = np.zeros((nL, 3, 3), dtype=angles.dtype)
    T_45 = np.zeros((nL, 2, 2), dtype=angles.dtype)
    for iL in prange(nL):
        a = angles[iL] * np.pi / 180
        T_126[iL, 0, 0] = np.cos(a)**2
        T_126[iL, 0, 1] = np.sin(a)**2
        T_126[iL, 0, 2] = -np.sin(2 * a)
        T_126[iL, 1, 0] = T_126[iL, 0, 1]
        T_126[iL, 1, 1] = T_126[iL, 0, 0]
        T_126[iL, 1, 2] = -T_126[iL, 0, 2]
        T_126[iL, 2, 0] = np.cos(a) * np.sin(a)
        T_126[iL, 2, 1] = -T_126[iL, 2, 0]
        T_126[iL, 2, 2] = np.cos(a)**2 - np.sin(a)**2
        T_45[iL, 0, 0] = np.cos(a)
        T_45[iL, 0, 1] = -np.sin(a)
        T_45[iL, 1, 1] = T_45[iL, 0, 0]
        T_45[iL, 1, 0] = -T_45[iL, 0, 1]
    return T_126, T_45


@njit(nogil=True, parallel=True, cache=__cache)
def material_stiffness_matrices(C_126: np.ndarray, C_45: np.ndarray,
                                angles: np.ndarray):
    """
    Returns the components of the material stiffness matrices C_126 and C_45
    in the global system.
    """
    nL = len(C_126)
    C_126_g = np.zeros_like(C_126)
    C_45_g = np.zeros_like(C_45)
    R_126 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=C_126_g.dtype)
    R_126_inv = np.linalg.inv(R_126)
    R_45 = np.array([[2, 0], [0, 2]], dtype=C_126_g.dtype)
    R_45_inv = np.linalg.inv(R_45)
    T_126, T_45 = rotation_matrices(angles)
    for iL in prange(nL):
        C_126_g[iL] = T_126[iL] @ C_126[iL] @ R_126_inv @ T_126[iL].T @ R_126
        C_45_g[iL] = T_45[iL] @ C_45[iL] @ R_45_inv @ T_45[iL].T @ R_45
    return C_126_g, C_45_g


@njit(nogil=True, parallel=True, cache=__cache)
def shear_factors_MT(ABDS: np.ndarray, C_126: np.ndarray,
                     z: np.ndarray):
    """
    # FIXME Should work with parallel = True, does not. Reason is
    propably a race condition due to using explicit parallel loops.
    """
    A11 = ABDS[0, 0]
    B11 = ABDS[0, 3]
    D11 = ABDS[3, 3]
    A22 = ABDS[1, 1]
    B22 = ABDS[1, 4]
    D22 = ABDS[4, 4]
    nL = C_126.shape[0]  # number of layers

    # calculate shear factors
    eta_x = 1 / (A11 * D11 - B11**2)
    eta_y = 1 / (A22 * D22 - B22**2)
    shear_factors = np.zeros((nL, 2, 3), dtype=ABDS.dtype)

    for iL in prange(nL-1):
        for iP in prange(2):
            zi0 = z[iL, iP]
            zi1 = z[iL, iP + 1]
            dsfx = - eta_x * C_126[iL, 0, 0] * \
                (0.5 * (zi1**2 - zi0**2) * A11 - (zi1 - zi0) * B11)
            shear_factors[iL, 0, iP + 1:] += dsfx
            shear_factors[iL + 1:, 0, :] += dsfx
            dsfy = - eta_y * C_126[iL, 1, 1] * \
                (0.5 * (zi1**2 - zi0**2) * A22 - (zi1 - zi0) * B22)
            # these slicings probably cause a race condition
            shear_factors[iL, 1, iP + 1:] += dsfy
            shear_factors[iL + 1:, 1, :] += dsfy
    # last layer
    iL = nL - 1
    for iP in prange(2):
        zi0 = z[iL, iP]
        zi1 = z[iL, iP + 1]
        dsfx = - eta_x * C_126[iL, 0, 0] * \
            (0.5 * (zi1**2 - zi0**2) * A11 - (zi1 - zi0) * B11)
        shear_factors[iL, 0, iP + 1:] += dsfx
        dsfy = - eta_y * C_126[iL, 1, 1] * \
            (0.5 * (zi1**2 - zi0**2) * A22 - (zi1 - zi0) * B22)
        shear_factors[iL, 1, iP + 1:] += dsfy
    shear_factors[iL, :, 2] = 0.
    return shear_factors


@njit(nogil=True, cache=__cache)
def shear_factors_ST(ABDS: np.ndarray, C_126: np.ndarray,
                     z: np.ndarray):
    """
    Single-thread implementation of calculation of shear factors for
    multi-layer Mindlin shells.
    """
    A11 = ABDS[0, 0]
    B11 = ABDS[0, 3]
    D11 = ABDS[3, 3]
    A22 = ABDS[1, 1]
    B22 = ABDS[1, 4]
    D22 = ABDS[4, 4]
    nL = z.shape[0]  # number of layers

    # calculate shear factors
    eta_x = 1 / (A11 * D11 - B11**2)
    eta_y = 1 / (A22 * D22 - B22**2)
    shear_factors = np.zeros((nL, 2, 3), dtype=ABDS.dtype)

    for iL in range(nL):
        zi = z[iL]
        # first point through the thickness
        shear_factors[iL, 0, 0] = shear_factors[iL-1, 0, 2]
        shear_factors[iL, 1, 0] = shear_factors[iL-1, 1, 2]
        # second point through the thickness
        shear_factors[iL, 0, 1] = shear_factors[iL, 0, 0] - \
            eta_x * C_126[iL, 0, 0] * (0.5 * (zi[1]**2 - zi[0]**2) * A11 -
                                       (zi[1] - zi[0]) * B11)
        shear_factors[iL, 1, 1] = shear_factors[iL, 1, 0] - \
            eta_y * C_126[iL, 1, 1] * (0.5 * (zi[1]**2 - zi[0]**2) * A22 -
                                       (zi[1] - zi[0]) * B22)
        # third point through the thickness
        shear_factors[iL, 0, 2] = shear_factors[iL, 0, 0] - \
            eta_x * C_126[iL, 0, 0] * (0.5 * (zi[2]**2 - zi[0]**2) * A11 -
                                       (zi[2] - zi[0]) * B11)
        shear_factors[iL, 1, 2] = shear_factors[iL, 1, 0] - \
            eta_y * C_126[iL, 1, 1] * (0.5 * (zi[2]**2 - zi[0]**2) * A22 -
                                       (zi[2] - zi[0]) * B22)
    shear_factors[nL-1, :, 2] = 0.
    return shear_factors


@njit(nogil=True, parallel=True, cache=__cache)
def shear_correction_data(ABDS: np.ndarray, C_126: np.ndarray,
                          C_45: np.ndarray, bounds: np.ndarray):
    """
    FIXME : Results are OK, but a bit slower than expected when measured
    against the pure python implementation.
    """

    nL = bounds.shape[0]  # number of layers

    # z coordinate of 3 points per each layer
    z = points_of_layers(bounds, 3)

    # calculate shear factors
    shear_factors = shear_factors_ST(ABDS, C_126, z)

    # compile shear factors
    sf = np.zeros((nL, 2, 3), dtype=ABDS.dtype)
    for iL in prange(nL):
        monoms_inv = inv(np.array([[1, z, z**2] for z in z[iL]],
                                  dtype=ABDS.dtype))
        sf[iL, 0] = monoms_inv @ shear_factors[iL, 0]
        sf[iL, 1] = monoms_inv @ shear_factors[iL, 1]

    # potential energy using constant stress distribution
    # and unit shear force
    pot_c_x = 0.5 / ABDS[6, 6]
    pot_c_y = 0.5 / ABDS[7, 7]

    # positions and weights of Gauss-points
    gP = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)], dtype=ABDS.dtype)
    gW = np.array([5/9, 8/9, 5/9], dtype=ABDS.dtype)

    # potential energy using parabolic stress distribution
    # and unit shear force
    pot_p_x, pot_p_y = 0., 0.
    for iL in prange(nL):
        dJ = 0.5 * (bounds[iL, 1] - bounds[iL, 0])
        Gxi = C_45[iL, 0, 0]
        Gyi = C_45[iL, 1, 1]
        for iG in prange(3):
            ziG = 0.5 * ((bounds[iL, 1] + bounds[iL, 0]) +
                         gP[iG] * (bounds[iL, 1] - bounds[iL, 0]))
            monoms = np.array([1, ziG, ziG**2], dtype=ABDS.dtype)
            sfx = np.dot(monoms, sf[iL, 0])
            sfy = np.dot(monoms, sf[iL, 1])
            pot_p_x += 0.5 * (sfx**2) * dJ * gW[iG] / Gxi
            pot_p_y += 0.5 * (sfy**2) * dJ * gW[iG] / Gyi
    kx = pot_c_x / pot_p_x
    ky = pot_c_y / pot_p_y

    return np.array([[kx, 0], [0, ky]]), shear_factors


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def stiffness_data_Mindlin(C_126: np.ndarray, C_45: np.ndarray,
                           angles: np.ndarray, bounds: np.ndarray, nZ=3):
    """
    FIXME Call
        Ks, sf = shear_correction_data(ABDS, C_126_g, C_45_g, bounds, dtype)
    is a bit slow for some reason.
    """
    ABDS = np.zeros((8, 8), dtype=C_126.dtype)
    nL = C_126.shape[0]
    C_126_g, C_45_g = material_stiffness_matrices(C_126, C_45, angles)
    for iL in prange(nL):
        ABDS[0:3, 0:3] += C_126_g[iL] * (bounds[iL, 1] - bounds[iL, 0])
        ABDS[0:3, 3:6] += (1 / 2) * C_126_g[iL] * \
            (bounds[iL, 1]**2 - bounds[iL, 0]**2)
        ABDS[3:6, 3:6] += (1 / 3) * C_126_g[iL] * \
            (bounds[iL, 1]**3 - bounds[iL, 0]**3)
        ABDS[6:8, 6:8] += C_45_g[iL] * (bounds[iL, 1] - bounds[iL, 0])
    ABDS[3:6, 0:3] = ABDS[0:3, 3:6]
    Ks, sf = shear_correction_data(ABDS, C_126_g, C_45_g, bounds)
    return ABDS, Ks, sf


@njit(nogil=True, parallel=True, cache=__cache)
def _pproc_Mindlin_3D(ABDS: np.ndarray, sfx: np.ndarray, sfy: np.ndarray,
                      C_126: np.ndarray, C_45: np.ndarray,
                      bounds: np.ndarray, points: np.ndarray,
                      e_126_n: np.ndarray, c_126: np.ndarray,
                      e_45: np.ndarray):
    nRHS, nP, _ = e_126_n.shape
    layerinds = layers_of_points(points, bounds)

    # results
    s_126_n = np.zeros((nRHS, nP, 3), dtype=ABDS.dtype)
    s_126_m = np.zeros((nRHS, nP, 3), dtype=ABDS.dtype)
    s_45 = np.zeros((nRHS, nP, 2), dtype=ABDS.dtype)
    e_126_m = np.zeros((nRHS, nP, 3), dtype=ABDS.dtype)
    e_45_new = np.zeros((nRHS, nP, 2), dtype=ABDS.dtype)

    for iP in prange(nP):
        zP = points[iP, 2]
        lP = layerinds[iP]
        sfxz, sfyz = z_to_shear_factors(zP, sfx[lP, :], sfy[lP, :])
        for iRHS in prange(nRHS):
            e_126_m[iRHS, iP] = zP * c_126[iRHS, iP]
            s_126_n[iRHS, iP] = C_126[lP] @ e_126_n[iRHS, iP]
            s_126_m[iRHS, iP] = C_126[lP] @ e_126_m[iRHS, iP]
            s_45[iRHS, iP, 0] = sfxz * ABDS[6, 6] * e_45[iRHS, iP, 0]
            s_45[iRHS, iP, 1] = sfyz * ABDS[7, 7] * e_45[iRHS, iP, 1]
            e_45_new[iRHS, iP, 0] = s_45[iRHS, iP, 0] / C_45[lP, 0, 0]
            e_45_new[iRHS, iP, 1] = s_45[iRHS, iP, 1] / C_45[lP, 1, 1]

    return s_126_n, s_126_m, s_45, e_126_n, e_126_m, e_45_new


@njit(nogil=True, parallel=True, cache=__cache)
def _pproc_Mindlin_3D_s(ABDS: np.ndarray, sfx: np.ndarray, sfy: np.ndarray,
                        C_126: np.ndarray, C_45: np.ndarray,
                        bounds: np.ndarray, ppl: int,
                        e_126_n: np.ndarray, c_126: np.ndarray,
                        e_45: np.ndarray):
    nRHS, nP, _ = e_126_n.shape
    z = points_of_layers(bounds, ppl)
    nL = z.shape[0]

    # results
    s_126_n = np.zeros((nRHS, nP, nL, ppl, 3), dtype=ABDS.dtype)
    s_126_m = np.zeros((nRHS, nP, nL, ppl, 3), dtype=ABDS.dtype)
    s_45 = np.zeros((nRHS, nP, nL, ppl, 2), dtype=ABDS.dtype)
    e_126_m = np.zeros((nRHS, nP, nL, ppl, 3), dtype=ABDS.dtype)
    e_45_new = np.zeros((nRHS, nP, nL, ppl, 2), dtype=ABDS.dtype)

    for iL in prange(nL):
        for ippl in prange(ppl):
            zP = z[iL, ippl]
            sfxz, sfyz = z_to_shear_factors(zP, sfx[iL, :], sfy[iL, :])
            for iP in prange(nP):
                for iRHS in prange(nRHS):
                    e_126_m[iRHS, iP, iL, ippl] = zP * c_126[iRHS, iP]
                    s_126_n[iRHS, iP, iL, ippl] = C_126[iL] @ e_126_n[iRHS, iP]
                    s_126_m[iRHS, iP, iL, ippl] = C_126[iL] @ e_126_m[iRHS, iP]
                    s_45[iRHS, iP, iL, ippl, 0] = \
                        sfxz * ABDS[6, 6] * e_45[iRHS, iP, 0]
                    s_45[iRHS, iP, iL, ippl, 1] = \
                        sfyz * ABDS[7, 7] * e_45[iRHS, iP, 1]
                    e_45_new[iRHS, iP, iL, ippl, 0] = \
                        s_45[iRHS, iP, iL, ippl, 0] / C_45[iL, 0, 0]
                    e_45_new[iRHS, iP, iL, ippl, 1] = \
                        s_45[iRHS, iP, iL, ippl, 1] / C_45[iL, 1, 1]

    return s_126_n, s_126_m, s_45, e_126_n, e_126_m, e_45_new


@njit(nogil=True, parallel=True, cache=__cache)
def _pproc_Mindlin_3D_rgrid(ABDS: np.ndarray, coords2d: np.ndarray,
                            topo2d: np.ndarray, bounds: np.ndarray,
                            cell_per_layer: int, eshape: tuple,
                            sfx: np.ndarray, sfy: np.ndarray,
                            C_126: np.ndarray, C_45: np.ndarray,
                            e_126_n: np.ndarray, c_126: np.ndarray,
                            e_45: np.ndarray):
    nNEx, nNEy, nNEz = eshape
    nNE3d = nNEx * nNEy * nNEz
    nRHS = e_126_n.shape[0]
    nL = bounds.shape[0]
    nE2d, nNE2d = topo2d.shape
    nE3d = nE2d * nL * cell_per_layer
    nN3d = nE3d * nNE3d

    ftype = coords2d.dtype
    itype = topo2d.dtype
    s_126_n = np.zeros((nRHS, nN3d, 3), dtype=ftype)
    s_126_m = np.zeros((nRHS, nN3d, 3), dtype=ftype)
    s_45 = np.zeros((nRHS, nN3d, 2), dtype=ftype)
    e_126_m = np.zeros((nRHS, nN3d, 3), dtype=ftype)
    e_45_new = np.zeros((nRHS, nN3d, 2), dtype=ftype)
    coords3d = np.zeros((nN3d, 3), dtype=ftype)
    topo3d = np.zeros((nE3d, nNE3d), dtype=itype)

    # precalculate shear factors
    sfxz = np.zeros((nL, cell_per_layer, nNEz), dtype=ftype)
    sfyz = np.zeros((nL, cell_per_layer, nNEz), dtype=ftype)
    for iL in prange(nL):
        dZ = (bounds[iL, 1] - bounds[iL, 0]) / cell_per_layer
        ddZ = dZ / (nNEz - 1)
        for iCL in prange(cell_per_layer):
            for iNEz in prange(nNEz):
                zP = dZ * iCL + ddZ * iNEz
                sfxz[iL, iCL, iNEz], sfyz[iL, iCL, iNEz] = \
                    z_to_shear_factors(zP, sfx[iL, :], sfy[iL, :])

    for iL in prange(nL):
        dZ = (bounds[iL, 1] - bounds[iL, 0]) / cell_per_layer
        ddZ = dZ / (nNEz - 1)
        for iCL in prange(cell_per_layer):
            for iE2d in prange(nE2d):
                iE3d = nE2d * (cell_per_layer * iL + iCL) + iE2d
                for iNEx in prange(nNEx):
                    for iNEy in prange(nNEy):
                        iNE2d = iNEx * nNEy + iNEy
                        iN2d = topo2d[iE2d, iNE2d]
                        for iNEz in prange(nNEz):
                            iNE3d = iNEx * iNEy * nNEz + iNEz
                            iN3d = iE3d * nNE3d + iNE3d
                            topo3d[iE3d, iNE3d] = iN3d
                            coords3d[iN3d, 0] = coords2d[iN2d, 0]
                            coords3d[iN3d, 1] = coords2d[iN2d, 1]
                            coords3d[iN3d, 2] = dZ * iCL + ddZ * iNEz
                            for iRHS in prange(nRHS):
                                e_126_m[iRHS, iN3d] = \
                                    coords3d[iN3d, 2] * c_126[iRHS, iN2d]
                                s_126_n[iRHS, iN3d] = \
                                    C_126[iL] @ e_126_n[iRHS, iN2d]
                                s_126_m[iRHS, iN3d] = \
                                    C_126[iL] @ e_126_m[iRHS, iN3d]
                                s_45[iRHS, iN3d, 0] = \
                                    sfxz[iL, iCL, iNEz] * ABDS[6, 6] * \
                                    e_45[iRHS, iN2d, 0]
                                s_45[iRHS, iN3d, 1] = \
                                    sfyz[iL, iCL, iNEz] * ABDS[7, 7] * \
                                    e_45[iRHS, iN2d, 1]
                                e_45_new[iRHS, iN3d, 0] = \
                                    s_45[iRHS, iN3d, 0] / C_45[iL, 0, 0]
                                e_45_new[iRHS, iN3d, 1] = \
                                    s_45[iRHS, iN3d, 1] / C_45[iL, 1, 1]

    return s_126_n, s_126_m, s_45, e_126_n, e_126_m, e_45_new


def pproc_Mindlin_3D(ABDS: np.ndarray,
                     C_126: Iterable, C_45: Iterable,
                     bounds: Iterable, points: np.ndarray,
                     strains2d: np.ndarray, *args,
                     squeeze=True,
                     angles: Iterable = None, separate=True,
                     shear_factors: Iterable = None, **kwargs):
    # formatting
    ABDS = atleast3d(ABDS)
    strains2d = atleast4d(strains2d)
    if isinstance(C_126, np.ndarray):
        C_126 = atleast4d(C_126)
    else:
        C_126 = [atleast3d(C_126[i]) for i in range(len(C_126))]
    if isinstance(C_45, np.ndarray):
        C_45 = atleast4d(C_45)
    else:
        C_45 = [atleast3d(C_45[i]) for i in range(len(C_45))]
    if isinstance(bounds, np.ndarray):
        bounds = atleast3d(bounds)
    else:
        bounds = [atleast2d(bounds[i])
                  for i in range(len(bounds))]

    # transform material stiffness matrices to global
    if angles is not None:
        C_126_g = np.zeros_like(C_126)
        C_45_g = np.zeros_like(C_45)
        if isinstance(angles, np.ndarray):
            angles = atleast2d(angles)
        else:
            angles = [atleast1d(angles[i])
                      for i in range(len(angles))]
        for iLHS in range(len(angles)):
            C_126_g[iLHS], C_45_g[iLHS] = \
                material_stiffness_matrices(C_126[iLHS], C_45[iLHS],
                                            angles[iLHS])
        C_126 = C_126_g
        C_45 = C_45_g
        del C_126_g
        del C_45_g

    # calculate shear factors
    if shear_factors is None:
        shear_factors = []
        for i in range(len(C_126)):
            _, sf = shear_correction_data(ABDS[i], C_126[i], C_45[i],
                                          bounds[i])
            shear_factors.append(sf)
    else:
        if isinstance(shear_factors, np.ndarray):
            shear_factors = atleast4d(shear_factors)
        else:
            shear_factors = [atleast3d(shear_factors[i])
                             for i in range(len(shear_factors))]

    nLHS, nRHS, nP, _ = strains2d.shape
    e_126_n = ascont(strains2d[:, :, :, :3])
    s_126_n = np.zeros((nLHS, nRHS, nP, 3), dtype=ABDS.dtype)
    s_126_m = np.zeros((nLHS, nRHS, nP, 3), dtype=ABDS.dtype)
    s_45 = np.zeros((nLHS, nRHS, nP, 2), dtype=ABDS.dtype)
    e_126_m = np.zeros((nLHS, nRHS, nP, 3), dtype=ABDS.dtype)
    e_45 = np.zeros((nLHS, nRHS, nP, 2), dtype=ABDS.dtype)

    for i in range(nLHS):
        sfx_i = ascont(shear_factors[i][:, 0, :])
        sfy_i = ascont(shear_factors[i][:, 1, :])
        e_126_n_i = ascont(strains2d[i, :, :, :3])
        c_126_i = ascont(strains2d[i, :, :, 3:6])
        e_45_i = ascont(strains2d[i, :, :, 6:8])
        s_126_n[i], s_126_m[i], s_45[i], e_126_n[i], e_126_m[i], e_45[i] = \
            _pproc_Mindlin_3D(ABDS[i], sfx_i, sfy_i, C_126[i], C_45[i],
                              bounds[i], points, e_126_n_i, c_126_i, e_45_i)

    if separate:
        res3d = s_126_n, s_126_m, s_45, e_126_n, e_126_m, e_45
    else:
        res3d = s_126_n + s_126_m, s_45, e_126_n + e_126_m, e_45

    if squeeze:
        return tuple(map(np.squeeze, res3d))
    else:
        return res3d


@njit(nogil=True, cache=__cache)
def glob_to_loc_layer(point: np.ndarray, bounds: np.ndarray):
    return (point[2] - bounds[0]) / (bounds[1] - bounds[0])


@njit(nogil=True, cache=__cache)
def z_to_shear_factors(z, sfx, sfy):
    monoms = np.array([1, z, z**2], dtype=sfx.dtype)
    return np.dot(monoms, sfx), np.dot(monoms, sfy)


@njit(nogil=True, cache=__cache)
def layers_of_points(points: np.ndarray, bounds: np.ndarray):
    nL = bounds.shape[0]
    bins = np.zeros((nL+1,), dtype=points.dtype)
    bins[0] = bounds[0, 0]
    bins[1:] = bounds[:, 1]
    return clip1d(np.digitize(points[:, 2], bins) - 1, 0, nL-1)


@njit(['f8[:, :](f8[:, :], i8)', 'f4[:, :](f4[:, :], i8)'],
      nogil=True, parallel=True, cache=__cache)
def points_of_layers(bounds: np.ndarray, nppl=3):
    nL = bounds.shape[0]
    res = np.zeros((nL, nppl), dtype=bounds.dtype)
    for iL in prange(nL):
        res[iL] = linspace1d(bounds[iL, 0], bounds[iL, 1], nppl)
    return res


if __name__ == '__main__':
    pass
