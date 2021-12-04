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


@njit(nogil = True, cache = __cache)
def glob_to_loc_layer(point : np.ndarray, bounds : np.ndarray):
    return (point[2] - bounds[0]) / (bounds[1] - bounds[0])


@njit(nogil = True, cache = __cache)
def z_to_shear_factors(z, sfx, sfy):
    monoms = np.array([1, z, z**2], dtype = sfx.dtype)
    return np.dot(monoms, sfx), np.dot(monoms, sfy)


@njit(nogil = True, cache = __cache)
def layers_of_points(points : np.ndarray, bounds : np.ndarray):
    nL = bounds.shape[0]
    bins = np.zeros((nL+1,), dtype = points.dtype)
    bins[0] = bounds[0, 0]
    bins[1:] = bounds[:, 1]
    return clip1d(np.digitize(points[:, 2], bins) - 1, 0, nL-1)


@njit(['f8[:, :](f8[:, :], i8)', 'f4[:, :](f4[:, :], i8)'],
      nogil = True, parallel = True, cache = __cache)
def points_of_layers(bounds : np.ndarray, nppl = 3):
    nL = bounds.shape[0]
    res = np.zeros((nL, nppl), dtype = bounds.dtype)
    for iL in prange(nL):
        res[iL] = linspace1d(bounds[iL, 0], bounds[iL, 1], nppl)
    return res


@njit(nogil = True, parallel = True, cache = __cache)
def rotation_matrices(angles : float, dtype = np.float32):
    """
    Returns transformation matrices T_126 and T_45 for each angle.
    Angles are expected in radians.
    """
    nL = len(angles)
    T_126 = np.zeros((nL, 3, 3), dtype = dtype)
    T_45 = np.zeros((nL, 2, 2), dtype = dtype)
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
    return T_126.astype(dtype), T_45.astype(dtype)


@njit(nogil = True, parallel = True, cache = __cache)
def material_stiffness_matrices(C_126 : np.ndarray, C_45 : np.ndarray,
                                angles : np.ndarray, dtype = np.float32):
    """
    Returns the components of the material stiffness matrices C_126 and C_45
    in the global system.
    """
    nL = len(C_126)
    C_126_g = np.zeros_like(C_126)
    C_45_g = np.zeros_like(C_45)
    R_126 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]]).astype(C_126_g.dtype)
    R_126_inv = np.linalg.inv(R_126)
    R_45 = np.array([[2, 0], [0, 2]]).astype(C_126_g.dtype)
    R_45_inv = np.linalg.inv(R_45)
    T_126, T_45 = rotation_matrices(angles, C_126_g.dtype)
    for iL in prange(nL):
        C_126_g[iL] = T_126[iL] @ C_126[iL] @ R_126_inv @ T_126[iL].T @ R_126
        C_45_g[iL] = T_45[iL] @ C_45[iL] @ R_45_inv @ T_45[iL].T @ R_45
    return C_126_g.astype(dtype), C_45_g.astype(dtype)


@njit(nogil = True, parallel = True, cache = __cache)
def shear_factors_MT(ABDS : np.ndarray, C_126 : np.ndarray,
                     z : np.ndarray, dtype = np.float32):
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
    shear_factors = np.zeros((nL, 2, 3), dtype = dtype)

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
    return shear_factors.astype(dtype)


@njit(nogil = True, cache = __cache)
def shear_factors_ST(ABDS : np.ndarray, C_126 : np.ndarray,
                     z : np.ndarray, dtype = np.float32):
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
    shear_factors = np.zeros((nL, 2, 3), dtype = dtype)

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
    return shear_factors.astype(dtype)


@njit(nogil = True, parallel = True, cache = __cache)
def shear_correction_data(ABDS : np.ndarray, C_126 : np.ndarray,
                          C_45 : np.ndarray, bounds : np.ndarray,
                          dtype = np.float32):
    """
    FIXME : Results are OK, but a bit slower than expected when measured
    against the pure python implementation.
    """

    nL = bounds.shape[0]  # number of layers

    # z coordinate of 3 points per each layer
    z = points_of_layers(bounds, 3, dtype)

    # calculate shear factors
    shear_factors = shear_factors_ST(ABDS, C_126, z, dtype)

    # compile shear factors
    sf = np.zeros((nL, 2, 3), dtype = dtype)
    for iL in prange(nL):
        monoms_inv = inv(np.array([[1, z, z**2] for z in z[iL]],
                                  dtype = dtype))
        sf[iL, 0] = monoms_inv @ shear_factors[iL, 0]
        sf[iL, 1] = monoms_inv @ shear_factors[iL, 1]

    # potential energy using constant stress distribution
    # and unit shear force
    pot_c_x = 0.5 / ABDS[6, 6]
    pot_c_y = 0.5 / ABDS[7, 7]

    # positions and weights of Gauss-points
    gP = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)], dtype = dtype)
    gW = np.array([5/9, 8/9, 5/9], dtype = dtype)

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
            monoms = np.array([1, ziG, ziG**2], dtype = dtype)
            sfx = np.dot(monoms, sf[iL, 0])
            sfy = np.dot(monoms, sf[iL, 1])
            pot_p_x += 0.5 * (sfx**2) * dJ * gW[iG] / Gxi
            pot_p_y += 0.5 * (sfy**2) * dJ * gW[iG] / Gyi
    kx = pot_c_x / pot_p_x
    ky = pot_c_y / pot_p_y

    return np.array([[kx, 0], [0, ky]]).astype(dtype), \
        shear_factors.astype(dtype)


@njit(nogil = True, parallel = True, cache = __cache)
def stiffness_data_Mindlin(C_126 : np.ndarray, C_45 : np.ndarray,
                           angles : np.ndarray, bounds : np.ndarray, nZ = 3,
                           dtype = np.float32):
    """
    FIXME Call
        Ks, sf = shear_correction_data(ABDS, C_126_g, C_45_g, bounds, dtype)
    is a bit slow for some reason.
    """
    ABDS = np.zeros((8, 8), dtype = dtype)
    nL = C_126.shape[0]
    bounds = bounds.astype(dtype)
    C_126_g, C_45_g = material_stiffness_matrices(C_126, C_45, angles, dtype)
    for iL in prange(nL):
        ABDS[0:3, 0:3] += C_126_g[iL] * (bounds[iL, 1] - bounds[iL, 0])
        ABDS[0:3, 3:6] += (1 / 2) * C_126_g[iL] * \
            (bounds[iL, 1]**2 - bounds[iL, 0]**2)
        ABDS[3:6, 3:6] += (1 / 3) * C_126_g[iL] * \
            (bounds[iL, 1]**3 - bounds[iL, 0]**3)
        ABDS[6:8, 6:8] += C_45_g[iL] * (bounds[iL, 1] - bounds[iL, 0])
    ABDS[3:6, 0:3] = ABDS[0:3, 3:6]
    Ks, sf = shear_correction_data(ABDS, C_126_g, C_45_g, bounds, dtype)
    return ABDS, Ks, sf


if __name__ == '__main__':
    from dewloosh.solidanics.material.hooke import Lamina
    from dewloosh.solidanics.model.mindlin import MindlinShell as Shell
    from dewloosh.math.array import repeat
    from time import time

    kdef = 0.8  #Service class 1
    C24 = {
        'E1' : 11000000. / (1 + kdef),
        'E2' : 11000000. / 20 / (1 + kdef),
        'G12' : 690000. / (1 + kdef),
        'G23' : 690000. / 10 / (1 + kdef),
        'nu12' : 0.4,
        }

    material = Lamina(**C24, stype = 'shell')
    shell = Shell()
    layer_1 = shell.Layer(m = material, t = 0.04, a = 0.)
    layer_2 = shell.Layer(m = material, t = 0.02, a = 90.)
    layer_3 = shell.Layer(m = material, t = 0.02, a = 0.)
    layer_4 = shell.Layer(m = material, t = 0.02, a = 90.)
    layer_5 = shell.Layer(m = material, t = 0.04, a = 0.)
    shell.add_layers(layer_1, layer_2, layer_3, layer_4, layer_5)
    ABDS = shell.stiffness_matrix()

    layers = shell.layers()
    bounds = np.stack([[layer.tmin, layer.tmax] for layer in layers])
    nppl = 3
    pol = points_of_layers(bounds, nppl)

    layers = shell.layers()
    bounds = np.stack([[layer.tmin, layer.tmax]
                       for layer in layers]).astype(np.float32)
    nppl = 3
    pol = points_of_layers(bounds, nppl)

    def stiffness_matrix(shell, material, dtype = np.float32):
        Cm = material.stiffness_matrix()
        Cm_126 = Cm[0:3, 0:3]
        Cm_45 = Cm[3:, 3:]
        layers = shell.layers()
        C_126 = repeat(Cm_126, len(layers)).astype(dtype)
        C_45 = repeat(Cm_45, len(layers)).astype(dtype)
        angles = np.stack([layer.angle for layer in layers]).astype(dtype)
        bounds = np.stack([[layer.tmin, layer.tmax]
                           for layer in layers]).astype(dtype)
        ABDS, Ks, sf = stiffness_data_Mindlin(C_126, C_45, angles, bounds,
                                              dtype)
        ABDS[-2:, -2:] *= Ks
        return ABDS

    t1 = time()
    for i in range(40):
        shell.stiffness_matrix()
    print(time()-t1)

    t1 = time()
    for i in range(40):
        stiffness_matrix(shell, material)
    print(time()-t1)
