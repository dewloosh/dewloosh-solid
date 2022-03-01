# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange
from collections import Iterable
from typing import Union

from dewloosh.math.array import atleast1d, atleast4d
from dewloosh.math.numint import GaussPoints as Gauss

from dewloosh.geom.cells import L2 as Line
from dewloosh.geom.utils import lengths_of_lines2

from .utils import to_range
from ..elem import FiniteElement
from ..model.beam import BernoulliBeam


__all__ = ['Bernoulli']


__cache = True


ArrayOrFloat = Union[ndarray, float]


@njit(nogil=True, cache=__cache)
def shape_function_values(x, x1, x2):
    """
    Evaluates the shape functions at a point x
    in the range [x1, x2]. 
    """
    return np.array([
        # for u_s and theta_x
        (x - x2)/(x1 - x2),
        (-x + x1)/(x1 - x2),
        # for v_s and theta_z
        (x - x2)**2 * (-2*x + 3*x1 - x2) / (x1 - x2)**3,
        (x - x1) * (x - x2)**2/(x1 - x2)**2,
        (x - x1)**2 * (2*x + x1 - 3*x2) / (x1 - x2)**3,
        (x - x1)**2 * (x - x2)/(x1 - x2)**2,
        # for w_s and theta_y
        (x - x2)**2*(-2*x + 3*x1 - x2) / (x1 - x2)**3,
        (-x + x1) * (x - x2)**2 / (x1 - x2)**2,
        (x - x1)**2 * (2*x + x1 - 3*x2) / (x1 - x2)**3,
        (-x + x2) * (x - x1)**2 / (x1 - x2)**2
    ])


@njit(nogil=True, cache=__cache)
def shp_loc(r):
    """
    Evaluates the shape functions at a point x
    in the range [-1, 1]. 
    """
    return shape_function_values(r, -1, 1)


@njit(nogil=True, parallel=True, cache=__cache)
def shp_loc_bulk(pcoords: ndarray):
    """
    Evaluates the shape functions at several points
    in the range [-1, 1].
    """
    nP = pcoords.shape[0]
    res = np.zeros((nP, 10), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shp_loc(pcoords[iP])
    return res


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_1(x, x1, x2):
    """
    Evaluates the first derivatives of the shape 
    functions at a point x in the range [x1, x2]. 
    """
    return np.array([
        1/(x1 - x2),
        -1/(x1 - x2),
        -6*(x - x1)*(x - x2)/(x1 - x2)**3,
        -(x - x2)*(-3*x + 2*x1 + x2)/(x1 - x2)**2,
        6*(x - x1)*(x - x2)/(x1 - x2)**3,
        -(x - x1)*(-3*x + x1 + 2*x2)/(x1 - x2)**2,
        -6*(x - x1)*(x - x2)/(x1 - x2)**3,
        (x - x2)*(-3*x + 2*x1 + x2)/(x1 - x2)**2,
        6*(x - x1)*(x - x2)/(x1 - x2)**3,
        (x - x1)*(-3*x + x1 + 2*x2)/(x1 - x2)**2
    ])


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_2(x, x1, x2):
    """
    Evaluates the second derivatives of the shape 
    functions at a point x in the range [x1, x2]. 
    """
    return np.array([
        0,
        0,
        6*(-2*x + x1 + x2)/(x1 - x2)**3,
        2*(3*x - x1 - 2*x2)/(x1 - x2)**2,
        6*(2*x - x1 - x2)/(x1 - x2)**3,
        2*(3*x - 2*x1 - x2)/(x1 - x2)**2,
        6*(-2*x + x1 + x2)/(x1 - x2)**3,
        2*(-3*x + x1 + 2*x2)/(x1 - x2)**2,
        6*(2*x - x1 - x2)/(x1 - x2)**3,
        2*(-3*x + 2*x1 + x2)/(x1 - x2)**2
    ])


@njit(nogil=True, cache=__cache)
def shape_function_derivatives_3(x, x1, x2):
    """
    Evaluates the third derivatives of the shape 
    functions at a point x in the range [x1, x2]. 
    """
    return np.array([
        0,
        0,
        -12/(x1 - x2)**3,
        6/(x1 - x2)**2,
        12/(x1 - x2)**3,
        6/(x1 - x2)**2,
        -12/(x1 - x2)**3,
        -6/(x1 - x2)**2,
        12/(x1 - x2)**3,
        -6/(x1 - x2)**2
    ])


@njit(nogil=True, cache=__cache)
def shape_function_derivatives(x, x1, x2):
    """
    Evaluates the derivatives of the shape 
    functions at a point x in the range [x1, x2]. 
    """
    res = np.zeros((10, 3))
    res[:, 0] = shape_function_derivatives_1(x, x1, x2)
    res[:, 1] = shape_function_derivatives_2(x, x1, x2)
    res[:, 2] = shape_function_derivatives_3(x, x1, x2)
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_loc_bulk(pcoords: ndarray):
    """
    Evaluates the derivatives of the shape 
    functions at several points in the range [-1, 1]. 
    """
    nP = pcoords.shape[0]
    res = np.zeros((nP, 10, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_derivatives(pcoords[iP], -1, 1)
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def gdshp_bulk_v1(pcoords: ndarray, lengths: ndarray):
    """
    Calculates derivatives of shape functions along the global axes
    using coordinates and the lengths of the elements.

    The evaluation points (probably the Gauss points) are expected in the 
    interval [-1, 1].
    """
    nP = pcoords.shape[0]
    nE = lengths.shape[0]
    res = np.zeros((nE, nP, 10, 3), dtype=pcoords.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            res[iE, iP] = shape_function_derivatives(pcoords[iP]*lengths[iE],
                                                     0, lengths[iE])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def gdshp_bulk_v2(dshp: ndarray, jac: ndarray):
    """
    Calculates derivatives of shape functions along the global axes
    using derivatives along local axes evaulates at some points in
    the interval [-1, 1], and jacobians of local-to-global mappings.
    """
    #  dshp (nP, 10, 3)
    nP = dshp.shape[0]
    nE = jac.shape[0]
    res = np.zeros((nE, nP, 10, 3), dtype=dshp.dtype)
    for iE in prange(nE):
        for jP in prange(nP):
            res[iE, jP, :, 0] = dshp[jP, :, 0] / jac[iE, jP, 0, 0]
            res[iE, jP, :, 1] = dshp[jP, :, 1] / jac[iE, jP, 0, 0]**2
            res[iE, jP, :, 2] = dshp[jP, :, 2] / jac[iE, jP, 0, 0]**3
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix(shp: ndarray, gdshp: ndarray):
    nE, nP = gdshp.shape[:2]
    N = np.zeros((nE, nP, 6, 12), dtype=gdshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            # u_s
            N[iE, iP, 0, 0] = shp[iP, 0]
            N[iE, iP, 0, 6] = shp[iP, 1]
            # v_s
            N[iE, iP, 1, 1] = shp[iP, 2]
            N[iE, iP, 1, 5] = shp[iP, 3]
            N[iE, iP, 1, 7] = shp[iP, 4]
            N[iE, iP, 1, 11] = shp[iP, 5]
            # w_s
            N[iE, iP, 2, 2] = shp[iP, 6]
            N[iE, iP, 2, 4] = shp[iP, 7]
            N[iE, iP, 2, 8] = shp[iP, 8]
            N[iE, iP, 2, 10] = shp[iP, 9]
            # theta_x
            N[iE, iP, 3, 3] = shp[iP, 0]
            N[iE, iP, 3, 9] = shp[iP, 1]
            # theta_y
            N[iE, iP, 4, 2] = - gdshp[iE, iP, 6, 0]
            N[iE, iP, 4, 4] = - gdshp[iE, iP, 7, 0]
            N[iE, iP, 4, 8] = - gdshp[iE, iP, 8, 0]
            N[iE, iP, 4, 10] = - gdshp[iE, iP, 9, 0]
            # theta_z
            N[iE, iP, 5, 1] = gdshp[iE, iP, 2, 0]
            N[iE, iP, 5, 5] = gdshp[iE, iP, 3, 0]
            N[iE, iP, 5, 7] = gdshp[iE, iP, 4, 0]
            N[iE, iP, 5, 11] = gdshp[iE, iP, 5, 0]
    return N


@njit(nogil=True, parallel=True, cache=__cache)
def body_load_vector(values: ndarray, shp: ndarray, gdshp: ndarray,
                     djac: ndarray, weights: ndarray):
    # values (nE, nG, nDOF=6, nRHS)
    nRHS = values.shape[-1]
    nE, nG = djac.shape
    res = np.zeros((nE, 12, nRHS), dtype=values.dtype)
    for iG in range(nG):
        for iRHS in prange(nRHS):
            for iE in prange(nE):
                # approximate
                v = values[iE, 0, :, iRHS] * shp[iG, 0] + \
                    values[iE, 1, :, iRHS] * shp[iG, 1]
                # sum Fx
                res[iE, 0, iRHS] += shp[iG, 0] * \
                    v[0] * djac[iE, iG] * weights[iG]
                res[iE, 6, iRHS] += shp[iG, 1] * \
                    v[0] * djac[iE, iG] * weights[iG]
                # sum Fy
                res[iE, 1, iRHS] += shp[iG, 2] * \
                    v[1] * djac[iE, iG] * weights[iG]
                res[iE, 5, iRHS] += shp[iG, 3] * \
                    v[1] * djac[iE, iG] * weights[iG]
                res[iE, 7, iRHS] += shp[iG, 4] * \
                    v[1] * djac[iE, iG] * weights[iG]
                res[iE, 11, iRHS] += shp[iG, 5] * \
                    v[1] * djac[iE, iG] * weights[iG]
                res[iE, 1, iRHS] -= gdshp[iE, iG, 2, 0] * \
                    v[4] * djac[iE, iG] * weights[iG]
                res[iE, 5, iRHS] -= gdshp[iE, iG, 3, 0] * \
                    v[4] * djac[iE, iG] * weights[iG]
                res[iE, 7, iRHS] -= gdshp[iE, iG, 4, 0] * \
                    v[4] * djac[iE, iG] * weights[iG]
                res[iE, 11, iRHS] -= gdshp[iE, iG, 5, 0] * \
                    v[4] * djac[iE, iG] * weights[iG]
                # sum Fz
                res[iE, 2, iRHS] += shp[iG, 6] * \
                    v[2] * djac[iE, iG] * weights[iG]
                res[iE, 4, iRHS] += shp[iG, 7] * \
                    v[2] * djac[iE, iG] * weights[iG]
                res[iE, 8, iRHS] += shp[iG, 8] * \
                    v[2] * djac[iE, iG] * weights[iG]
                res[iE, 10, iRHS] += shp[iG, 9] * \
                    v[2] * djac[iE, iG] * weights[iG]
                res[iE, 2, iRHS] -= gdshp[iE, iG, 6, 0] * \
                    v[5] * djac[iE, iG] * weights[iG]
                res[iE, 4, iRHS] -= gdshp[iE, iG, 7, 0] * \
                    v[5] * djac[iE, iG] * weights[iG]
                res[iE, 8, iRHS] -= gdshp[iE, iG, 8, 0] * \
                    v[5] * djac[iE, iG] * weights[iG]
                res[iE, 10, iRHS] -= gdshp[iE, iG, 9, 0] * \
                    v[5] * djac[iE, iG] * weights[iG]
                # sum Mx
                res[iE, 3, iRHS] += shp[iG, 0] * \
                    v[3] * djac[iE, iG] * weights[iG]
                res[iE, 9, iRHS] += shp[iG, 1] * \
                    v[3] * djac[iE, iG] * weights[iG]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def calculate_element_forces_bulk(dofsol: ndarray, B: ndarray,
                                  D: ndarray, shp: ndarray,
                                  gdshp: ndarray, body_forces: ndarray):
    """
    Calculates internal forces from dof solution.
    """
    nE = dofsol.shape[0]
    nP = B.shape[1]
    res = np.zeros((6, nE, nP), dtype=dofsol.dtype)
    for i in prange(nE):
        for j in prange(nP):
            pyy, pzz = body_forces[i, 0, 4:] * shp[j, 0] + \
                body_forces[i, 1, 4:] * shp[j, 1]
            N, T, My, Mz = D[i] @ B[i, j] @ dofsol[i]
            res[0, i, j] = N
            # Vy
            res[1, i, j] = -D[i, 3, 3] * (
                gdshp[i, j, 2, 2] * dofsol[i, 1] +
                gdshp[i, j, 3, 2] * dofsol[i, 5] +
                gdshp[i, j, 4, 2] * dofsol[i, 7] +
                gdshp[i, j, 5, 2] * dofsol[i, 11]) - pzz
            # Vz
            res[2, i, j] = -D[i, 2, 2] * (
                gdshp[i, j, 6, 2] * dofsol[i, 2] +
                gdshp[i, j, 7, 2] * dofsol[i, 4] +
                gdshp[i, j, 8, 2] * dofsol[i, 8] +
                gdshp[i, j, 9, 2] * dofsol[i, 10]) + pyy
            res[3, i, j] = T
            res[4, i, j] = My
            res[5, i, j] = Mz
    return res


njit(nogil=True, parallel=True, cache=__cache)
def interpolate_element_forces_bulk(edata: ndarray, pcoords: ndarray):
    """
    Approximates internal forces in several elements at several points, 
    from previously calculated internal force data.

    The points are exprected in the range [0, 1]. 
    """
    nE = edata.shape[0]
    nP = pcoords.shape[0]
    res = np.zeros((nE, nP, 6))
    for iP in prange(nP):
        shp_i = 1 - pcoords[iP]
        shp_j = pcoords[iP]
        for iE in prange(nE):
            res[iE, iP, :] = edata[iE, 0] * shp_i + edata[iE, 1] * shp_j
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_det_bulk(jac: ndarray):
    nE, nG = jac.shape[:2]
    res = np.zeros((nE, nG), dtype=jac.dtype)
    for iE in prange(nE):
        res[iE, :] = jac[iE, :, 0, 0]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_matrix_bulk(dshp: ndarray, ecoords: ndarray):
    lengths = lengths_of_lines2(ecoords)
    nE = ecoords.shape[0]
    nG = dshp.shape[0]
    res = np.zeros((nE, nG, 1, 1), dtype=dshp.dtype)
    for iE in prange(nE):
        res[iE, :, 0, 0] = lengths[iE] / 2
    return res


class Bernoulli2(Line, BernoulliBeam, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': Gauss(2)
    }

    def jacobian_matrix(self, *args, dshp=None, ecoords=None, topo=None, **kwargs):
        ecoords = self.local_coordinates(
            topo=topo) if ecoords is None else ecoords
        return jacobian_matrix_bulk(dshp, ecoords)

    def jacobian(self, *args, jac=None, **kwargs):
        return jacobian_det_bulk(jac)

    @classmethod
    def shape_function_values(cls, pcoords: ArrayOrFloat, *args,
                              rng: Iterable = None, **kwargs) -> ndarray:
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        pcoords = to_range(pcoords, source=rng, target=[-1, 1])
        if isinstance(pcoords, ndarray):
            return shp_loc_bulk(pcoords)
        else:
            return shp_loc(pcoords)

    @classmethod
    def shape_function_derivatives(cls, pcoords=None, *args,
                                   rng: Iterable = None, lengths: ndarray = None,
                                   jac: ndarray = None, dshp: ndarray = None, **kwargs):
        if pcoords is None:
            if not (dshp is not None and jac is not None):
                raise RuntimeError(
                    "Either 'pcoords', or both 'dshp' and 'jac' must be provided")
            return gdshp_bulk_v2(dshp, jac)
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        pcoords = to_range(pcoords, source=rng, target=[-1, 1])
        if lengths is not None:
            return gdshp_bulk_v1(pcoords, lengths)
        elif jac is not None:
            return gdshp_bulk_v2(dshp_loc_bulk(pcoords), jac)
        else:
            if isinstance(pcoords, ndarray):
                return dshp_loc_bulk(pcoords)  # (nP, 10, 3)
            else:
                return shape_function_derivatives(pcoords, -1, 1)

    @classmethod
    def shape_function_matrix(cls, pcoords=None, *args, rng=None, **kwargs):
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        shp = cls.shape_function_values(pcoords=pcoords, rng=rng)
        gdshp = cls.shape_function_derivatives(*args, pcoords=pcoords,
                                               rng=rng, **kwargs)
        return shape_function_matrix(shp, gdshp)

    def integrate_body_loads(self, values: ndarray) -> ndarray:
        values = atleast4d(values)
        qpos, qweights = self.quadrature['full']
        rng = np.array([-1., 1.])
        shp = self.shape_function_values(pcoords=qpos, rng=rng)
        dshp = self.shape_function_derivatives(pcoords=qpos, rng=rng)
        ecoords = self.local_coordinates()
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
        djac = self.jacobian(jac=jac)
        gdshp = self.shape_function_derivatives(
            qpos, rng=rng, jac=jac, dshp=dshp)
        return body_load_vector(values, shp, gdshp, djac, qweights)

    @classmethod
    def _calc_element_forces_bulk(cls, dofsol: ndarray, B: ndarray, D: ndarray,
                                  shp: ndarray, gdshp: ndarray,
                                  body_forces: ndarray) -> ndarray:
        return calculate_element_forces_bulk(dofsol, B, D, shp, gdshp, body_forces)

    @classmethod
    def interpolate_element_forces(cls, edata: ndarray, pcoords: ndarray,
                                   rng: Iterable = None) -> ndarray:
        pcoords = to_range(pcoords, source=rng, target=[0, 1])
        return interpolate_element_forces_bulk(edata, pcoords)
