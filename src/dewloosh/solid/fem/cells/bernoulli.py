# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray, swapaxes as swap
from collections import Iterable
from typing import Union

from dewloosh.core import squeeze

from dewloosh.math.array import atleast1d, atleastnd, ascont
from dewloosh.math.utils import to_range

from ..postproc import approx_element_solution_bulk, calculate_internal_forces_bulk
from ..utils import tr_cells_1d_in_multi, element_dof_solution_bulk
from ..elem import FiniteElement
from ..model.beam import BernoulliBeam, calculate_shear_forces

from .utils.bernoulli import global_shape_function_derivatives_bulk as gdshpB2, \
    shape_function_matrix_bulk, body_load_vector_bulk


__all__ = ['BernoulliBase']


ArrayOrFloat = Union[ndarray, float]


class BernoulliBase(BernoulliBeam, FiniteElement):

    qrule: str = None
    quadrature: dict = None

    def shape_function_values(self, pcoords: ArrayOrFloat, *args,
                              rng: Iterable = None, lengths=None,
                              **kwargs) -> ndarray:
        """
        ---
        (nE, nP, nNE=2, nDOF=6)
        """
        raise NotImplementedError

    def shape_function_derivatives(self, pcoords=None, *args, rng=None,
                                   jac: ndarray = None, dshp: ndarray = None,
                                   lengths=None, **kwargs) -> ndarray:
        """
        ---
        (nE, nP, nNE=2, nDOF=6, 3)
        """
        raise NotImplementedError

    def shape_function_matrix(self, pcoords=None, *args, rng=None,
                              lengths=None, **kwargs) -> ndarray:
        """
        ---
        (nE, nP, nDOF, nDOF * nNODE)
        """
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        lengths = self.lengths() if lengths is None else lengths
        shp = self.shape_function_values(
            pcoords=pcoords, rng=rng, lengths=lengths)
        gdshp = self.shape_function_derivatives(*args, pcoords=pcoords,
                                                rng=rng, lengths=lengths, **kwargs)
        return shape_function_matrix_bulk(shp, gdshp).astype(float)

    def integrate_body_loads(self, values: ndarray) -> ndarray:
        """
        values (nE, nNE * nDOF, nRHS)
        """
        values = atleastnd(values, 3, back=True)
        # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
        values = np.swapaxes(values, 1, 2)
        values = ascont(values)
        qpos, qweights = self.quadrature['full']
        rng = np.array([-1., 1.])
        shp = self.shape_function_values(
            pcoords=qpos, rng=rng)  # (nE, nP, nNE=2, nDOF=6)
        dshp = self.shape_function_derivatives(
            pcoords=qpos, rng=rng)  # (nE, nP, nNE=2, nDOF=6, 3)
        ecoords = self.local_coordinates()
        jac = self.jacobian_matrix(
            dshp=dshp, ecoords=ecoords)  # (nE, nP, 1, 1)
        djac = self.jacobian(jac=jac)  # (nE, nG)
        gdshp = self.shape_function_derivatives(
            qpos, rng=rng, jac=jac, dshp=dshp)  # (nE, nP, nNE=2, nDOF=6, 3)
        return body_load_vector_bulk(values, shp, gdshp, djac, qweights).astype(float)

    @squeeze(True)
    def internal_forces(self, *args, cells=None, points=None, rng=None,
                        flatten=True, target='local', **kwargs):
        """
        Returns strains for the cells.

        Parameters
        ----------
        points : scalar or array, Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng' parameter. 
                If not provided, results are returned for the nodes of the selected elements.
                Default is None.

        rng : array, Optional
            Range where the points of evauation are understood. Default is [0, 1].

        cells : integer or array, Optional.
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.

        ---
        Returns
        -------
        numpy array or dictionary
        (nE, nP * nSTRE, nRHS) if flatten else (nE, nP, nSTRE, nRHS) 

        """
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id.to_numpy())
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
        else:
            cells = np.s_[:]

        dofsol = self.pointdata.dofsol.to_numpy()
        dofsol = atleastnd(dofsol, 3, back=True)
        nP, nDOF, nRHS = dofsol. shape
        dofsol = dofsol.reshape(nP * nDOF, nRHS)
        gnum = self.global_dof_numbering()[cells]
        dcm = self.direction_cosine_matrix()[cells]
        ecoords = self.local_coordinates()[cells]

        # transform dofsol to cell-local frames
        dofsol = element_dof_solution_bulk(dofsol, gnum)  # (nE, nEVAB, nRHS)
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))
        dofsol = tr_cells_1d_in_multi(dofsol, dcm)
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nEVAB, nRHS)

        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)

        # approximate at points
        # values : (nE, nEVAB, nRHS)
        points, rng = to_range(
            points, source=rng, target=[-1, 1]).flatten(), [-1, 1]

        dshp = self.shape_function_derivatives(points, rng=rng)[cells]
        jac = self.jacobian_matrix(
            dshp=dshp, ecoords=ecoords)  # (nE, nP, 1, 1)
        B = self.strain_displacement_matrix(
            dshp=dshp, jac=jac)  # (nE, nP, 4, nNODE * 6)

        values = ascont(swap(dofsol, 1, 2))  # (nE, nRHS, nEVAB)
        values = approx_element_solution_bulk(values, B)  # (nE, nRHS, nP, 4)
        D = self.model_stiffness_matrix()[cells]
        values = calculate_internal_forces_bulk(values, D)  # (nE, nRHS, nP, 4)
        values = ascont(np.moveaxis(values, 1, -1))   # (nE, nP, 4, nRHS)

        gdshp = self.shape_function_derivatives(
            points, rng=rng, jac=jac, dshp=dshp)
        nE, _, nRHS = dofsol.shape
        nNE, nDOF = self.__class__.NNODE, self.__class__.NDOFN
        # (nE, nEVAB, nRHS) -> (nE, nNE, nDOF=6, nRHS)
        dofsol = dofsol.reshape(nE, nNE, nDOF, nRHS)
        values = calculate_shear_forces(
            dofsol, values, D, gdshp)  # (nE, nP, nDOF=6, nRHS)

        if target is not None:
            # transform values to a destination frame, otherwise return
            # the results in the local frames of the cells
            assert target == 'local'

        if flatten:
            nE, nP, nSTRE, nRHS = values.shape
            values = values.reshape(nE, nP * nSTRE, nRHS)

        # values : (nE, nP, 4, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements
            data = values
        elif isinstance(cells, Iterable):
            data = {c: values[i] for i, c in enumerate(cells)}
        else:
            raise TypeError(
                "Invalid data type <> for cells.".format(type(cells)))

        return data
