# -*- coding: utf-8 -*-
from scipy.sparse import coo_matrix
import numpy as np
from collections import namedtuple, Iterable
from typing import Callable, Iterable

from dewloosh.core import squeeze, config

from dewloosh.math.linalg import ReferenceFrame
from dewloosh.math.array import atleast1d, atleastnd, \
    ascont, isboolarray, is1dfloatarray
from dewloosh.math.utils import to_range

from dewloosh.geom.celldata import CellData
from dewloosh.geom.utils import distribute_nodal_data, \
    collect_nodal_data

from .meta import FemMixin
from .preproc import fem_coeff_matrix_coo
from .postproc import approx_element_solution_bulk, calculate_internal_forces_bulk, \
    explode_kinetic_strains
from .utils import topo_to_gnum, approximation_matrix, nodal_approximation_matrix, \
    nodal_compatibility_factors, compatibility_factors_to_coo, \
    compatibility_factors, penalty_factor_matrix, assemble_load_vector
from .utils import tr_cells_1d_in_multi, tr_cells_1d_out_multi, element_dof_solution_bulk, \
    transform_stiffness, constrain_local_stiffness_bulk, assert_min_stiffness_bulk
from .cells.utils import stiffness_matrix_bulk2, strain_displacement_matrix_bulk2, \
    unit_strain_load_vector_bulk, strain_load_vector_bulk, mass_matrix_bulk

Quadrature = namedtuple('QuadratureRule', ['inds', 'pos', 'weight'])


UX, UY, UZ, ROTX, ROTY, ROTZ = FX, FY, FZ, MX, MY, MZ = list(range(6))
ulabels = {UX: 'UX', UY: 'UY', UZ: 'UZ',
           ROTX: 'ROTX', ROTY: 'ROTY', ROTZ: 'ROTZ'}
flabels = {FX: 'FX', FY: 'FY', FZ: 'FZ', MX: 'MX', MY: 'MY', MZ: 'MZ'}
ulabel_to_int = {v: k for k, v in ulabels.items()}
flabel_to_int = {v: k for k, v in flabels.items()}

fem_db_glossary = {
    'density': 'densities'
}


def integrate(fnc: Callable, quadrature, qrule, *args, qkey='gauss', **kwargs):

    def _eval(qinds, qvalue, *args, **kwargs):
        if isinstance(qvalue, str):
            # NOTE: this could be generalized. Now, the first referenece must be
            # a direct reference to a pure type. The container object should be
            # a Library instance, allowing for more complex definitions.
            qpos, qweight = quadrature[qvalue]
        else:
            qpos, qweight = qvalue
        kwargs[qkey] = Quadrature(qinds, qpos, qweight)
        return fnc(*args, **kwargs)

    q = quadrature[qrule]
    if isinstance(q, dict):
        # selective integration
        res = np.sum([_eval(qi, qv, *args, **kwargs)
                      for qi, qv in q.items()], axis=0)
    else:
        # uniform integration
        res = _eval(None, q, *args, **kwargs)
    return res


class FiniteElement(CellData, FemMixin):

    def __init__(self, *args, model=None, activity=None, densities=None,
                 loads=None, strain_loads=None, t=None, fields=None, **kwargs):
        
        activity = fields.pop('activity', activity)
        loads = fields.pop('loads', loads)
        strain_loads = fields.pop('strain_loads', strain_loads)
        densities = fields.pop('densities', densities)
        t = fields.pop('t', t)
        
        super().__init__(*args, fields=fields, **kwargs)
        
        if self.db is not None:
            topo = self.nodes.to_numpy()
            nE, nNE = topo.shape
            NDOFN = self.__class__.NDOFN
            NSTRE = self.__class__.NSTRE

            if activity is None:
                activity = np.ones(nE, dtype=bool)
            else:
                assert isboolarray(activity) and len(activity.shape) == 1, \
                    "'activity' must be a 1d boolean numpy array!"
            self.db['active'] = activity

            # densities
            if isinstance(densities, np.ndarray):
                assert len(densities.shape) == 1, \
                    "'densities' must be a 1d float or integer numpy array!"
                self.db['densities'] = densities
            else:
                if densities is not None and 'densities' not in fields:
                    raise TypeError(
                        "'densities' must be a 1d float or integer numpy array!")

            # body loads
            if loads is None:
                loads = np.zeros((nE, nNE, NDOFN, 1))
            if loads is not None:
                assert isinstance(loads, np.ndarray)
                if loads.shape[0] == nE and loads.shape[1] == nNE:
                    self.db['loads'] = loads
                elif loads.shape[0] == nE and loads.shape[1] == nNE * NDOFN:
                    loads = atleastnd(loads, 3, back=True)
                    self.db['loads'] = loads.reshape(
                        nE, nNE, NDOFN, loads.shape[-1])

            # strain loads
            if strain_loads is None:
                strain_loads = np.zeros((nE, NSTRE, 1))
            if strain_loads is not None:
                assert isinstance(strain_loads, np.ndarray)
                assert strain_loads.shape[0] == nE
                self.db['strain-loads'] = strain_loads

            if self.__class__.NDIM == 2:
                if t is None:
                    t = np.ones(nE, dtype=float)
                else:
                    if isinstance(t, float):
                        t = np.full(nE, t)
                    else:
                        assert is1dfloatarray(t), \
                            "'t' must be a 1d numpy array or a float!"
                self.db['t'] = t

        self._model = model

    def transform_coeff_matrix(self, K, *args, **kwargs):
        """
        Transforms element coefficient matrices (eg. the stiffness or the 
        mass matrix) from local to global.
        """
        dcm = self.direction_cosine_matrix()
        return K if dcm is None else transform_stiffness(K, dcm)

    @squeeze(True)
    def dof_solution(self, *args, target='local', cells=None, points=None,
                     rng=None, flatten=True, **kwargs):
        """
        Returns nodal displacements for the cells, wrt. their local frames.

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

        target : frame, Optional.
            Reference frame for the output. A value of None or 'local' refers to the local system
            of the cells. Default is 'local'.

        ---
        Returns
        -------
        numpy array or dictionary
        (nE, nEVAB, nRHS) if flatten else (nE, nNE, nDOF, nRHS) 

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

        # transform values to cell-local frames
        values = element_dof_solution_bulk(dofsol, gnum)  # (nE, nEVAB, nRHS)
        values = ascont(np.swapaxes(values, 1, 2))
        values = tr_cells_1d_in_multi(values, dcm)
        values = ascont(np.swapaxes(values, 1, 2))

        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)

        # approximate at points
        # values : (nE, nEVAB, nRHS)
        points, rng = to_range(
            points, source=rng, target=[-1, 1]).flatten(), [-1, 1]
        N = self.shape_function_matrix(points, rng=rng)[
            cells]  # (nE, nP, nDOF, nDOF * nNODE)
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = approx_element_solution_bulk(
            values, N)  # (nE, nRHS, nP, nDOF)

        if target is not None:
            # transform values to a destination frame, otherwise return
            # the results in the local frames of the cells
            if isinstance(target, str):
                if target == 'local':
                    pass
                elif target == 'global':
                    nDOF = values.shape[2]
                    target = ReferenceFrame(dim=nDOF)
            if isinstance(target, ReferenceFrame):
                nE, nRHS, nP, nDOF = values.shape
                values = values.reshape(nE, nRHS, nP * nDOF)
                dcm = self.direction_cosine_matrix(N=nP)[cells]
                values = tr_cells_1d_out_multi(values, dcm)
                values = values.reshape(nE, nRHS, nP, nDOF)

        values = np.moveaxis(values, 1, -1)   # (nE, nP, nDOF, nRHS)

        if flatten:
            nE, nP, nDOF, nRHS = values.shape
            values = values.reshape(nE, nP * nDOF, nRHS)

        # values : (nE, nP, nDOF, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements
            data = values
        elif isinstance(cells, Iterable):
            data = {c: values[i] for i, c in enumerate(cells)}
        else:
            raise TypeError(
                "Invalid data type <> for cells.".format(type(cells)))

        return data

    @squeeze(True)
    def strains(self, *args, cells=None, points=None, rng=None, separate=False,
                source='total', **kwargs):
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
        assert separate is False  # separate heat and force sources

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

        # transform values to cell-local frames
        values = element_dof_solution_bulk(dofsol, gnum)  # (nE, nEVAB, nRHS)
        values = ascont(np.swapaxes(values, 1, 2))
        values = tr_cells_1d_in_multi(values, dcm)
        values = ascont(np.swapaxes(values, 1, 2))

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

        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = approx_element_solution_bulk(values, B)  # (nE, nRHS, nP, 4)
        values = ascont(np.moveaxis(values, 1, -1))   # (nE, nP, nDOF, nRHS)

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

    @squeeze(True)
    def kinetic_strains(self, *args, cells=None, points=None, **kwargs):
        """
        Returns kinetic strains.

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

        Returns
        -------
        numpy array of shape (nE, nSTRE, nRHS)
        """
        try:
            sloads = self.db['strain-loads'].to_numpy()
        except Exception as e:
            if 'strain-loads' not in self.db.fields:
                nE = len(self)
                nRHS = self.pointdata.loads.to_numpy().shape[-1]
                nSTRE = self.__class__.NSTRE
                sloads = np.zeros((nE, nSTRE, nRHS))
            else:
                raise e
        finally:
            sloads = atleastnd(sloads, 3, back=True)  # (nE, nSTRE=4, nRHS)
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id.to_numpy())
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
        else:
            cells = np.s_[:]
        if isinstance(points, Iterable):
            nP = len(points)
            return explode_kinetic_strains(sloads[cells], nP)
        else:
            return sloads[cells]

    def _postproc_local_internal_forces(self, forces, *args, points, rng, cells, **kwargs):
        """
        The aim of this function os to guarantee a standard shape of output, that contains
        values for each of the 6 internal force compoents, irrespective of the kinematical
        model being used.

        Example use case : the Bernoulli beam element, where, as a consequence of the absence
        of shear forces, there are only 4 internal force components, and the shear forces
        must be calculated a-posteriori.

        Parameters
        ----------
        forces : numpy array of shape (nE, nP, nSTRE, nRHS)
            4d float array of internal forces for many elements, evaluation points,
            and load cases. The number of force components (nSTRE) is dictated by the
            reduced kinematical model (eg. 4 for the Bernoulli beam).

        dofsol (nE, nEVAB, nRHS)

        Returns
        -------
        numpy array of shape (nE, nP, 6, nRHS)

        Notes
        -----
        Arguments are based on the reimplementation in the Bernoulli base element.
        """
        return forces

    @squeeze(True)
    def internal_forces(self, *args, cells=None, points=None, rng=None,
                        flatten=True, target='local', **kwargs):
        """
        Returns internal forces for many cells and evaluation points.

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
            if isinstance(points, Iterable):
                points = np.array(points)
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

        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nRHS, nEVAB)
        strains = approx_element_solution_bulk(dofsol, B)  # (nE, nRHS, nP, 4)
        strains -= self.kinetic_strains(points=points, squeeze=False)[cells]
        D = self.model_stiffness_matrix()[cells]
        forces = calculate_internal_forces_bulk(
            strains, D)  # (nE, nRHS, nP, 4)
        forces = ascont(np.moveaxis(forces, 1, -1))   # (nE, nP, 4, nRHS)
        dofsol = ascont(np.swapaxes(dofsol, 1, 2))  # (nE, nEVAB, nRHS)
        forces = self._postproc_local_internal_forces(forces, points=points, rng=rng,
                                                      cells=cells, dofsol=dofsol)

        if target is not None:
            # transform values to a destination frame, otherwise return
            # the results in the local frames of the cells
            assert target == 'local'

        if flatten:
            nE, nP, nSTRE, nRHS = forces.shape
            forces = forces.reshape(nE, nP * nSTRE, nRHS)

        # values : (nE, nP, 4, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements
            data = forces
        elif isinstance(cells, Iterable):
            data = {c: forces[i] for i, c in enumerate(cells)}
        else:
            raise TypeError(
                "Invalid data type <> for cells.".format(type(cells)))

        return data

    def global_dof_numbering(self, *args, topo=None, **kwargs):
        topo = self.nodes.to_numpy() if topo is None else topo
        return topo_to_gnum(topo, self.NDOFN)

    def approximation_matrix(self, *args, **kwargs):
        return approximation_matrix(self.ndf.to_numpy(), self.NDOFN)

    def approximation_matrix_coo(self, *args, **kwargs):
        N = len(self.pointdata) * self.NDOFN
        topo = self.nodes.to_numpy()
        appr = self.approximation_matrix()
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(appr, *args, inds=gnum, N=N, **kwargs)

    def nodal_approximation_matrix(self, *args, **kwargs):
        return nodal_approximation_matrix(self.ndf.to_numpy())

    def nodal_approximation_matrix_coo(self, *args, **kwargs):
        N = len(self.pointdata)
        topo = self.nodes.to_numpy()
        appr = self.nodal_approximation_matrix()
        return fem_coeff_matrix_coo(appr, *args, inds=topo, N=N, **kwargs)

    def distribute_nodal_data(self, data, key):
        topo = self.nodes.to_numpy()
        ndf = self.ndf.to_numpy()
        self.db[key] = distribute_nodal_data(data, topo, ndf)

    def collect_nodal_data(self, key, *args, N=None, **kwargs):
        topo = self.nodes.to_numpy()
        N = len(self.pointdata) if N is None else N
        cellloads = self.db[key].to_numpy()
        return collect_nodal_data(cellloads, topo, N)

    def compatibility_penalty_matrix_coo(self, *args, nam_csr_tot, p=1e12, **kwargs):
        """
        Parameters
        ----------
        nam_csr_tot : nodal_approximation matrix for the whole structure
                      in csr format.  
        """
        topo = self.nodes.to_numpy()
        nam = self.nodal_approximation_matrix()
        factors, nreg = nodal_compatibility_factors(nam_csr_tot, nam, topo)
        factors, nreg = compatibility_factors(factors, nreg, self.NDOFN)
        data, rows, cols = compatibility_factors_to_coo(factors, nreg)
        return coo_matrix((data*p, (rows, cols)))

    def penalty_factor_matrix(self, *args, **kwargs):
        cellfixity = self.fixity.to_numpy()
        shp = self.shape_function_values(self.lcoords())
        return penalty_factor_matrix(cellfixity, shp)

    def penalty_matrix(self, *args, p=1e12, **kwargs):
        return self.penalty_factor_matrix() * p

    def penalty_matrix_coo(self, *args, p=1e12, **kwargs):
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.nodes.to_numpy()
        K_bulk = self.penalty_matrix(*args, topo=topo, p=p, **kwargs)
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(K_bulk, *args, inds=gnum, N=N, **kwargs)

    @squeeze(True)
    def stiffness_matrix(self, *args, **kwargs):
        K = self.stiffness_matrix_v2(*args, transform=False, **kwargs)
        if 'conn' in self.db.fields:
            # only for line meshes at the moment
            conn = self.db.conn.to_numpy()
            if len(conn.shape) == 3 and conn.shape[1] == 2:  # nE, 2, nDOF
                nE, _, nDOF = conn.shape
                #conn = np.reshape(conn, (nE, 2*nDOF))
                factors = np.ones((nE, K.shape[-1]))
                factors[:, :nDOF] = conn[:, 0, :]
                factors[:, -nDOF:] = conn[:, -1, :]
                nEVAB2 = 2*nDOF - 0.5
                cond = np.sum(factors, axis=1) < nEVAB2
                i = np.where(cond)[0]
                K[i] = constrain_local_stiffness_bulk(K[i], factors[i])
                assert_min_stiffness_bulk(K)
            else:
                raise NotImplementedError(
                    "Unknown shape of <{}> for 'connectivity'.".format(conn.shape))
        self.db['K'] = K
        return self.transform_coeff_matrix(K)

    @config(store_strains=False)
    def stiffness_matrix_v1(self, *args, **kwargs):
        _topo = kwargs.get('_topo', self.db.nodes.to_numpy())
        if 'frames' not in kwargs:
            if 'frames' in self.db.fields:
                frames = self.db.frames.to_numpy()
            else:
                frames = None
        _ecoords = kwargs.get('_ecoords', None)
        if _ecoords is None:
            if frames is not None:
                _ecoords = self.local_coordinates(_topo=_topo, frames=frames)
            else:
                _ecoords = self.points_of_cells(topo=_topo)

        q = kwargs.get('_q', None)
        if q is None:
            # main loop
            q = self.quadrature[self.qrule]
            if isinstance(q, dict):
                # many side loops
                N = self.NNODE * self.NDOFN
                res = np.zeros((len(self), N, N))
                for qinds, qvalue in q.items():
                    if isinstance(qvalue, str):
                        qpos, qweight = self.quadrature[qvalue]
                    else:
                        qpos, qweight = qvalue
                    q = Quadrature(qinds, qpos, qweight)
                    res[:, :, :] += self.stiffness_matrix_v1(*args, _topo=_topo,
                                                             frames=frames, _q=q,
                                                             _ecoords=_ecoords,
                                                             **kwargs)
            else:
                # one side loop
                qpos, qweight = self.quadrature[self.qrule]
                q = Quadrature(None, qpos, qweight)
                res = self.stiffness_matrix_v1(*args, _topo=_topo,
                                               frames=frames, _q=q,
                                               _ecoords=_ecoords,
                                               **kwargs)
            # end of main cycle
            self.db['K'] = res
            return self.transform_coeff_matrix(res)

        # in side loop
        dshp = self.shape_function_derivatives(q.pos)
        jac = self.jacobian_matrix(dshp=dshp, ecoords=_ecoords)
        B = self.strain_displacement_matrix(dshp=dshp, jac=jac)
        if q.inds is not None:
            # zero out unused indices, only for selective integration
            inds = np.where(~np.in1d(np.arange(self.NSTRE), q.inds))[0]
            B[:, :, inds, :] = 0.
        djac = self.jacobian(jac=jac)
        D = self.model_stiffness_matrix()

        if kwargs.get('store_strains', False):
            raise NotImplementedError()

        return stiffness_matrix_bulk2(D, B, djac, q.weight)

    @config(store_strains=True)
    def stiffness_matrix_v2(self, *args, transform=True, **kwargs):
        nSTRE = self.__class__.NSTRE
        nDOF = self.__class__.NDOFN
        nNE = self.__class__.NNODE
        nE = len(self)
        _gauss_strains_ = None

        def _stiffness_matrix_(*args, gauss, ecoords, D, **kwargs):
            nonlocal _gauss_strains_
            dshp = self.shape_function_derivatives(gauss.pos)
            jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)
            B = self.strain_displacement_matrix(dshp=dshp, jac=jac)
            if gauss.inds is not None:
                # zero out unused indices, only for selective integration
                missing_indices = np.where(
                    ~np.in1d(np.arange(nSTRE), gauss.inds))[0]
                B[:, :, missing_indices, :] = 0.
            djac = self.jacobian(jac=jac)
            if kwargs.get('store_strains', False):
                # B (nE, nG, nSTRE=4, nNODE * nDOF=6)
                _B = strain_displacement_matrix_bulk2(B, djac, gauss.weight)
                """if gauss.inds is None:
                    qinds = tuple(range(nSTRE))
                else:
                    qinds = gauss.inds
                self._gauss_strains_[qinds] = _B"""
                _gauss_strains_ += _B
            return stiffness_matrix_bulk2(D, B, djac, gauss.weight)

        if kwargs.get('store_strains', False):
            #_gauss_strains_ = {}
            _gauss_strains_ = np.zeros((nE, nSTRE, nDOF * nNE))

        Hooke = self.model_stiffness_matrix()
        topo = self.nodes.to_numpy()
        frames = self.db.frames.to_numpy()
        ecoords = self.local_coordinates(_topo=topo, frames=frames)
        kwargs['ecoords'] = ecoords
        kwargs['D'] = Hooke
        K = integrate(_stiffness_matrix_, self.quadrature,
                      self.qrule, *args, qkey='gauss', **kwargs)
        self.db['K'] = K
        if kwargs.get('store_strains', False):
            self.db['B'] = _gauss_strains_

        return self.transform_coeff_matrix(K) if transform else K

    def stiffness_matrix_coo(self, *args, **kwargs):
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.nodes.to_numpy()
        K_bulk = self.stiffness_matrix(*args, _topo=topo, **kwargs)
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(K_bulk, *args, inds=gnum, N=N, **kwargs)

    @squeeze(True)
    def mass_matrix(self, *args, **kwargs):
        return self.mass_matrix_v1(*args, **kwargs)

    def mass_matrix_v1(self, *args, values=None, **kwargs):
        _topo = kwargs.get('_topo', self.db.nodes.to_numpy())
        if 'frames' not in kwargs:
            if 'frames' in self.db.fields:
                frames = self.db.frames.to_numpy()
            else:
                frames = None
        _ecoords = kwargs.get('_ecoords', None)
        if _ecoords is None:
            if frames is not None:
                _ecoords = self.local_coordinates(_topo=_topo, frames=frames)
            else:
                _ecoords = self.points_of_cells(topo=_topo)
        if isinstance(values, np.ndarray):
            _dens = values
        else:
            _dens = kwargs.get('_dens', self.db.densities.to_numpy())

        try:
            _areas = self.areas()
        except Exception:
            _areas = np.ones_like(_dens)

        """if 'areas' in self.db.fields:
            _areas = kwargs.get('_areas', self.db.areas.to_numpy())
        else:
            _areas = np.ones_like(_dens)"""

        q = kwargs.get('_q', None)
        if q is None:
            q = self.quadrature['mass']
            if isinstance(q, dict):
                N = self.NNODE * self.NDOFN
                nE = len(self)
                res = np.zeros((nE, N, N))
                for qinds, qvalue in q.items():
                    if isinstance(qvalue, str):
                        qpos, qweight = self.quadrature[qvalue]
                    else:
                        qpos, qweight = qvalue
                    q = Quadrature(qinds, qpos, qweight)
                    res += self.mass_matrix_v1(*args, _topo=_topo,
                                               frames=frames, _q=q,
                                               values=_dens,
                                               _ecoords=_ecoords,
                                               **kwargs)
            else:
                qpos, qweight = self.quadrature['mass']
                q = Quadrature(None, qpos, qweight)
                res = self.mass_matrix_v1(*args, _topo=_topo,
                                          frames=frames, _q=q,
                                          values=_dens,
                                          _ecoords=_ecoords,
                                          **kwargs)
            self.db['M'] = res
            return self.transform_coeff_matrix(res)

        rng = np.array([-1., 1.])
        dshp = self.shape_function_derivatives(q.pos)
        jac = self.jacobian_matrix(dshp=dshp, ecoords=_ecoords)
        djac = self.jacobian(jac=jac)
        N = self.shape_function_matrix(q.pos, rng=rng)
        return mass_matrix_bulk(N, _dens, _areas, djac, q.weight)

    def mass_matrix_coo(self, *args, **kwargs):
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.db.nodes.to_numpy()
        M_bulk = self.mass_matrix(*args, **kwargs)
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(M_bulk, *args, inds=gnum, N=N, **kwargs)

    @squeeze(True)
    def strain_load_vector(self, values=None, *args, squeeze, **kwargs):
        nRHS = self.pointdata.loads.to_numpy().shape[-1]
        nSTRE = self.__class__.NSTRE
        try:
            if values is None:
                values = self.db['strain-loads'].to_numpy()
        except Exception as e:
            if 'strain-loads' not in self.db.fields:
                nE = len(self)
                values = np.zeros((nE, nSTRE, nRHS))
            else:
                raise e
        finally:
            values = atleastnd(values, 3, back=True)  # (nE, nSTRE=4, nRHS)

        if 'B' not in self.db.fields:
            self.stiffness_matrix(*args, squeeze=False,
                                  store_strains=True, **kwargs)
        B = self.db['B'].to_numpy()  # (nE, nSTRE=4, nNODE * nDOF=6)
        D = self.model_stiffness_matrix()  # (nE, nSTRE=4, nSTRE=4)
        BTD = unit_strain_load_vector_bulk(D, B)
        values = np.swapaxes(values, 1, 2)  # (nE, nRHS, nSTRE=4)
        nodal_loads = strain_load_vector_bulk(BTD, ascont(values))
        return transform_local_nodal_loads(self, nodal_loads)

    @squeeze(True)
    def body_load_vector(self, values=None, *args, source=None,
                         constant=False, **kwargs):
        nNE = self.__class__.NNODE
        # prepare data to shape (nE, nNE * nDOF, nRHS)
        if values is None:
            values = self.loads.to_numpy()
            values = atleastnd(values, 4, back=True)  # (nE, nNE, nDOF, nRHS)
        else:
            if constant:
                values = atleastnd(values, 3, back=True)  # (nE, nDOF, nRHS)
                #np.insert(values, 1, values, axis=1)
                nE, nDOF, nRHS = values.shape
                values_ = np.zeros((nE, nNE, nDOF, nRHS), dtype=values.dtype)
                for i in range(nNE):
                    values_[:, i, :, :] = values
                values = values_
            values = atleastnd(values, 4, back=True)  # (nE, nNE, nDOF, nRHS)
        nE, _, nDOF, nRHS = values.shape
        # (nE, nNE, nDOF, nRHS) -> (nE, nNE * nDOF, nRHS)
        values = values.reshape(nE, nNE * nDOF, nRHS)
        values = ascont(values)

        if source is not None:
            raise NotImplementedError
            nE, nNE, nDOF, nRHS = values.shape
            dcm = self.direction_cosine_matrix(source=source)
            # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
            values = np.swapaxes(values, 1, 2)
            values = ascont(values)
            values = tr_cells_1d_in_multi(values, dcm)
            # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)
            values = np.swapaxes(values, 1, 2)
            values = ascont(values)

        nodal_loads = self.integrate_body_loads(values)
        return transform_local_nodal_loads(self, nodal_loads)


def transform_local_nodal_loads(obj: FiniteElement, nodal_loads):
    """
    nodal_loads (nE, nNE * nDOF, nRHS) == (nE, nX, nRHS)
    ---
    (nX, nRHS)
    """
    dcm = obj.direction_cosine_matrix()
    # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
    nodal_loads = np.swapaxes(nodal_loads, 1, 2)
    nodal_loads = ascont(nodal_loads)
    nodal_loads = tr_cells_1d_out_multi(nodal_loads, dcm)
    # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)
    nodal_loads = np.swapaxes(nodal_loads, 1, 2)
    # assemble
    topo = obj.db.nodes.to_numpy()
    gnum = obj.global_dof_numbering(topo=topo)
    nX = len(obj.pointdata) * obj.__class__.NDOFN
    return assemble_load_vector(nodal_loads, gnum, nX)  # (nX, nRHS)
