# -*- coding: utf-8 -*-
from typing import Collection
import numpy as np

from dewloosh.math import squeeze
from dewloosh.math.array import atleast3d

from dewloosh.mesh import PolyData

from .pointdata import PointData
from .cells.celldata import CellData
from .cells.meta import ABCFiniteElement
from .preproc import fem_load_vector, fem_penalty_matrix_coo, \
    fem_nodal_mass_matrix_coo


class FemMesh(PolyData):

    dofs = ('UX', 'UY', 'UZ', 'ROTX', 'ROTY', 'ROTZ')
    NDOFN = 6
    _point_class_ = PointData
    
    def __init__(self, *args, model=None, fixity=None, loads=None, body_loads=None,
                 strain_loads=None, t=None, density=None, mass=None,
                 cell_fields=None, point_fields=None, activity=None, 
                 connectivity=None, **kwargs):
        # fill up data objects with obvious data
        pkeys = self.__class__._point_class_._attr_map_
        point_fields = {} if point_fields is None else point_fields
        point_fields[pkeys['loads']] = loads
        point_fields[pkeys['mass']] = mass
        point_fields[pkeys['fixity']] = fixity
        point_fields[pkeys['loads']] = loads
        ckeys = CellData._attr_map_
        cell_fields = {} if cell_fields is None else cell_fields
        cell_fields[ckeys['loads']] = body_loads
        cell_fields[ckeys['strain-loads']] = strain_loads
        cell_fields[ckeys['density']] = density
        cell_fields[ckeys['t']] = t
        cell_fields[ckeys['connectivity']] = connectivity
        super().__init__(*args, point_fields=point_fields,
                         cell_fields=cell_fields, **kwargs)
        # nodal data can only be provided for the root object
        if not self.is_root():
            assert loads is None, "At object creation, nodal loads can only \
                be provided at the top level."
            assert fixity is None, "At object creation, fixity information \
                can only be provided at the top level."
            assert mass is None, "At object creation, nodal masses can only \
                be provided at the top level."
        # it is determined by the size of `activity` whether it refers to
        # cells or points
        if isinstance(activity, np.ndarray):
            nA = activity.shape[0]
            if self.celldata is not None:
                N = len(self.celldata)
                if nA == N:
                    self.celldata.activity = activity
                    nA = -1
            if nA > 0 and self.pointdata is not None:
                N = len(self.pointdata)
                if nA == N:
                    self.pointdata.activity = activity
                    nA = -1
            assert nA < 0
        self._model = model

    def cells_coords(self, *args, points=None, cells=None, **kwargs):
        if points is None and cells is None:
            return super().cells_coords(*args, **kwargs)
        else:
            blocks = self.cellblocks(inclusive=True)
            kwargs.update(points=points, squeeze=False)
            if cells is not None:
                kwargs.update(cells=cells)
                res = {}
                def foo(b): return b.celldata.coords(*args, **kwargs)
                [res.update(d) for d in map(foo, blocks)]
                return res
            else:
                def foo(b): return b.celldata.coords(*args, **kwargs)
                return np.vstack(list(map(foo, blocks)))

    def element_dof_numbering(self, *args, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.global_dof_numbering()
        return np.vstack(list(map(foo, blocks)))

    def stiffness_matrix(self, *args, sparse=True, **kwargs):
        """Elastic stiffness matrix in dense format."""
        if sparse:
            return self.stiffness_matrix_coo(*args, **kwargs)
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.stiffness_matrix()
        return np.vstack(list(map(foo, blocks)))

    def stiffness_matrix_coo(self, *args, eliminate_zeros=True,
                             sum_duplicates=True, **kwargs):
        """Elastic stiffness matrix in coo format."""
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.stiffness_matrix_coo()
        K = np.sum(list(map(foo, blocks))).tocoo()
        if eliminate_zeros:
            K.eliminate_zeros()
        if sum_duplicates:
            K.sum_duplicates()
        return K
    
    def masses(self, *args, **kwargs) -> np.ndarray:
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        vmap = map(lambda b: b.celldata.masses(), blocks)
        return np.concatenate(list(vmap))
    
    def mass(self, *args, **kwargs) -> float:
        return np.sum(self.masses(*args, **kwargs))

    def mass_matrix(self, *args, sparse=True, **kwargs):
        """
        Returns the mass matrix of the mesh with either dense 
        or sparse layout.

        Notes
        -----
        If there are nodal masses defined, only sparse output is 
        available at the moment.

        """
        if sparse:
            return self.mass_matrix_coo(*args, **kwargs)
        else:
            dbkey = self.__class__._point_class_._attr_map_['mass']
            if dbkey in self.root().pointdata.fields:
                return self.mass_matrix_coo(*args, **kwargs)
        # distributed masses (cells)
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.mass_matrix()
        return np.vstack(list(map(foo, blocks)))

    def mass_matrix_coo(self, *args, eliminate_zeros=True,
                        sum_duplicates=True, distribute=False, **kwargs):
        """
        Returns the mass matrix in coo format. If `distribute` is set 
        to `True`, nodal masses are distributed over neighbouring cells 
        and handled as self-weight is.

        """
        # distributed masses (cells)
        blocks = list(self.cellblocks(inclusive=True))
        def foo(b): return b.celldata.mass_matrix_coo()
        M = np.sum(list(map(foo, blocks))).tocoo()
        # nodal masses
        pd = self.root().pointdata
        dbkey = self.__class__._point_class_._attr_map_['mass']
        if dbkey in pd.fields:
            d = pd.mass()
            if distribute:
                v = self.volumes()
                edata = list(
                    map(lambda b: b.celldata.pull(data=d, avg=v), blocks))

                def foo(bv): return bv[0].celldata.mass_matrix_coo(
                    values=bv[1])
                moo = map(lambda i: (blocks[i], edata[i]), range(len(blocks)))
                M += np.sum(list(map(foo, moo))).tocoo()
            else:
                ndof = self.__class__.NDOFN
                M += fem_nodal_mass_matrix_coo(values=d, eliminate_zeros=eliminate_zeros,
                                               sum_duplicates=sum_duplicates, ndof=ndof)
        if eliminate_zeros:
            M.eliminate_zeros()
        if sum_duplicates:
            M.sum_duplicates()
        return M

    def penalty_matrix_coo(self, *args, eliminate_zeros=True,
                           sum_duplicates=True, ensure_comp=False,
                           distribute=False, **kwargs):
        """
        A penalty matrix that enforces Dirichlet boundary conditions. 
        Returns a scipy sparse matrix in coo format.
        """
        fixity = self.root().pointdata.fixity
        K_coo = fem_penalty_matrix_coo(values=fixity, eliminate_zeros=eliminate_zeros,
                                       sum_duplicates=sum_duplicates)
        return K_coo.tocoo()

    def approximation_matrix_coo(self, *args, eliminate_zeros=True, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.approximation_matrix_coo()
        res = np.sum(list(map(foo, blocks))).tocoo()
        if eliminate_zeros:
            res.eliminate_zeros()
        return res

    def nodal_approximation_matrix_coo(self, *args, eliminate_zeros=True, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.nodal_approximation_matrix_coo()
        res = np.sum(list(map(foo, blocks))).tocoo()
        if eliminate_zeros:
            res.eliminate_zeros()
        return res

    @squeeze(True)
    def load_vector(self, *args, **kwargs):
        # concentrated nodal loads
        nodal_data = self.root().pointdata.loads
        nodal_data = atleast3d(nodal_data, back=True)  # (nP, nDOF, nRHS)
        res = fem_load_vector(values=nodal_data, squeeze=False)
        # cells
        blocks = list(self.cellblocks(inclusive=True))
        # body loads
        def foo(b): return b.celldata.body_load_vector(squeeze=False)
        try:
            res += np.sum(list(map(foo, blocks)), axis=0)
        except Exception:
            pass
        # strain loads
        def foo(b): return b.celldata.strain_load_vector(squeeze=False)
        try:
            res += np.sum(list(map(foo, blocks)), axis=0)
        except Exception:
            pass
        return res

    def prostproc_dof_solution(self, *args, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        dofsol = self.root().pointdata.dofsol
        def foo(b): return b.celldata.prostproc_dof_solution(dofsol=dofsol)
        list(map(foo, blocks))

    @squeeze(True)
    def nodal_dof_solution(self, *args, flatten=False, **kwargs):
        dofsol = self.root().pointdata.dofsol
        if flatten:
            if len(dofsol.shape) == 2:
                return dofsol.flatten()
            else:
                nN, nDOFN, nRHS = dofsol.shape
                return dofsol.reshape((nN * nDOFN, nRHS))
        else:
            return dofsol

    @squeeze(True)
    def cell_dof_solution(self, *args, cells=None, flatten=True,
                          squeeze=True, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        kwargs.update(flatten=flatten, squeeze=False)
        if cells is not None:
            kwargs.update(cells=cells)
            res = {}
            def foo(b): return b.celldata.dof_solution(*args, **kwargs)
            [res.update(d) for d in map(foo, blocks)]
            return res
        else:
            def foo(b): return b.celldata.dof_solution(*args, **kwargs)
            return np.vstack(list(map(foo, blocks)))

    @squeeze(True)
    def strains(self, *args, cells=None, squeeze=True, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        kwargs.update(squeeze=False)
        if cells is not None:
            kwargs.update(cells=cells)
            res = {}
            def foo(b): return b.celldata.strains(*args, **kwargs)
            [res.update(d) for d in map(foo, blocks)]
            return res
        else:
            def foo(b): return b.celldata.strains(*args, **kwargs)
            return np.vstack(list(map(foo, blocks)))

    @squeeze(True)
    def internal_forces(self, *args, cells=None, flatten=True, squeeze=True, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        kwargs.update(flatten=flatten, squeeze=False)
        if cells is not None:
            kwargs.update(cells=cells)
            res = {}
            def foo(b): return b.celldata.internal_forces(*args, **kwargs)
            [res.update(d) for d in map(foo, blocks)]
            return res
        else:
            def foo(b): return b.celldata.internal_forces(*args, **kwargs)
            return np.vstack(list(map(foo, blocks)))

    @squeeze(True)
    def reaction_forces(self, *args, flatten=False, squeeze=True, **kwargs):
        x = self.root().pointdata.reactions
        if flatten:
            if len(x.shape) == 2:
                return x.flatten()
            else:
                nN, nDOFN, nRHS = x.shape
                return x.reshape((nN * nDOFN, nRHS))
        else:
            return x

    @squeeze(True)
    def nodal_forces(self, *args, flatten=False, **kwargs):
        x = self.root().pointdata.forces
        if flatten:
            if len(x.shape) == 2:
                return x.flatten()
            else:
                nN, nDOFN, nRHS = x.shape
                return x.reshape((nN * nDOFN, nRHS))
        else:
            return x

    def stresses_at_cells_nodes(self, *args, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.stresses_at_nodes(*args, **kwargs)
        return np.vstack(list(map(foo, blocks)))

    def stresses_at_centers(self, *args, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.stresses_at_centers(*args, **kwargs)
        return np.squeeze(np.vstack(list(map(foo, blocks))))

    @property
    def model(self):
        if self._model is not None:
            return self._model
        else:
            if self.is_root():
                return self._model
            else:
                return self.parent.model

    def material_stiffness_matrix(self):
        m = self.model
        if isinstance(m, np.ndarray):
            return m
        else:
            try:
                return m.stiffness_matrix()
            except Exception:
                raise RuntimeError("Invalid model type {}".format(type(m)))

    def postprocess(self, *args, **kwargs):
        pass
