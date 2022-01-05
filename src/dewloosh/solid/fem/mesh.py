# -*- coding: utf-8 -*-
from dewloosh.geom.polydata import PolyData
from dewloosh.math.array import isboolarray, is1dfloatarray
from dewloosh.solid.fem.preproc import fem_load_vector, fem_penalty_matrix_coo
from dewloosh.math.linalg.sparse.csr import csr_matrix as csr
import numpy as np


class FemMesh(PolyData):

    NDOFN = 3

    def __init__(self, *args,  model=None, fixity=None, activity=None,
                 loads=None, t=None, **kwargs):
        # parent class handles pointdata and celldata creation
        super().__init__(*args, **kwargs)

        # initialize activity information, default is active (True)
        if self.celldata is not None:
            nE = len(self.celldata)
            if activity is None:
                activity = np.ones(nE, dtype=bool)
            else:
                assert isboolarray(activity) and len(activity.shape) == 1, \
                    "'activity' must be a 1d boolean numpy array!"
            self.celldata._wrapped['active'] = activity

            if self.celldata.NDIM == 2:
                if t is None:
                    t = np.ones(nE, dtype=float)
                else:
                    if isinstance(t, float):
                        t = np.full(nE, t)
                    else:
                        assert is1dfloatarray(t), \
                            "'t' must be a 1d numpy array or a float!"
                self.celldata._wrapped['t'] = t
            self.NDOFN = self.celldata.NDOFN

        # initialize boundary conditions
        if self.is_root():
            # initialize essential boundary conditions, default is free (False)
            if fixity is None and self.pointdata is not None:
                fixity = np.zeros(
                    (len(self.pointdata), self.NDOFN), dtype=bool)
            elif fixity is not None:
                assert isinstance(fixity, np.ndarray) and len(
                    fixity.shape) == 2
                self.pointdata['fixity'] = fixity
            # initialize natural boundary conditions
            if loads is None and self.pointdata is not None:
                loads = np.zeros((len(self.pointdata), self.NDOFN))
            elif loads is not None:
                assert isinstance(loads, np.ndarray) and len(loads.shape) == 2
                self.pointdata['loads'] = loads
        else:
            assert loads is None, "At object creation, nodal loads can only \
                be provided at the top level."
            assert fixity is None, "At object creation, fixity information \
                can only be provided at the top level."

        # material model
        self._model = model

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

    def stiffness_matrix(self, *args, sparse=True, **kwargs):
        """Elastic stiffness matrix in dense format."""
        if sparse:
            return self.stiffness_matrix_coo(*args, **kwargs)
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.stiffness_matrix()
        return np.vstack(list(map(foo, blocks)))

    def penalty_matrix_coo(self, *args, eliminate_zeros=True,
                           sum_duplicates=True, ensure_comp=False, **kwargs):
        """A penalty matrix that enforces essential(Dirichlet) 
        boundary conditions. Returns a scipy sparse matrix in coo format."""
        # essential boundary conditions
        fixity = self.root().pointdata.fixity.to_numpy()
        K_coo = fem_penalty_matrix_coo(pen=fixity)
        """
        # distribute nodal fixity
        fixity = self.root().pointdata.fixity.to_numpy().astype(int)
        blocks = list(self.cellblocks(inclusive=True))
        def foo(b): return b.celldata.distribute_nodal_data(fixity, 'fixity')
        list(map(foo, blocks))
        # assemble
        def foo(b): return b.celldata.penalty_matrix_coo()
        K_coo = np.sum(list(map(foo, blocks))).tocoo()
        """
        if ensure_comp and not self.is_compatible():
            # penalty arising from incompatibility
            p = kwargs.get('compatibility_penalty', None)
            if p is not None:
                K_coo += self.compatibility_penalty_matrix_coo(eliminate_zeros=False, p=p)
        if eliminate_zeros:
            K_coo.eliminate_zeros()
        if sum_duplicates:
            K_coo.sum_duplicates()
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
    
    def compatibility_penalty_matrix_coo(self, *args, eliminate_zeros=True, p=1e12, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        nam_csr_tot = csr(self.nodal_approximation_matrix_coo(eliminate_zeros=False))
        def foo(b): return b.celldata.compatibility_penalty_matrix_coo(nam_csr_tot=nam_csr_tot, p=p)
        res = np.sum(list(map(foo, blocks))).tocoo()
        if eliminate_zeros:
            res.eliminate_zeros()
        return res

    def load_vector(self, *args, **kwargs):
        loads = self.root().pointdata.loads.to_numpy()
        """
        if not self.is_compatible():
            # distribute nodal loads
            blocks = list(self.cellblocks(inclusive=True))
            def foo(b): return b.celldata.distribute_nodal_data(loads, 'loads')
            list(map(foo, blocks))
            # collect nodal loads
            N = len(loads)
            def foo(b): return b.celldata.collect_nodal_data('loads', N=N)
            loads = np.sum(list(map(foo, blocks)), axis=0)
        """
        return fem_load_vector(vals=loads)
    
    def prostproc_dof_solution(self, *args, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        dofsol = self.root().pointdata.dofsol.to_numpy()
        def foo(b): return b.celldata.prostproc_dof_solution(dofsol=dofsol)
        list(map(foo, blocks))

    def stresses_at_cells_nodes(self, *args, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.stresses_at_nodes(*args, **kwargs)
        return np.vstack(list(map(foo, blocks)))

    def stresses_at_centers(self, *args, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.stresses_at_centers(*args, **kwargs)
        return np.squeeze(np.vstack(list(map(foo, blocks))))

    def element_dof_numbering(self, *args, **kwargs):
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.global_dof_numbering()
        return np.vstack(list(map(foo, blocks)))

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

    def is_compatible(self):
        blocks = self.cellblocks(inclusive=True)
        def fltr(b): return not b.celldata.compatible
        return len(list(filter(fltr, blocks))) == 0


def fem_mesh_from_obj(*args, **kwargs):
    raise NotImplementedError


if __name__ == '__main__':
    pass
