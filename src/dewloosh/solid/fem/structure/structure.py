# -*- coding: utf-8 -*-
from scipy.sparse import coo_matrix as npcoo, csc_matrix as npcsc
from scipy.sparse.linalg import spsolve
from time import time
import numpy as np
from scipy.sparse import isspmatrix as isspmatrix_np

from dewloosh.core.abc.wrap import Wrapper

from dewloosh.math.array import repeat

from ..mesh import FemMesh, fem_mesh_from_obj
from ..linsolve import box_fem_data_bulk, unbox_lhs
from ..utils import irows_icols_bulk




__all__ = ['Structure']


try:
    import pypardiso as ppd
    from pypardiso import PyPardisoSolver
    #from pypardiso.scipy_aliases import pypardiso_solver
    __haspardiso__ = True
except Exception:
    __haspardiso__ = False
"""
PARDISO MATRIX TYPES
1
real and structurally symmetric
2
real and symmetric positive definite
-2
real and symmetric indefinite
3
complex and structurally symmetric
4
complex and Hermitian positive definite
-4
complex and Hermitian indefinite
6
complex and symmetric
11
real and nonsymmetric
13
complex and nonsymmetric
"""


class Structure(Wrapper):

    def __init__(self, *args, mesh: FemMesh=None, **kwargs):
        if not isinstance(mesh, FemMesh):
            mesh = fem_mesh_from_obj(mesh)
        super().__init__(wrap=mesh)
        assert mesh is not None, "Some kind of a finite element mesh must be \
            provided with keyword 'mesh'!"
        self.summary = {}
        self.solver = 'scipy'

    @property
    def mesh(self):
        return self._wrapped

    @mesh.setter
    def mesh(self, value: FemMesh):
        self._wrapped = value

    def linsolve(self, *args, **kwargs):
        self.preprocess(*args, **kwargs)
        self.process(*args, **kwargs)
        self.postprocess(*args, **kwargs)
    
    def populate_model(self, *args, **kwargs):
        blocks = self.mesh.cellblocks(inclusive=True)
        for block in blocks:
            nE = len(block.celldata)
            
            # populate material stiffness matrices
            if not 'mat' in block.celldata.fields:
                C = repeat(block.material_stiffness_matrix(), nE)
                block.celldata._wrapped['mat'] = C
            
            # populate frames
            if not 'frames' in block.celldata.fields:
                frames = repeat(block.frame.show(), nE)
                block.celldata._wrapped['frames'] = frames

    def preprocess(self, *args, sparsify=False, summary=True, 
                   ensure_comp=False, **kwargs):
        
        mesh = self._wrapped

        # --- populate model stiffness matrices ---
        self.populate_model()
        # --- nodal distribution factors ---
        mesh.set_nodal_distribution_factors()  # sets mesh.celldata.ndf   
        # natural boundary conditions
        _f = mesh.load_vector(*args, **kwargs)
        # essential boundary conditions
        _Kp_coo = mesh.penalty_matrix_coo(ensure_comp=ensure_comp, **kwargs)
        # stiffness matrix
        self._K_bulk = mesh.stiffness_matrix(*args, sparse=False, **kwargs)

        # assembly and boxing
        gnum = mesh.element_dof_numbering()
        self._N_ = _f.shape[0]
        self._Kp_coo, self._gnum, self._f, self._loc_to_glob = \
            box_fem_data_bulk(_Kp_coo, gnum, _f)
        self._N = self._gnum.max() + 1
        self._f = npcsc(self._f) if sparsify else self._f
        self._krows, self._kcols = irows_icols_bulk(self._gnum)
        self._krows = self._krows.flatten()
        self._kcols = self._kcols.flatten()

        self.summary = {'preproc': {}, 'proc': {}, 'postproc': {}}
        if summary:
            self.summary['preproc']['sparsify'] = sparsify

    def process(self, *args, use_umfpack=True, summary=True,
                permc_spec='COLAMD', solver='pardiso', mtype=11, **kwargs):
        Kr_coo = npcoo((self._K_bulk.flatten(), (self._krows, self._kcols)),
                       shape=(self._N, self._N), dtype=self._K_bulk.dtype)
        K_coo = Kr_coo + self._Kp_coo
        K_coo.eliminate_zeros()
        K_coo.sum_duplicates()
        t0 = time()
        
        if solver == 'pardiso' and __haspardiso__:
            if mtype == 11:
                self._du = ppd.spsolve(K_coo, self._f)
            else:
                pds = PyPardisoSolver(mtype=mtype)
                self._du = pds.solve(K_coo, self._f)
        elif solver == 'scipy':
            self._du = spsolve(K_coo.astype(np.float32),
                               self._f.astype(np.float32),
                               permc_spec=permc_spec,
                               use_umfpack=use_umfpack).astype(np.float64)
        else:
            raise NotImplementedError("Selected solver '{}' is not supported!".format(solver))
        
        dt = time() - t0
        if isspmatrix_np(self._du):
            self._du = self._du.todense()
        if summary:
            self.summary['proc'] = {
                'time [ms]': dt * 1000,
                'N': self._du.shape[0],
                'use_umfpack': use_umfpack,
                'permc_spec': permc_spec,
                'solver': solver
            }

    def postprocess(self, *args, summary=True, cleanup=True, **kwargs):
        # unbox
        du = unbox_lhs(self._du, self._loc_to_glob, self._N_)

        # store dof solution
        mesh = self._wrapped
        nDOFN = mesh.NDOFN
        nN = mesh.number_of_points()
        mesh.pointdata['dofsol'] = np.reshape(du, (nN, nDOFN))
        #mesh.postproc_dof_solution()

        # clean up
        if cleanup:
            self._K_bulk, self._Kp_coo, self._gnum, self._f, \
                self._loc_to_glob, self._du, self._N_, self._N = \
                None, None, None, None, None, None, None, None

        if summary:
            self.summary['number of elements'] = mesh.number_of_cells()
            self.summary['number of nodes'] = nN
            self.summary['dofs per node'] = nDOFN

    def dofsol(self, *args, flatten=True, **kwargs):
        if flatten:
            return self.mesh.pointdata.dofsol.to_numpy().flatten()
        else:
            return self.mesh.pointdata.dofsol.to_numpy()


if __name__ == '__main__':
    pass
