# -*- coding: utf-8 -*-
from re import U
import numpy as np

from dewloosh.core.abc.wrap import Wrapper
from dewloosh.core import squeeze

from dewloosh.math.array import repeat

from ..mesh import FemMesh, fem_mesh_from_obj
from ..linsolve import FemSolver as Solver


__all__ = ['Structure']


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

    def __init__(self, *args, mesh: FemMesh = None, **kwargs):
        if not isinstance(mesh, FemMesh):
            mesh = fem_mesh_from_obj(mesh)
        super().__init__(wrap=mesh)
        assert mesh is not None, "Some kind of a finite element mesh must be \
            provided with keyword 'mesh'!"
        self.summary = {}
        self.solver = 'scipy'
        self._SBF_ = None  # Bilinear Standard Form

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

    def initialize(self, *args, **kwargs):
        blocks = self.mesh.cellblocks(inclusive=True)
        for block in blocks:
            nE = len(block.celldata)

            # populate material stiffness matrices
            if not 'mat' in block.celldata.fields:
                C = block.material_stiffness_matrix()
                if not len(C.shape) == 3:
                    C = repeat(block.material_stiffness_matrix(), nE)
                block.celldata._wrapped['mat'] = C

            # populate frames
            if not 'frames' in block.celldata.fields:
                frames = repeat(block.frame.show(), nE)
                block.celldata._wrapped['frames'] = frames

    def preprocess(self, *args, **kwargs):
        mesh = self._wrapped
        self.initialize()
        mesh.nodal_distribution_factors(store=True, key='ndf')  # sets mesh.celldata.ndf
        self.Solver = self.to_standard_form()

    def to_standard_form(self, *args, ensure_comp=False, **kwargs) -> Solver:
        mesh = self._wrapped
        f = mesh.load_vector()
        Kp_coo = mesh.penalty_matrix_coo(ensure_comp=ensure_comp, **kwargs)
        K_bulk = mesh.stiffness_matrix(*args, sparse=False, **kwargs)
        gnum = mesh.element_dof_numbering()
        return Solver(K_bulk, Kp_coo, f, gnum, regular=False)

    def process(self, *args, **kwargs):
        self.Solver.linsolve(*args, **kwargs)
        if kwargs.get('summary', False):
            self.summary = self.Solver.summary

    def postprocess(self, *args, summary=True, cleanup=False, **kwargs):
        mesh = self._wrapped
        nDOFN = mesh.NDOFN
        nN = mesh.number_of_points()
        
        # store dof solution
        u = self.Solver.u
        nRHS = 1 if len(u.shape) == 1 else u.shape[-1]
        if nRHS == 1:
            mesh.pointdata['dofsol'] = np.reshape(u, (nN, nDOFN))
        else:
            mesh.pointdata['dofsol'] = np.reshape(u, (nN, nDOFN, nRHS))

        # store nodal loads
        f = self.Solver.f
        nRHS = 1 if len(f.shape) == 1 else f.shape[-1]
        if nRHS == 1:
            mesh.pointdata['forces'] = np.reshape(f, (nN, nDOFN))
        else:
            mesh.pointdata['forces'] = np.reshape(f, (nN, nDOFN, nRHS))

        # store dof solution
        r = self.Solver.r
        nRHS = 1 if len(r.shape) == 1 else r.shape[-1]
        if nRHS == 1:
            mesh.pointdata['reactions'] = np.reshape(r, (nN, nDOFN))
        else:
            mesh.pointdata['reactions'] = np.reshape(r, (nN, nDOFN, nRHS))

        mesh.postprocess(*args, **kwargs)

        # clean up
        _ = self.cleanup() if cleanup else None

        if summary:
            self.summary['number of elements'] = mesh.number_of_cells()
            self.summary['number of nodes'] = nN
            self.summary['dofs per node'] = nDOFN

    def cleanup(self):
        self.Solver = None

    @squeeze(True)
    def nodal_dof_solution(self, *args, flatten=False, squeeze=True, **kwargs):
        return self.mesh.nodal_dof_solution(*args, flatten=flatten, squeeze=False, **kwargs)

    @squeeze(True)
    def reaction_forces(self, *args, flatten=False, squeeze=True, **kwargs):
        return self.mesh.reaction_forces(*args, flatten=flatten, squeeze=False, **kwargs)

    @squeeze(True)
    def nodal_forces(self, *args, flatten=False, squeeze=True, **kwargs):
        return self.mesh.nodal_forces(*args, flatten=flatten, squeeze=False, **kwargs)
    
    @squeeze(True)
    def internal_forces(self, *args, flatten=False, squeeze=True, **kwargs):
        return self.mesh.internal_forces(*args, flatten=flatten, squeeze=False, **kwargs)


if __name__ == '__main__':
    pass
