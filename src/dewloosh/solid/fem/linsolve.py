# -*- coding: utf-8 -*-
from re import U
import numpy as np
from numpy import ndarray
import numpy as np
from scipy.sparse import coo_matrix as coo, spmatrix, isspmatrix as isspmatrix_np
from scipy.sparse.linalg import spsolve, spsolve_triangular, splu
from scipy.sparse import coo_matrix as npcoo, csc_matrix as npcsc
from typing import Union
from time import time
#from concurrent.futures import ThreadPoolExecutor

from dewloosh.math.array import matrixform

from .imap import index_mappers, box_spmatrix, box_rhs, unbox_lhs, box_dof_numbering
from .utils import irows_icols_bulk
from .preproc import fem_coeff_matrix_coo
from .utils import irows_icols_bulk

from ..config import __haspardiso__

if __haspardiso__:
    import pypardiso as ppd
    from pypardiso import PyPardisoSolver
    #from pypardiso.scipy_aliases import pypardiso_solver

arraylike = Union[ndarray, spmatrix]


def solve_standard_form(K: coo, f: np.ndarray, *args, use_umfpack=True, summary=False,
                        permc_spec='COLAMD', solver=None, mtype=11, assume_regular=False,
                        **kwargs):
    solver = 'pardiso' if __haspardiso__ else 'scipy' if solver is None else solver
    if solver == 'pardiso' and not __haspardiso__:
        raise ImportError(
            "You need to install 'pypardiso' for solver type <{}>".format(solver))

    if not assume_regular:
        K.eliminate_zeros()
        K.sum_duplicates()
        f = matrixform(f)

    if solver == 'pardiso':
        if mtype == 11:
            t0 = time()
            u = ppd.spsolve(K, f)
            dt = time() - t0
        else:
            pds = PyPardisoSolver(mtype=mtype)
            t0 = time()
            u = pds.solve(K, f)
            dt = time() - t0
    elif solver == 'scipy':
        if len(args) > 0:
            if 'lower' in args:
                solver = 'scipy.sparse.linalg.spsolve_triangular (lower)'
                t0 = time()
                u = spsolve_triangular(K, f, lower=True)
                dt = time() - t0
            elif 'upper' in args:
                solver = 'scipy.sparse.linalg.spsolve_triangular (upper)'
                t0 = time()
                u = spsolve_triangular(K, f, lower=False)
                dt = time() - t0
            elif 'SLU' in args:
                solver = 'scipy.sparse.linalg.SuperLU'
                K = K.tocsc()
                t0 = time()
                LU = splu(K)
                u = LU.solve(f)
                dt = time() - t0
        else:
            if use_umfpack:
                use_umfpack = True if f.shape[1] == 1 else False
            solver = 'scipy.sparse.linalg.spsolve'
            t0 = time()
            u = spsolve(K, f, permc_spec=permc_spec, use_umfpack=use_umfpack)
            dt = time() - t0
    else:
        raise NotImplementedError(
            "Selected solver '{}' is not supported!".format(solver))

    if isspmatrix_np(u):
        u = u.todense()
    u = u.reshape(f.shape)

    # residual
    if summary:
        d = {
            'time [s]': dt * 1000,
            'N': u.shape[0],
            'solver': solver,
            'options': {}}
        if solver == 'scipy':
            d['options']['use_umfpack'] = use_umfpack
            d['options']['permc_spec'] = permc_spec
        elif solver == 'pardiso':
            d['options']['mtype'] = mtype
        return u, d
    else:
        return u


def box_fem_data_sparse(K_coo: coo, Kp_coo: coo, f: ndarray):
    """
    Notes:
    ------
        (1) If the load vector 'f' is dense, it must contain values for all
            nodes, even the passive ones.
    """
    # data for boxing and unboxing
    loc_to_glob, glob_to_loc = index_mappers(K_coo, return_inverse=True)
    # boxing
    K = box_spmatrix(K_coo, glob_to_loc) + box_spmatrix(Kp_coo, glob_to_loc)
    f = box_rhs(matrixform(f), loc_to_glob)
    return K, f, loc_to_glob


def box_fem_data_bulk(Kp_coo: coo, gnum: ndarray, f: ndarray):
    """
    Notes:
    ------
        (1) If the load vector 'f' is dense, it must contain values for all
            nodes, even the passive ones.
    """
    # data for boxing and unboxing
    N = f.shape[0]
    loc_to_glob, glob_to_loc = index_mappers(gnum, N=N, return_inverse=True)
    # boxing
    gnum = box_dof_numbering(gnum, glob_to_loc)
    Kp_coo = box_spmatrix(Kp_coo, glob_to_loc)
    f = box_rhs(matrixform(f), loc_to_glob)
    return Kp_coo, gnum, f, loc_to_glob


def box_fem_data_bulk2(Kp_coo: coo, gnum: ndarray, f: ndarray):
    """
    Notes:
    ------
        (1) If the load vector 'f' is dense, it must contain values for all
            nodes, even the passive ones.
    """
    # data for boxing and unboxing
    N = f.shape[0]
    glob_to_loc = np.full(N, -1, dtype=int)
    inds = np.unique(gnum.flatten())
    glob_to_loc[inds] = np.arange(len(inds))
    gnum = box_dof_numbering(gnum, glob_to_loc)
    Kp_coo = box_spmatrix(Kp_coo, glob_to_loc)

    loc_to_glob, glob_to_loc = index_mappers(gnum, N=N, return_inverse=True)
    # boxing
    gnum = box_dof_numbering(gnum, glob_to_loc)
    Kp_coo = box_spmatrix(Kp_coo, glob_to_loc)
    f = box_rhs(matrixform(f), loc_to_glob)
    return Kp_coo, gnum, f, loc_to_glob


def linsolve_sparse(K_coo: coo, Kp_coo: coo,
                    f: np.ndarray, *args, use_umfpack=True,
                    sparsify=False, summary=True, **kwargs):
    """
    Notes:
    ------
        (1) If the load vector 'f' is dense, it must contain values for all
            nodes, even the passive ones.
    """
    # boxing
    N = f.shape[0]
    K, f, loc_to_glob = box_fem_data_sparse(K_coo, Kp_coo, f)
    # solution
    u, d = solve_standard_form(K, f, *args, sparsify=sparsify,
                               use_umfpack=use_umfpack,
                               summary=True, **kwargs)
    # unboxing
    u = unbox_lhs(u, loc_to_glob, N)

    if summary:
        d['regular'] = loc_to_glob is not None
        return u, d
    return u


def linsolve_bulk(K_bulk: np.ndarray, Kp_coo: coo, f: np.ndarray,
                  gnum: np.ndarray, *args, **kwargs):
    K_coo = fem_coeff_matrix_coo(K_bulk, inds=gnum, N=f.shape[0])
    return linsolve_sparse(K_coo, Kp_coo, f, *args, **kwargs)


def linsolve(K: arraylike, *args, gnum=None, **kwargs):
    if isspmatrix_np(K):
        return linsolve_sparse(K, *args, **kwargs)
    elif isinstance(K, np.ndarray) and gnum is not None:
        return linsolve_bulk(K, *args, gnum, **kwargs)


class FemSolver:

    def __init__(self, K, Kp, f, gnum, imap=None, regular=True, **config):
        self.K = K
        self.Kp = Kp
        self.gnum = gnum
        self.f = f
        self.config = config
        self.regular = regular
        self.imap = imap
        if imap is not None:
            # Ff imap.shape[0] > f.shape[0] it means that the inverse of
            # the mapping is given. It would require to store more data, but
            # it would enable to tell the total size of the equation system, that the
            # input is a representation of.
            assert imap.shape[0] == f.shape[0]
        self.regular = False if imap is not None else regular
        self.solver = self.encode()
        self.READY = False

    def encode(self) -> 'FemSolver':
        if self.imap is None and not self.regular:
            Kp, gnum, f, imap = box_fem_data_bulk(self.Kp, self.gnum, self.f)
            self.imap = imap
            return FemSolver(self.K, Kp, f, gnum, regular=True)
        else:
            return self

    def preproc(self, force=False):
        if self.READY and not force:
            return
        self.N = self.gnum.max() + 1
        self.f = npcsc(self.f) if self.config.get(
            'sparsify', False) else self.f
        self.krows, self.kcols = irows_icols_bulk(self.gnum)
        self.krows = self.krows.flatten()
        self.kcols = self.kcols.flatten()

        if not self.regular:
            self.solver.preproc()
            self.Ke = self.solver.Ke
        else:
            self.Ke = npcoo((self.K.flatten(), (self.krows, self.kcols)),
                            shape=(self.N, self.N), dtype=self.K.dtype)
        self.READY = True

    def proc(self, solver=None):
        if not self.regular:
            return self.solver.proc()
        Kcoo = self.Ke + self.Kp
        Kcoo.eliminate_zeros()
        Kcoo.sum_duplicates()
        self.u, self.summary = solve_standard_form(
            Kcoo, self.f, summary=True, solver=solver)

    def postproc(self):
        if not self.regular:
            self.solver.postproc()
            return self.decode()
        self.r = np.reshape(self.Ke.dot(self.u), self.f.shape) - self.f

    def linsolve(self, *args, solver=None, summary=False, **kwargs):
        self.summary = {}

        self.preproc()
        self.proc(solver=solver)
        self.postproc()

        if summary:
            return self.u, self.summary
        else:
            return self.u

    def decode(self):
        assert not self.regular
        N = self.gnum.max() + 1
        self.u = unbox_lhs(self.solver.u, self.imap, N=N)
        self.r = unbox_lhs(self.solver.r, self.imap, N=N)


"""def linsolve_bulk_pop(K_bulk: np.ndarray, Kp_coo: coo, f: np.ndarray,
                      gnum: np.ndarray, fsolvers: np.ndarray, *args,
                      sparsify=True, use_umfpack=True,
                      summary=False, parallel=True,
                      max_workers=4, permc_spec='COLAMD', **kwargs):
    f = matrixform(f)
    N = f.shape[0]
    loc_to_glob, glob_to_loc = \
        index_mappers(gnum, N=N, return_inverse=True)
    nPop, nE = fsolvers.shape
    gnum = box_dof_numbering(gnum, glob_to_loc)
    M = gnum.max() + 1
    dtype = K_bulk.dtype
    shape = (M, M)

    # rhs, lhs
    u = np.zeros((nPop,) + f.shape, dtype=dtype)
    f = box_rhs(f.astype(dtype), loc_to_glob)
    Kp_coo = box_spmatrix(Kp_coo, glob_to_loc)

    # stiffness utils
    rows, cols = list(map(lambda x: x.flatten(), irows_icols_bulk(gnum)))

    def K_bulk_w(i): return weighted_stiffness_bulk(
        K_bulk, fsolvers[i]).flatten()
    def K_coo(i): return coo((K_bulk_w(i), (rows, cols)), shape=shape,
                             dtype=dtype)

    def K(i): return K_coo(i) + Kp_coo

    def solve_and_unbox(i):
        ui = solve_standard_form(K(i), f, *args, sparsify=sparsify,
                                 use_umfpack=use_umfpack, permc_spec=permc_spec,
                                 **kwargs)
        u[i, :, :] = unbox_lhs(ui, loc_to_glob, N)

    # serial and parallel solutions
    if not parallel:
        t0 = time()
        list(map(solve_and_unbox, range(nPop)))
        dt = time() - t0
    else:
        max_workers = min(max_workers, nPop)
        t0 = time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(solve_and_unbox, range(nPop))
        dt = time() - t0

    if summary:
        d = {
            'time [ms]': dt * 1000,
            'avg. time [ms]': dt * 1000 / nPop,
            'N': M,
            'use_umfpack': use_umfpack,
            'permc_spec': permc_spec,
            'sparsify': sparsify,
            'solver': 'scipy.sparse.linalg.spsolve',
            'max workers': max_workers if parallel else 1,
            'population size': nPop
        }
        return u, d
    return u
"""
