# -*- coding: utf-8 -*-
import numpy as np
import numpy as np
from scipy.sparse import coo_matrix as npcoo, csc_matrix as npcsc

from .utils import irows_icols_bulk
from .linsolve import box_fem_data_bulk, solve_standard_form, unbox_lhs
from .eigsolve import eig_sparse, eig_dense, calc_eig_res


class FemSolver:

    def __init__(self, K, Kp, f, gnum, imap=None, regular=True, M=None, **config):
        self.K = K
        self.M = M
        self.Kp = Kp
        self.gnum = gnum
        self.f = f
        self.vmodes = None  # vibration modes : Tuple(vals, vecs)
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
        self.core = self.encode()
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
            self.core.preproc()
            self.Ke = self.core.Ke
        else:
            self.Ke = npcoo((self.K.flatten(), (self.krows, self.kcols)),
                            shape=(self.N, self.N), dtype=self.K.dtype)
        self.READY = True

    def proc(self, core=None):
        if not self.regular:
            return self.core.proc()
        Kcoo = self.Ke + self.Kp
        Kcoo.eliminate_zeros()
        Kcoo.sum_duplicates()
        self.u, self.summary = solve_standard_form(
            Kcoo, self.f, summary=True, core=core)

    def postproc(self):
        if not self.regular:
            self.core.postproc()
            return self.decode()
        self.r = np.reshape(self.Ke.dot(self.u), self.f.shape) - self.f

    def linsolve(self, *args, core=None, summary=False, **kwargs):
        self.summary = {}

        self.preproc()
        self.proc(core=core)
        self.postproc()

        if summary:
            return self.u, self.summary
        else:
            return self.u

    def decode(self):
        assert not self.regular
        N = self.gnum.max() + 1
        self.u = unbox_lhs(self.core.u, self.imap, N=N)
        self.r = unbox_lhs(self.core.r, self.imap, N=N)

    def natural_circular_frequencies(self, *args, k=10, return_vectors=False,
                                     maxiter=5000, normalize=True, as_dense=False, 
                                     **kwargs):
        """
        Returns the circular frequencies (\omega).
        """
        K = self.Ke + self.Kp
        if as_dense:
            vals, vecs = eig_dense(K, *args, M=self.M, nmode='M',
                                   normalize=normalize,
                                   return_residuals=False, **kwargs)
        else:
            vals, vecs = eig_sparse(K, *args, k=k, M=self.M, nmode='M',
                                    normalize=normalize, maxiter=maxiter,
                                    return_residuals=False, **kwargs)
        cfreqs = np.sqrt(vals)
        if return_vectors:
            return cfreqs, vecs
        return cfreqs

    def natural_cyclic_frequencies(self, *args, return_vectors=False, **kwargs):
        """
        Returns total oscillations done by the body in unit time (f).
        """
        kwargs['return_vectors'] = True
        vals, vecs = self.natural_circular_frequencies(*args, **kwargs)
        vals = vals / (2 * np.pi)
        if return_vectors:
            return vals, vecs
        return vals

    def natural_periods(self, *args, return_vectors=False, **kwargs):
        """
        Returns the times required to make a full cycle of vibration (T).
        """
        kwargs['return_vectors'] = True
        vals, vecs = self.natural_cyclic_frequencies(*args, **kwargs)
        vals = 1 / vals
        if return_vectors:
            return vals, vecs
        return vals

    def modes_of_vibration(self, *args, around=None, normalize=True, 
                           return_residuals=False, **kwargs):
        """
        Returns eigenvalues and eigenvectors as a tuple of two numpy arrays.        
        
        Notes
        -----
        Evalauated values are available as `obj.vmodes`.
        
        """
        if around is not None:
            sigma = (np.pi * 2 * around)**2
            kwargs['sigma'] = sigma
        self.vmodes = self.natural_cyclic_frequencies(
            *args, normalize=normalize, return_vectors=True, **kwargs)
        if return_residuals:
            vals, vecs = self.vmodes
            K = self.Ke + self.Kp
            r = calc_eig_res(K, self.M, vals, vecs)
            return self.vmodes, r
        return self.vmodes