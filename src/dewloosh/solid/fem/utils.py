# -*- coding: utf-8 -*-
from numba import njit, prange
import numpy as np
from numpy import ndarray

from dewloosh.math.linalg.sparse.csr import csr_matrix as csr
from dewloosh.math.array import find1d, flatten2dC

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def fixity2d_to_dofs1d(fixity2d: np.ndarray, inds: np.ndarray = None):
    """
    Returns the indices of the degrees of freedoms
    being supressed. 

    Optionally, global indices of the rows in 'fixity2d' 
    array can be provided by the optional argument 'inds'.

    Parameters
    ----------
    fixity2d : np.ndarray(bool)[:, :]
        2d numpy array of booleans. It has as many rows as nodes, and as 
        many columns as derees of freedom per node in the model. A True
        value means that the corresponding dof is fully supressed.

    inds : np.ndarray, optional
        1d numpy array of integers, default is None. If provided, it should
        list the global indices of the nodes, for which the data is provided
        by the array 'fixity2d'.

    Returns
    -------
    dofinds : np.ndarray
        1d numpy array of integers. 

    """
    ndof = fixity2d.shape[1]
    args = np.argwhere(fixity2d == True)
    N = args.shape[0]
    res = np.zeros(N, dtype=args.dtype)
    for i in prange(N):
        res[i] = args[i, 0]*ndof + args[i, 1]
    if inds is None:
        return res
    else:
        dofinds = np.zeros_like(res)
        for i in prange(len(inds)):
            for j in prange(ndof):
                dofinds[i*ndof + j] = inds[i]*ndof + j
        return dofinds[res]


@njit(nogil=True, parallel=True, cache=__cache)
def nodes2d_to_dofs1d(inds: np.ndarray, values: np.ndarray):
    """
    Returns a tuple of degree of freedom indices and data, 
    based on a nodal definition.

    Parameters
    ----------
    inds : np.ndarray
        1d numpy array of integers, listing global node indices.

    values : int
        2d numpy array of floats, listing values for each node
        in 'inds'.

    Returns
    -------
    dofinds : np.ndarray
        1d numpy array of integers, denoting global dof indices.

    dofvals : np.ndarray
        1d numpy array of floats, denoting values on dofs in 'dofinds'.

    """
    nN, dof_per_node = values.shape
    dofinds = np.zeros((nN * dof_per_node), dtype=inds.dtype)
    dofvals = np.zeros(dofinds.shape, dtype=values.dtype)
    for node in prange(nN):
        for dof in prange(dof_per_node):
            i = node * dof_per_node + dof
            dofinds[i] = inds[node] * dof_per_node + dof
            dofvals[i] = values[node, dof]
    return dofinds, dofvals


@njit(nogil=True, cache=__cache, parallel=True)
def weighted_stiffness_bulk(K: np.ndarray, weights: np.ndarray):
    """
    Returns a weighted stiffness matrix.

    Parameters
    ----------
    K : np.ndarray
        2d numpy array of floats

    weights : np.ndarray
        1d numpy array of floats

    Returns
    -------
    Kw : np.ndarray
        2d numpy array of floats

    Notes
    -----
    (1) It is assumed that the firs axis of K runs
        along finite elements.

    """
    nE = len(weights)
    res = np.zeros_like(K)
    for i in prange(nE):
        res[i, :] = weights[i] * K[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def irows_icols_bulk(edofs: np.ndarray):
    """
    Returns row and column index data for several finite elements.

    Parameters
    ----------
    edofs : np.ndarray
        2d numpy array. Each row has the meaning of global degree of 
        freedom numbering for a given finite element.

    Returns
    -------
    irows, icols : np.ndarray, np.ndarray
        Global indices of the rows and columns of the stiffness matrices
        of the elements.

    Notes
    -----
    The implementation assumes that every cell has the same number of
    degrees of freedom.

    """
    nE, nNE = edofs.shape
    nTOTV = nNE * nNE
    irows = np.zeros((nE, nTOTV), dtype=edofs.dtype)
    icols = np.zeros((nE, nTOTV), dtype=edofs.dtype)
    for iNE in prange(nNE):
        for jNE in prange(nNE):
            i = iNE * nNE + jNE
            for iE in prange(nE):
                irows[iE, i] = edofs[iE, iNE]
                icols[iE, i] = edofs[iE, jNE]
    return irows, icols


@njit(nogil=True, cache=__cache, parallel=True)
def irows_icols_bulk_filtered(edofs: np.ndarray, inds: np.ndarray):
    """
    Returns row and column index data for finite elements specified
    by the index array `inds`.

    Parameters
    ----------
    edofs : np.ndarray
        2d numpy array. Each row has the meaning of global degree of 
        freedom numbering for a given finite element.

    inds: np.ndarray
        1d numpy array of integers specifying active elements in an assembly.

    Returns
    -------
    irows, icols : np.ndarray, np.ndarray
        Global indices of the rows and columns of the stiffness matrices
        of the elements.

    Notes
    -----
    The implementation assumes that every cell has the same number of
    degrees of freedom.

    """
    nI = len(inds)
    nNE = edofs.shape[1]
    nTOTV = nNE * nNE
    irows = np.zeros((nI, nTOTV), dtype=edofs.dtype)
    icols = np.zeros((nI, nTOTV), dtype=edofs.dtype)
    for iNE in prange(nNE):
        for jNE in prange(nNE):
            i = iNE * nNE + jNE
            for iI in prange(nI):
                irows[iI, i] = edofs[inds[iI], iNE]
                icols[iI, i] = edofs[inds[iI], jNE]
    return irows, icols


@njit(nogil=True, cache=__cache, parallel=True)
def topo_to_gnum(topo: np.ndarray, ndofn: int):
    """
    Returns global dof numbering based on element 
    topology data.

    Parameters
    ----------
    topo : np.ndarray
        2d numpy array of integers. Topology array listing global
        node numbers for several elements.

    ndofn : int
        Number of degrees of freedoms per node.

    Returns
    -------
    gnum : np.ndarray
        2d numpy array of integers.

    """
    nE, nNE = topo.shape
    gnum = np.zeros((nE, nNE*ndofn), dtype=topo.dtype)
    for i in prange(nE):
        for j in prange(nNE):
            for k in prange(ndofn):
                gnum[i, j*ndofn + k] = topo[i, j]*ndofn + k
    return gnum


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def penalty_factor_matrix(cellfixity: ndarray, shp: ndarray):
    nE, nNE, nDOF = cellfixity.shape
    N = nDOF * nNE
    res = np.zeros((nE, N, N), dtype=shp.dtype)
    for iE in prange(nE):
        for iNE in prange(nNE):
            for iDOF in prange(nDOF):
                fix = cellfixity[iE, iNE, iDOF]
                for jNE in prange(nNE):
                    i = jNE * nDOF + iDOF
                    for kNE in prange(nNE):
                        j = kNE * nDOF + iDOF
                        res[iE, i, j] += fix * shp[iNE, jNE] * shp[iNE, kNE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def approximation_matrix(ndf: ndarray, NDOFN : int):
    """Returns a matrix of approximation coefficients 
    for all elements."""
    nE, nNE = ndf.shape[:2]
    N = nNE * NDOFN
    nappr = np.eye(N, dtype=ndf.dtype)
    res = np.zeros((nE, N, N), dtype=ndf.dtype)
    for iE in prange(nE):
        for i in prange(nNE):
            for j in prange(nNE):
                for ii in prange(NDOFN):
                    for jj in prange(NDOFN):
                        res[iE, i*2 + ii, j*2 + jj] = nappr[i, j] * ndf[iE, i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_approximation_matrix(ndf: ndarray):
    """Returns a matrix of nodal approximation coefficients 
    for all elements."""
    nE, nNE = ndf.shape[:2]
    nappr = np.eye(nNE, dtype=ndf.dtype)
    res = np.zeros((nE, nNE, nNE), dtype=ndf.dtype)
    for iE in prange(nE):
        for i in prange(nNE):
            for j in prange(nNE):
                res[iE, i, j] = nappr[i, j] * ndf[iE, i]
    return res


@njit(nogil=True, cache=__cache)
def nodal_compatibility_factors(nam_csr_tot: csr, nam: ndarray, topo: ndarray):
    indptr = nam_csr_tot.indptr
    indices = nam_csr_tot.indices
    data = nam_csr_tot.data
    nN = len(indptr)-1
    widths = np.zeros(nN, dtype=indptr.dtype)
    for iN in range(nN):
        widths[iN] = indptr[iN+1] - indptr[iN]
    factors = dict()
    nreg = dict()
    for iN in range(nN):
        factors[iN] = np.zeros((widths[iN], widths[iN]), dtype=nam.dtype)
        nreg[iN] = indices[indptr[iN]: indptr[iN+1]]
    nE, nNE = topo.shape
    for iE in range(nE):
        topoE = topo[iE]
        for jNE in range(nNE):
            nID = topo[iE, jNE]
            dataN = data[indptr[nID]: indptr[nID+1]]
            topoN = nreg[nID]
            dataNE = np.zeros(widths[nID], dtype=nam.dtype)
            imap = find1d(topoE, topoN)
            dataNE[imap] = nam[iE, jNE]
            rNE = dataN - dataNE  # residual
            factors[nID] += np.outer(rNE, rNE)
    return factors, nreg


@njit(nogil=True, parallel=True, cache=__cache)
def compatibility_factors_to_coo(ncf: dict, nreg: dict):
    """ncf : nodal_compatibility_factors"""
    nN = len(ncf)
    widths = np.zeros(nN, dtype=np.int32)
    for iN in prange(nN):
        widths[iN] = len(nreg[iN])
    shapes = (widths**2).astype(np.int64)
    N = np.sum(shapes)
    data = np.zeros(N, dtype=np.float64)
    rows = np.zeros(N, dtype=np.int32)
    cols = np.zeros(N, dtype=np.int32)
    c = 0
    for iN in range(nN):
        data[c : c + shapes[iN]] = flatten2dC(ncf[iN])
        nNN = widths[iN]
        for jNN in prange(nNN):
            for kNN in prange(nNN):
                rows[c + jNN*nNN + kNN] = nreg[iN][jNN]
                cols[c + jNN*nNN + kNN] = nreg[iN][kNN]
        c += shapes[iN]
    return data, rows, cols


@njit(nogil=True, parallel=True, cache=__cache)
def topo1d_to_gnum1d(topo1d : ndarray, NDOFN : int):
    nN = topo1d.shape[0]
    gnum1d = np.zeros(nN * NDOFN, dtype=topo1d.dtype)
    for i in prange(nN):
        for j in prange(NDOFN):
            gnum1d[i*NDOFN + j] = topo1d[i] * NDOFN + j
    return gnum1d


@njit(nogil=True, parallel=True, cache=__cache)
def ncf_to_cf(ncf : ndarray, NDOFN : int):
    nN = ncf.shape[0]
    cf = np.zeros((nN * NDOFN, nN * NDOFN), dtype=ncf.dtype)
    for i in prange(nN):
        for ii in prange(NDOFN):
            for j in prange(nN):
                cf[i*NDOFN + ii, j*NDOFN + ii] = ncf[i, j]
    return cf


@njit(nogil=True, cache=__cache)
def compatibility_factors(ncf: dict, nreg: dict, NDOFN : int):
    """ncf : nodal_compatibility_factors"""
    nN = len(ncf)
    widths = np.zeros(nN, dtype=np.int32)
    for iN in prange(nN):
        widths[iN] = len(nreg[iN])
    cf = dict()
    reg = dict()
    for iN in range(nN):
        cf[iN] = ncf_to_cf(ncf[iN], NDOFN)
        reg[iN] = topo1d_to_gnum1d(nreg[iN], NDOFN)
    return cf, reg


@njit(nogil=True, parallel=True, cache=__cache)
def element_transformation_matrices_bulk(Q: ndarray, nNE: int=2):
    nE = Q.shape[0]
    nEVAB = nNE * 6
    res = np.zeros((nE, nEVAB, nEVAB), dtype=Q.dtype)
    for iE in prange(nE):
        for j in prange(2*nNE):
            res[iE, 3*j : 3*(j+1), 3*j : 3*(j+1)] = Q[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def transform_element_stiffness_matrices(K: ndarray, frames: ndarray):
    res = np.zeros_like(K)
    for iE in prange(res.shape[0]):
        res[iE] = frames[iE].T @ K[iE] @ frames[iE]
    return res
