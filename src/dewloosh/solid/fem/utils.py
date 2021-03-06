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

    values : int of shape (nN, nDOF, nRHS)
        3d numpy array of floats, listing values for each node
        in 'inds'.

    Returns
    -------
    dofinds : np.ndarray
        1d numpy array of integers, denoting global dof indices.

    dofvals : np.ndarray of shape (nN * nDOF, nRHS)
        2d numpy array of floats, denoting values on dofs in 'dofinds'.

    """
    nN, dof_per_node, nRHS = values.shape
    dofinds = np.zeros((nN * dof_per_node), dtype=inds.dtype)
    dofvals = np.zeros((nN * dof_per_node, nRHS), dtype=values.dtype)
    for node in prange(nN):
        for dof in prange(dof_per_node):
            i = node * dof_per_node + dof
            dofinds[i] = inds[node] * dof_per_node + dof
            for rhs in prange(nRHS):
                dofvals[i, rhs] = values[node, dof, rhs]
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
    (1) It is assumed that the first axis of K runs
        along the elements.

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


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def assemble_load_vector(values: ndarray, gnum: ndarray, N: int = -1):
    """
    Returns global dof numbering based on element 
    topology data.

    Parameters
    ----------
    values : np.ndarray of shape (nE, nEVAB, nRHS)
        3d numpy array of floats, representing element data.
        The length of the second axis matches the the number of
        degrees of freedom per cell.

    gnum : int
        Global indices of local degrees of freedoms of elements.

    N : int, Optional.
        The number of total unknowns in the system. Must be specified correcly,
        to get a vector the same size of the global system. If not specified, it is
        inherited from 'gnum' (as 'gnum.max() + 1'), but this can lead to a chopped 
        array.

    Returns
    -------
    np.ndarray
        2d numpy array of integers with a shape of (N, nRHS), where nRHS is the number
        if right hand sizes (load cases).

    """
    nE, nEVAB, nRHS = values.shape 
    if N < 0:
        N = gnum.max() + 1
    res = np.zeros((N, nRHS), dtype=values.dtype)
    for i in range(nE):
        for j in range(nEVAB):
            for k in prange(nRHS):
                res[gnum[i, j], k] += values[i, j, k]
    return res


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
def approximation_matrix(ndf: ndarray, NDOFN: int):
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
    """
    ncf : nodal_compatibility_factors
    """
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
        data[c: c + shapes[iN]] = flatten2dC(ncf[iN])
        nNN = widths[iN]
        for jNN in prange(nNN):
            for kNN in prange(nNN):
                rows[c + jNN*nNN + kNN] = nreg[iN][jNN]
                cols[c + jNN*nNN + kNN] = nreg[iN][kNN]
        c += shapes[iN]
    return data, rows, cols


@njit(nogil=True, parallel=True, cache=__cache)
def topo1d_to_gnum1d(topo1d: ndarray, NDOFN: int):
    nN = topo1d.shape[0]
    gnum1d = np.zeros(nN * NDOFN, dtype=topo1d.dtype)
    for i in prange(nN):
        for j in prange(NDOFN):
            gnum1d[i*NDOFN + j] = topo1d[i] * NDOFN + j
    return gnum1d


@njit(nogil=True, parallel=True, cache=__cache)
def ncf_to_cf(ncf: ndarray, NDOFN: int):
    nN = ncf.shape[0]
    cf = np.zeros((nN * NDOFN, nN * NDOFN), dtype=ncf.dtype)
    for i in prange(nN):
        for ii in prange(NDOFN):
            for j in prange(nN):
                cf[i*NDOFN + ii, j*NDOFN + ii] = ncf[i, j]
    return cf


@njit(nogil=True, cache=__cache)
def compatibility_factors(ncf: dict, nreg: dict, NDOFN: int):
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
def tr_cells_1d_in(A: ndarray, Q: ndarray):
    """
    Transforms element vectors (like the load vector) from global to local.
    
    Parameters
    ----------
    A : 3d NumPy float array of shape (nE, nEVAB)
        Array of coefficients to transform.
        
    Q : 3d NumPy float array of shape (nE, nEVAB, nEVAB)
        Transformation matrices.
        
    Returns
    -------
    numpy array
        NumPy array with the same shape as 'A'
    """
    res = np.zeros_like(A)
    for iE in prange(res.shape[0]):
        res[iE] = Q[iE] @ A[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tr_cells_1d_out(A: ndarray, Q: ndarray):
    """
    Transforms element vectors (like the load vector) from local to global.
    (nE, nNE * nDOF)
    """
    res = np.zeros_like(A)
    for iE in prange(res.shape[0]):
        res[iE] = Q[iE].T @ A[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tr_cells_1d_in_multi(A: ndarray, Q: ndarray):
    """
    Transforms element vectors (like the load vector) from local to global
    for multiple cases.
    (nE, nRHS, nNE * nDOF)
    """
    res = np.zeros_like(A)
    for iE in prange(res.shape[0]):
        for jRHS in prange(res.shape[1]):
            res[iE, jRHS] = Q[iE] @ A[iE, jRHS]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tr_cells_1d_out_multi(A: ndarray, Q: ndarray):
    """
    Transforms element vectors (like the load vector) from local to global
    for multiple cases.
    A (nE, nRHS, nP * nDOF)
    Q (nE, nP * nDOF)
    """
    res = np.zeros_like(A)
    for iE in prange(res.shape[0]):
        for jRHS in prange(res.shape[1]):
            res[iE, jRHS] = Q[iE].T @ A[iE, jRHS]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def element_dof_solution_bulk(dofsol1d: ndarray, gnum: ndarray):
    """
    dofsol (nN * nDOF, nRHS)
    gnum (nE, nEVAB)
    ---
    (nE, nEVAB, nRHS)
    """
    nRHS = dofsol1d.shape[1]
    nE, nEVAB = gnum.shape
    res = np.zeros((nE, nEVAB, nRHS), dtype=dofsol1d.dtype)
    for i in prange(nE):
        for j in prange(nRHS):
            res[i, :, j] = dofsol1d[gnum[i, :], j]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def transform_stiffness(K: ndarray, dcm: ndarray):
    """
    Transforms element stiffness matrices from local to global.
    """
    res = np.zeros_like(K)
    for iE in prange(res.shape[0]):
        res[iE] = dcm[iE].T @ K[iE] @ dcm[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def element_dofmap_bulk(dofmap: ndarray, nDOF: int, nNODE: int):
    nDOF_ = dofmap.shape[0]
    res = np.zeros(nNODE * nDOF_, dtype=dofmap.dtype)
    for i in prange(nNODE):
        for j in prange(nDOF_):
            res[i*nDOF_ + j] = i*nDOF + dofmap[j]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def expand_stiffness_bulk(K_in: ndarray, K_out: ndarray, dofmap: ndarray):
    """
    Expands the local stiffness matrix into a standard form.
    """
    nDOF = dofmap.shape[0]    
    for i in prange(nDOF):
        for j in prange(nDOF):
            K_out[:, dofmap[i], dofmap[j]] = K_in[:, i, j]
    return K_out


@njit(nogil=True, parallel=True, cache=__cache)
def pull_submatrix(A, r, c):
    nR = r.shape[0]
    nC = c.shape[0]
    res = np.zeros((nR, nC), dtype=A.dtype)
    for i in prange(nR):
        for j in prange(nC):
            res[i, j] = A[r[i], c[j]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def push_submatrix(A, Asub, r, c, new=True):
    nR = r.shape[0]
    nC = c.shape[0]
    res = np.zeros_like(A)
    if not new:
        res[:, :] = A
    for i in prange(nR):
        for j in prange(nC):
            res[r[i], c[j]] = Asub[i, j]
    return res
    

@njit(nogil=True, parallel=True, cache=__cache)
def constrain_local_stiffness_bulk(K: ndarray, factors: ndarray):
    """
    Returns the condensed stiffness matrices representing constraints
    on the internal forces of the elements (eg. hinges).
    
    Currently this solution is only able to handle two states, being total free 
    and being fully constrained. The factors are expected to be numbers between
    0 and 1, where dofs with a factor > 0.5 are assumed to be the constrained ones.
    
    Parameters
    ----------
    K : numpy.ndarray
        3d float array, the stiffness matrix for several elements of the same kind.
        
    factors: numpy.ndarray
        2d float array of connectivity facotors for each dof of every element.
                    
    Notes
    -----
    This solution applies the idea of static condensation.
    
    Returns
    -------
    numpy.ndarray
        The constrained stiffness matrices with the same shape as `K`.
        
    """
    nE, nV, _ = K.shape
    res = np.zeros_like(K)
    for iE in prange(nE):
        b = np.where(factors[iE] > 0.5)[0]
        i = np.where(factors[iE] <= 0.5)[0]
        Kbb = pull_submatrix(K[iE], b, b)
        Kii = pull_submatrix(K[iE], i, i)
        Kib = pull_submatrix(K[iE], i, b)
        Kbi = pull_submatrix(K[iE], b, i)
        Kbb -= Kbi @ np.linalg.inv(Kii) @ Kib
        res[iE] = push_submatrix(K[iE], Kbb, b, b, True)
    return res


def assert_min_stiffness_bulk(K: ndarray, minval = 1e-12):
    inds = np.arange(K.shape[-1])
    d = K[:, inds, inds]
    eid, vid = np.where(d < minval)
    K[eid, vid, vid] = minval
    return K


@njit(nogil=True, parallel=True, cache=__cache)
def internal_forces(K: ndarray, dofsol: ndarray):
    """
    Transforms element stiffness matrices from local to global.
    ---
    (nE, nRHS, nEVAB)
    """
    nE, nRHS, nEVAB  = dofsol.shape
    res = np.zeros_like(dofsol)
    for i in prange(nE):
        for j in prange(nRHS):
            res[i, j] = K[i] @ dofsol[i, j]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_mass_matrix_data(nodal_masses: ndarray, ndof:int=6):
    N = nodal_masses.shape[0]
    res = np.zeros((N*ndof))
    for i in prange(N):
        res[i*ndof : i*ndof + 3] = nodal_masses[i]
    return res
