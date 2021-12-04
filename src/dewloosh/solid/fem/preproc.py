# -*- coding: utf-8 -*-
from scipy.sparse import coo_matrix
import numpy as np
from pyoneer.math.array import isintegerarray, isfloatarray, \
    is1dintarray, is1dfloatarray, isboolarray, bool_to_float
from pyoneer.math.linalg.sparse.utils import lower_spdata, upper_spdata
from pyoneer.mech.fem.utils import nodes2d_to_dofs1d, irows_icols_bulk


def fem_coeff_matrix_coo(A: np.ndarray, *args,
                         inds: np.ndarray = None,
                         rows: np.ndarray = None,
                         cols: np.ndarray = None,
                         N: int = None, **kwargs):
    """
    Returns the coefficient matrix in sparse 'coo' format.
    Index mapping is either specified with `inds` or with
    both `rows` and `cols`.

    If `lower` or `upper` is provided as a positional argument, 
    the result is returned as a lower- or upper-triangular matrix, 
    respectively. 

    Parameters
    ----------
    A : np.ndarray[:, :, :]
        Element coefficient matrices in dense format.

    inds : np.ndarray[:, :], optional
        Global indices. The default is None.

    rows : np.ndarray[:], optional
        Global numbering of the rows. The default is None.

    cols : np.ndarray[:], optional
        Global numbering of the columns. The default is None.

    N : int, optional
        Total number of coefficients in the system. If not provided, 
        it is inferred from index data.

    Returns
    -------
    scipy.sparse.coo_matrix
        Coefficient matrix in sparse coo format (see scipy doc for the details).

    """
    if N is None:
        assert inds is not None, "shape or `inds` must be provided!."
        N = inds.max() + 1
    if rows is None:
        assert inds is not None, "Row and column indices (`rows` and `cols`) " \
            "or global index numbering array (inds) must be provided!"
        rows, cols = irows_icols_bulk(inds)

    data, rows, cols = A.flatten(), rows.flatten(), cols.flatten()

    if len(args) > 0:
        if 'lower' in args:
            data, rows, cols = lower_spdata(data, rows, cols)
        elif 'upper' in args:
            data, rows, cols = upper_spdata(data, rows, cols)

    return coo_matrix((data, (rows, cols)), shape=(N, N))


def fem_nbc_data(*args, inds: np.ndarray = None, vals: np.ndarray = None,
                 N: int = None, **kwargs):
    """
    Returns the data necessary to create a load vector
    representing natural boundary conditions.

    Parameters
    ----------

    inds : np.ndarray(int), optional
        1d numpy integer array specifying node or dof indices, based on 
        the shape of other parameters providing load values (see 'vals'). 

    vals : np.ndarray([float]), optional
        1d or 2d numpy array of floats, that specify load values 
        imposed on dofs specified with 'inds'. If 'inds' is None, we assume
        that the provided definition covers all dofs in the model.
        If 1d and indices are specified, we assume that they refer to global dof indices. 
        If 2d and indices are specified, we assume that they refer to node indices and
        that every node in the model has the same number of dofs.

    N : int, optional
        The overall number of dofs in the model. If not provided, we assume that
        it equals the highest index in 'inds' + 1 (this assumes zero-based indexing).

    Notes
    -----
    Call 'fem_load_vector' on the same arguments to get the vector 
    itself as a numpy array. 


    Returns
    -------
    (inds, vals, N) or None
        The load values as floats, their indices and the overall
        size of the equation system.

    """

    if isfloatarray(vals):
        size = len(vals.shape)
        if size == 1:
            # dof based definition
            if inds is None:
                N = len(vals)
                inds = np.arange(N)
            else:
                assert isintegerarray(inds)
                assert len(vals) == len(inds)
        elif size == 2:
            # node based definition
            if inds is None:
                inds = np.arange(len(vals))  # node indices
            else:
                assert isintegerarray(inds)
                assert len(vals) == len(inds)
            # transform to nodal defintion
            inds, vals = nodes2d_to_dofs1d(inds, vals)

    if is1dintarray(inds) and is1dfloatarray(vals):
        N = inds.max() + 1 if N is None else N
        return inds, vals, N


def fem_load_vector(*args, **kwargs) -> np.ndarray:
    """
    Returns the sparse, COO format penalty matrix, equivalent of 
    a Courant-type penalization of the essential boundary conditions.

    Parameters
    ----------
        See the documentation of 'fem_ebc_data' for the discription
        of the possible arguments.

    Returns
    -------
    numpy.ndarray(float)[:]
        The load vector as a numpy array of floats.
    """
    inds, vals, N = fem_nbc_data(*args, **kwargs)
    f = np.zeros(N)
    f[inds] = vals
    return f


def fem_ebc_data(*args, inds: np.ndarray = None, pen: np.ndarray = None, N: int = None,
                 pfix: float = 1e12, **kwargs):
    """
    Returns the data necessary to create a penalty matrix for a Courant-type 
    penalization of the essential boundary conditions.

    Parameters
    ----------

    inds : np.ndarray(int), optional
        1d numpy integer array specifying node or dof indices, based on 
        the shape of other parameters providing penalty values (see 'pen'). 
        If penalty values are not provided, the specified dofs are penalized
        with value 'pfix'.

    pen : np.ndarray([float, bool]), optional
        1d or 2d numpy array of floats or booleans, that specify penalties 
        imposed on dofs specified with 'inds'. If 'inds' is None, we assume
        that the provided definition covers all dofs in the model.
        If 1d and indices are specified, we assume that they refer to global dof indices. 
        If 2d and indices are specified, we assume that they refer to node indices and
        that every node in the model has the same number of dofs.

    N : int, optional
        The overall number of dofs in the model. If not provided, we assume that
        it equals the highest index in 'inds' + 1 (this assumes zero-based indexing).

    pfix : float, optional
        Penalty value for fixed dofs. It is used to transform boolean penalty
        data, or to make up for missing values (e.g. only indices are provided).
        Default value is 1e+12.

    Notes
    -----
    Call 'fem_penalty_matrix_coo' on the same arguments to get the penalty 
    matrix itself. 

    Returns
    -------
    (inds, pen, N) or None
        The penalty values as floats, their indices and the overall
        size of the equation system.

    """

    if isinstance(pen, np.ndarray):
        size = len(pen.shape)
        if isfloatarray(pen):
            if size == 1:
                if inds is None:
                    N = len(pen)
                    inds = np.arange(N)
                else:
                    assert isintegerarray(inds)
                    assert len(pen) == len(inds)
            elif size == 2:
                if inds is None:
                    inds = np.arange(len(pen))
                else:
                    assert isintegerarray(inds)
                    assert len(pen) == len(inds)
                inds, pen = nodes2d_to_dofs1d(inds, pen)
        elif isboolarray(pen):
            if size == 1:
                pass
            elif size == 2:
                pen = bool_to_float(pen, pfix)
                if inds is None:
                    inds = np.arange(len(pen))
                else:
                    assert isintegerarray(inds)
                    assert len(pen) == len(inds)
                inds, pen = nodes2d_to_dofs1d(inds, pen)

    if is1dintarray(inds) and is1dfloatarray(pen):
        N = inds.max() + 1 if N is None else N
        return inds, pen, N


def fem_penalty_matrix_coo(*args, **kwargs) -> coo_matrix:
    """
    Returns the sparse, COO format penalty matrix, equivalent of 
    a Courant-type penalization of the essential boundary conditions.

    Parameters
    ----------
        See the documentation of 'fem_ebc_data' for the discription
        of the possible arguments.

    Returns
    -------
    scipy.sparse.coo_matrix
        The penalty matrix in sparse COO format.
    """
    inds, pen, N = fem_ebc_data(*args, **kwargs)
    K = coo_matrix((pen, (inds, inds)), shape=(N, N))
    K.eliminate_zeros()
    return K





if __name__ == '__main__':
    inds = np.array([2, 4, 12])
    pen = np.array([1e5, 1e5, 1e12])
    N = 100
    args = fem_ebc_data(inds=inds, pen=pen, N=N)
