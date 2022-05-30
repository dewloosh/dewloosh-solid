# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange
import numpy as np
from numpy import ndarray, swapaxes as swap, ascontiguousarray as ascont
from scipy.interpolate import interp1d

from dewloosh.geom.config import __haspyvista__, __hasplotly__, __hasmatplotlib__

from ..mesh import FemMesh
from ..cells.bernoulli2 import Bernoulli2
from ..cells.bernoulli3 import Bernoulli3

from .tr import \
    transform_numint_forces_out as tr_cell_gauss_out, \
    transform_numint_forces_in as tr_cell_gauss_in


__cache = True


if __haspyvista__:
    import pyvista as pv


class LineMesh(FemMesh):
    
    _cell_classes_ = {
        2: Bernoulli2,
        3: Bernoulli3,
    }

    def __init__(self, *args, areas=None, connectivity=None, **kwargs):          
                
        super().__init__(*args, **kwargs)
        
        if self.celldata is not None:
            nE = len(self.celldata)
            if areas is None:
                areas = np.ones(nE)
            else:
                assert len(areas.shape) == 1, \
                    "'areas' must be a 1d float or integer numpy array!"
            self.celldata.db['areas'] = areas
            
            if connectivity is not None:
                if isinstance(connectivity, np.ndarray):
                    assert len(connectivity.shape) == 3
                    assert connectivity.shape[0] == nE
                    assert connectivity.shape[1] == 2
                    assert connectivity.shape[2] == self.__class__.NDOFN
                    self.celldata.db['conn'] = connectivity
    
    def masses(self, *args, **kwargs):
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        vmap = map(lambda b: b.celldata.masses(), blocks)
        return np.concatenate(list(vmap))
    
    def mass(self, *args, **kwargs):
        return np.sum(self.masses(*args, **kwargs))
               
    def plot(self, *args, as_tubes=True, radius=0.1, **kwargs):
        if not as_tubes:
            return super().plot(*args, **kwargs)
        else:
            self.to_pv(as_tubes=True, radius=radius).plot(smooth_shading=True)

    def to_pv(self, *args, as_tubes=True, radius=0.1, **kwargs):
        """
        Returns the mesh as a `pyvista` object.
        """
        assert __haspyvista__
        if not as_tubes:
            return super().to_pv(*args, **kwargs)
        else:
            poly = pv.PolyData()
            poly.points = self.coords()
            topo = self.topology()
            lines = np.full((len(topo), 3), 2, dtype=int)
            lines[:, 1:] = topo
            poly.lines = lines
            return poly.tube(radius=radius)

    def to_plotly(self):
        assert __hasplotly__
        raise NotImplementedError

    def to_mpl(self):
        """
        Returns the mesh as a `matplotlib` figure.
        """
        assert __hasmatplotlib__
        raise NotImplementedError


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_distribution_factors(topo: ndarray, volumes: ndarray):
    """
    Determines the share of an element of a value at its nodes.

    The j-th factor of the i-th row is the contribution of
    element i to the j-th node. Assumes a regular topology.
    """
    factors = np.zeros(topo.shape, dtype=volumes.dtype)
    nodal_volumes = np.zeros(topo.max() + 1, dtype=volumes.dtype)
    for iE in range(topo.shape[0]):
        nodal_volumes[topo[iE]] += volumes[iE]
    for iE in prange(topo.shape[0]):
        for jNE in prange(topo.shape[1]):
            factors[iE, jNE] = volumes[iE] / nodal_volumes[topo[iE, jNE]]
    return factors


@njit(nogil=True, parallel=True, cache=__cache)
def _distribute_nodal_data_ndf_(data: ndarray, topo: ndarray, ndf: ndarray):
    """
    data (N, nRHS, nDOF)
    topo (nE, nNE)
    ---
    (nE, nNE, nRHS, nDOF)
    """
    _, nRHS, nDOF = data.shape
    nE, nNE = topo.shape
    res = np.zeros((nE, nNE, nRHS, nDOF))
    for i in prange(nE):
        for j in prange(nNE):
            for k in prange(nRHS):
                res[i, j, k] = data[topo[i, j], k] * ndf[i, j]
    return res


def distribute_nodal_data(data: ndarray, topo: ndarray, ndf: ndarray = None):
    ndf = np.ones_like(topo) if ndf is None else ndf
    return _distribute_nodal_data_ndf_(data, topo, ndf)


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _collect_nodal_data_ndf_(data: ndarray, topo: ndarray, ndf: ndarray, N: int):
    """
    data (nE, nNE, nRHS, nDOFN)
    topo (nE, nNE)
    ---
    (N, nRHS, nDOFN)
    """
    nE, nNE, nRHS, nDOFN = data.shape
    res = np.zeros((N, nRHS, nDOFN), dtype=data.dtype)
    for i in prange(nE):
        for j in prange(nNE):
            for k in prange(nRHS):
                res[topo[i, j], k] += data[i, j, k] * ndf[i, j]
    return res


def collect_nodal_data(data: ndarray, topo: ndarray, *args,
                       ndf: ndarray = None, N: int = None, **kwargs):
    N = topo.max() + 1 if N is None else N
    ndf = np.ones_like(topo) if ndf is None else ndf
    return _collect_nodal_data_ndf_(data, topo, ndf, N)


def extrapolate_gauss_data(gpos: ndarray, gdata: ndarray):
    gdata = swap(gdata, 0, 2)  # (nDOFN, nE, nP) --> (nP, nE, nDOFN)
    approx = interp1d(gpos, gdata, fill_value='extrapolate', axis=0)

    def inner(*args, **kwargs):
        res = approx(*args, **kwargs)
        return swap(res, 0, 2)  # (nP, nE, nDOFN) --> (nDOFN, nE, nP)
    return inner


class BernoulliFrame(LineMesh):
    
    NDOFN = 6
    
    """def internal_forces(self, *args, key=None, **kwargs):
        key = 'internal_forces' if key is None else key
        return super().internal_forces(*args, key=key, **kwargs)"""

    """def postprocess(self, *args, smoothen=True, **kwargs):
        self.internal_forces(store=True, key='internal_forces', integrate=False)
        if smoothen:
            self.smoothen_internal_forces(key='internal_forces')
        pass"""

    def smoothen_internal_forces(self, key='internal_forces'):
        """
        The implementation assumed a regular and tight topology and that
        the frames have a uniform material distribution.
        """
        topo = self.topology()
        nE, nNE = topo.shape
        def f(b): return b.celldata.frames.to_numpy()
        frames = np.vstack(list(map(f, self.cellblocks(inclusive=True))))
        N = (topo.max() + 1) * self.NDOFN
        # inperpolate with fitting polynomial of smoothed nodal values
        ndf = self.nodal_distribution_factors(store=False, measure='uniform')
        # we need to transform all values to the global frame, so we can add them
        # (nDOF, nE, nNE, nRHS) --> (nE, nNE, nRHS, nDOF)
        data = self.internal_forces(key=key)  # (nE, nEVAB, nRHS)
        # (nE, nNE, nDOF, nRHS)
        data = data.reshape(nE, nNE, self.NDOFN, data.shape[-1])
        data = swap(data, 2, 3)  # (nE, nNE, nRHS, nDOFN)
        data[:, :, :, :3] = tr_cell_gauss_out(
            ascont(data[:, :, :, :3]), frames)
        data[:, :, :, 3:] = tr_cell_gauss_out(
            ascont(data[:, :, :, 3:]), frames)
        # the following line calculates nodal values as the weighted average of cell-nodal data,
        # the weights are the nodal distribution factors
        data = collect_nodal_data(data, topo, ndf=ndf, N=N)  # (N, nRHS, nDOF)
        # the following line redistributes nodal data to the cells (same nodal value for neighbouring cells)
        data = distribute_nodal_data(data, topo)  # (nE, nNE, nRHS, nDOF)
        # transform all values back to the local frames of the elements
        data[:, :, :, :3] = tr_cell_gauss_in(ascont(data[:, :, :, :3]), frames)
        data[:, :, :, 3:] = tr_cell_gauss_in(ascont(data[:, :, :, 3:]), frames)
        data = swap(data, 2, 3)  # -> (nE, nNE, nDOF, nRHS)
        # (nE, nNE * nDOF, nRHS)
        data = data.reshape(nE, nNE * self.NDOFN, data.shape[-1])
        # store results
        # data : (nE, nNE * nDOF, nRHS)
        self.internal_forces(store=data, key=key)


if __name__ == '__main__':
    pass
