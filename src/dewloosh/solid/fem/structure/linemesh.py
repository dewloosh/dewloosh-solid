# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.geom.config import __haspyvista__, __hasplotly__, __hasmatplotlib__

from ..mesh import FemMesh
from ..cells.bernoulli2 import Bernoulli2
from ..cells.bernoulli3 import Bernoulli3

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


class BernoulliFrame(LineMesh):
    
    NDOFN = 6
    

if __name__ == '__main__':
    pass
