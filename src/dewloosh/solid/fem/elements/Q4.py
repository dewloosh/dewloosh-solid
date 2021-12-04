# -*- coding: utf-8 -*-
from pyoneer.mesh.Q4 import Q4 as Quadrilateral
from pyoneer.mech.fem.elem import FiniteElement
from pyoneer.math.numint import GaussPoints as Gauss
from pyoneer.mesh.tri.triutils import area_tri_bulk
from pyoneer.mesh.utils import cell_coords_bulk
from pyoneer.mech.fem.model.membrane import Membrane
from pyoneer.mech.fem.model.plate import Plate
import numpy as np
from numba import njit
__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def area_Q4_bulk(ecoords : np.ndarray):
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    res += area_tri_bulk(ecoords[:, :3, :])
    res += area_tri_bulk(ecoords[:, np.array([0, 2, 3]), :])
    return res


class Q4M(Quadrilateral, Membrane, FiniteElement):
    
    qrule = 'selective'
    quadrature = {
        'full' : Gauss(2, 2), 
        'selective' : {
            (0, 1) : 'full',
            (2,) : 'reduced'
            },
        'reduced' : Gauss(1, 1)
        }
        
    def areas(self, *args, coords=None, topo=None, **kwargs):
        """This shadows the original geometrical implementation."""
        coords = self.pointdata.x.to_numpy() if coords is None else coords
        topo = self.nodes.to_numpy() if topo is None else topo
        return area_Q4_bulk(cell_coords_bulk(coords, topo))
    

class Q4P(Quadrilateral, Plate, FiniteElement):
    
    qrule = 'selective'
    quadrature = {
        'full' : Gauss(2, 2), 
        'selective' : {
            (0, 1, 2) : 'full',
            (3, 4) : 'reduced'
            },
        'reduced' : Gauss(1, 1)
        }
        
    def areas(self, *args, coords=None, topo=None, **kwargs):
        """This shadows the original geometrical implementation."""
        coords = self.pointdata.x.to_numpy() if coords is None else coords
        topo = self.nodes.to_numpy() if topo is None else topo
        self.local_coordinates(topo=topo)
        return area_Q4_bulk(cell_coords_bulk(coords, topo))
    
    
if __name__ == '__main__':
    from sympy import symbols

    r, s = symbols('r s')

    N1 = 0.125*(1-r)*(1-s)
    N2 = 0.125*(1+r)*(1-s)
    N3 = 0.125*(1+r)*(1+s)
    N4 = 0.125*(1-r)*(1+s)
