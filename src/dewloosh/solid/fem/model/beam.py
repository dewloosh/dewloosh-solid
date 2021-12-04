# -*- coding: utf-8 -*-
from pyoneer.mech.fem.model.solid import Solid
from pyoneer.mesh.utils import cell_coords_bulk


class BernoulliBeam(Solid):
    
    NDOFN = 6
    NSTRE = 4    
        
    def local_coordinates(self, *args, topo=None, **kwargs):
        topo = self.nodes.to_numpy() if topo is None else topo
        coords = self.pointdata.x.to_numpy()
        return cell_coords_bulk(coords, topo)
    
    def model_stiffness_matrix(self, *args, **kwargs):
        return self.material_stiffness_matrix()
    
    
