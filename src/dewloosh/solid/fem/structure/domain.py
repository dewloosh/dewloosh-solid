from pyoneer.mech.fem.structure.structure import Structure
from pyoneer.mech.fem.mesh2d import FemMesh2d


class Domain(Structure):
    
    def __init__(self, *args, mesh: FemMesh2d = None, **kwargs):
        super().__init__(wrap=mesh)
        