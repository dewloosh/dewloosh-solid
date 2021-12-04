# -*- coding: utf-8 -*-
from dewloosh.solid.fem.mesh import FemMesh
from dewloosh.geom.utils import frame_of_plane
from dewloosh.math.linalg.csys import CoordinateSystem as CS
import numpy as np


class FemMesh2d(FemMesh):
    
    def __init__(self, *args, coords=None, csys:CS=None, **kwargs):
        if len(coords.shape) == 2:
            coords3d = np.zeros((coords.shape[0], 3), dtype=coords.dtype)
            coords3d[:, :2] = coords[:, :2]
            coords = coords3d
        if csys is not None:
            tr = csys.trMatrix()
            # transform coordinates to global
        else:
            center, tr = frame_of_plane(coords)
            csys = CS(origo=center, axes=tr)
        self.csys = csys
        super().__init__(*args, coords=coords, **kwargs)
        