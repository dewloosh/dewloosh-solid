# -*- coding: utf-8 -*-
from dewloosh.solid.fem.mesh import FemMesh
from dewloosh.geom.space.utils import frame_of_plane
from dewloosh.math.linalg.frame import ReferenceFrame as FrameLike
import numpy as np


class FemMesh2d(FemMesh):
    
    def __init__(self, *args, coords=None, frame:FrameLike=None, **kwargs):
        if len(coords.shape) == 2:
            coords3d = np.zeros((coords.shape[0], 3), dtype=coords.dtype)
            coords3d[:, :2] = coords[:, :2]
            coords = coords3d
        if frame is not None:
            tr = frame.trMatrix()
            # transform coordinates to global
        else:
            center, tr = frame_of_plane(coords)
            frame = FrameLike(origo=center, axes=tr)
        super().__init__(*args, coords=coords, frame=frame, **kwargs)
        