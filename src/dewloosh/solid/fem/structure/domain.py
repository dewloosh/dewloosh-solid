# -*- coding: utf-8 -*-
from .structure import Structure
from ..mesh2d import FemMesh2d


class Domain2d(Structure):

    def __init__(self, *args, mesh: FemMesh2d = None, **kwargs):
        super().__init__(wrap=mesh)
