# -*- coding: utf-8 -*-
from dewloosh.solid.fem.model.solid3d import Solid3d
from dewloosh.geom.TET4 import TET4 as Tetra
from dewloosh.solid.fem.elem import FiniteElement
import numpy as np


class TET4(Tetra, Solid3d, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': (np.array([[1/3, 1/3, 1/3]]), np.array([1/6])),
    }


if __name__ == '__main__':
    pass