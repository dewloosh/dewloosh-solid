# -*- coding: utf-8 -*-
from pyoneer.mech.fem.elem import FiniteElement
from pyoneer.mesh.T3 import T3 as Triangle
from pyoneer.mech.fem.model.membrane import Membrane
from pyoneer.mech.fem.model.plate import Plate
import numpy as np


class CSTM(Triangle, Membrane, FiniteElement):
    """
    The constant-strain triangle (a.k.a., CST triangle, Turner triangle)
    """

    qrule = 'full'
    quadrature = {
        'full': (np.array([[1/3, 1/3]]), np.array([1/2])),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CSTP(Triangle, Plate, FiniteElement):
    """
    The constant-strain triangle (a.k.a., CST triangle, Turner triangle)
    """

    qrule = 'full'
    quadrature = {
        'full': (np.array([[1/3, 1/3]]), np.array([1/2])),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    pass
