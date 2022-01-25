# -*- coding: utf-8 -*-
from dewloosh.math.numint import GaussPoints as Gauss
from dewloosh.solid.fem.model.solid3d import Solid3d
from dewloosh.geom.H8 import H8 as HexaHedron
from dewloosh.solid.fem.elem import FiniteElement


class H8(HexaHedron, Solid3d, FiniteElement):

    qrule = 'selective'
    quadrature = {
        'full': Gauss(2, 2, 2),
        'selective': {
            (0, 1, 2): 'full',
            (3, 4, 5): 'reduced'
        },
        'reduced': Gauss(1, 1, 1)
    }


if __name__ == '__main__':
    pass
