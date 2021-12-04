# -*- coding: utf-8 -*-
from pyoneer.math.numint import GaussPoints as Gauss
from pyoneer.mech.fem.model.solid3d import Solid3d
from pyoneer.mesh.H8 import H8 as HexaHedron
from pyoneer.mech.fem.elem import FiniteElement


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
