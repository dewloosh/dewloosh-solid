# -*- coding: utf-8 -*-
from dewloosh.math.numint import GaussPoints as Gauss
from dewloosh.solid.fem.model.solid3d import Solid3d
from dewloosh.geom.polyhedron import TriquadraticHexaHedron
from dewloosh.solid.fem.elem import FiniteElement


class H27(FiniteElement, Solid3d, TriquadraticHexaHedron):

    qrule = 'selective'
    quadrature = {
        'full': Gauss(3, 3, 3),
        'selective': {
            (0, 1, 2): 'full',
            (3, 4, 5): 'reduced'
        },
        'reduced': Gauss(2, 2, 2)
    }


if __name__ == '__main__':
    pass
