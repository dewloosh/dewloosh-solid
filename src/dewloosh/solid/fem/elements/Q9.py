# -*- coding: utf-8 -*-
from pyoneer.mesh.Q9 import Q9 as Quadrilateral
from pyoneer.mech.fem.elem import FiniteElement
from pyoneer.math.numint import GaussPoints as Gauss
from pyoneer.mech.fem.model.membrane import Membrane
from pyoneer.mech.fem.model.plate import Plate


class Q9M(Quadrilateral, Membrane, FiniteElement):

    qrule = 'full'
    quadrature = {
        'full': Gauss(3, 3),
        'selective': {
            (0, 1): 'full',
            (2,): 'reduced'
        },
        'reduced': Gauss(2, 2)
    }


class Q9P(Quadrilateral, Plate, FiniteElement):

    qrule = 'selective'
    quadrature = {
        'full': Gauss(3, 3),
        'selective': {
            (0, 1, 2): 'full',
            (3, 4): 'reduced'
        },
        'reduced': Gauss(2, 2)
    }
