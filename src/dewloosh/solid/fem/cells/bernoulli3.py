# -*- coding: utf-8 -*-
from dewloosh.math.numint import GaussPoints as Gauss

from dewloosh.geom.cells import QuadraticLine as Line

from .bernoulli import BernoulliBase

from .gen.b3 import shape_function_values_bulk as shpB3, \
    shape_function_derivatives_bulk as dshpB3


__all__ = ['Bernoulli3']


class Bernoulli3(Line, BernoulliBase):

    qrule = 'full'
    quadrature = {
        'full': Gauss(6)
    }
    shpfnc = shpB3
    dshpfnc = dshpB3
