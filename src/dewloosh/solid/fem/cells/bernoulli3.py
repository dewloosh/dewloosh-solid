# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from collections import Iterable
from typing import Union

from dewloosh.math.numint import GaussPoints as Gauss
from dewloosh.math.array import atleast1d
from dewloosh.math.utils import to_range

from dewloosh.geom.cells import QuadraticLine as Line

from .bernoulli import BernoulliBase
from .utils.bernoulli import global_shape_function_derivatives_bulk as gdshpB

from .gen.b3 import shape_function_values_bulk as shpB3, \
    shape_function_derivatives_bulk as dshpB3


__all__ = ['Bernoulli3']


ArrayOrFloat = Union[ndarray, float]


class Bernoulli3(Line, BernoulliBase):

    qrule = 'full'
    quadrature = {
        'full': Gauss(6)
    }

    def shape_function_values(self, pcoords: ArrayOrFloat, *args,
                              rng: Iterable = None, lengths=None,
                              **kwargs) -> ndarray:
        """
        ---
        (nE, nP, nNE=3, nDOF=6)
        """
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords) if isinstance(
            pcoords, list) else pcoords)
        pcoords = to_range(pcoords, source=rng, target=[-1, 1])
        lengths = self.lengths() if lengths is None else lengths
        return shpB3(pcoords, lengths).astype(float)

    def shape_function_derivatives(self, pcoords=None, *args, rng=None,
                                   jac: ndarray = None, dshp: ndarray = None,
                                   lengths=None, **kwargs) -> ndarray:
        """
        ---
        (nE, nP, nNE=3, nDOF=6, 3)
        """
        lengths = self.lengths() if lengths is None else lengths
        if pcoords is not None:
            # calculate derivatives wrt. the parametric coordinates in the range [-1, 1]
            pcoords = atleast1d(np.array(pcoords) if isinstance(
                pcoords, list) else pcoords)
            rng = np.array([-1, 1]) if rng is None else np.array(rng)
            pcoords = to_range(pcoords, source=rng, target=[-1, 1])
            dshp = dshpB3(pcoords, lengths)
            # return derivatives wrt the local frame if jacobian is provided, otherwise
            # return derivatives wrt. the parametric coordinate in the range [-1, 1]
            return dshp.astype(float) if jac is None else gdshpB(dshp, jac).astype(float)
        elif dshp is not None and jac is not None:
            # return derivatives wrt the local frame
            return gdshpB(dshp, jac).astype(float)
