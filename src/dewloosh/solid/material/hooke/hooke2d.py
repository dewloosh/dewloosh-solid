# -*- coding: utf-8 -*-
import numpy as np

from .hooke import Hooke3D


class Hooke2D(Hooke3D):

    __imap__ = {0 : (0, 0), 1 : (1, 1), 2 : (2, 2),
                5 : (1, 2), 4 : (0, 2), 3 : (0, 1)}

    def expand(self):
        S = np.array(Hooke3D.symbolic(**self.constants), \
            dtype=self._array.dtype)
        self._array = np.linalg.inv(S)
        return super().expand()


class HookeMembrane(Hooke2D):
    """
    Hooke's law for membranes.
    """

    @classmethod
    def symbolic(cls, **subs):
        S = Hooke2D.symbolic()
        S.row_del(2)
        S.row_del(2)
        S.row_del(2)
        S.col_del(2)
        S.col_del(2)
        S.col_del(2)
        if len(subs) > 0:
            S = S.subs([(sym, val) for sym, val in subs.items()])
        return S

    def collapse(self):
        super().collapse()
        inds = [3, 4, 5]
        self._array = np.delete(np.delete(self._array, inds, axis = 0),
                                inds, axis = 1)
        return self


class HookePlate(Hooke2D):
    """
    Hooke's law for shells.
    """

    @classmethod
    def symbolic(cls, **subs):
        S = Hooke2D.symbolic()
        S.row_del(2)
        S.col_del(2)
        if len(subs) > 0:
            S = S.subs([(sym, val) for sym, val in subs.items()])
        return S

    def collapse(self):
        super().collapse()
        self._array = np.delete(np.delete(self._array, 2, axis = 0),
                                2, axis = 1)
        return self


class HookeShell(HookePlate):
    """
    Hooke's law for shells.
    """


def Lamina(*args, plane_strain=False, stype='shell', **kwargs):
    if not plane_strain:
        if stype == 'membrane':
            return HookeMembrane(*args, **kwargs)
        elif stype == 'plate':
            return HookePlate(*args, **kwargs)
        elif stype == 'shell':
            return HookeShell(*args, **kwargs)
        else:
            return RuntimeError("Invalid surface type : {}".format(stype))
    else:
        raise NotImplementedError


if __name__ == '__main__':

    from dewloosh.math.linalg.frame import ReferenceFrame
    from dewloosh.math.linalg import Tensor3333 as Tensor
    import sympy as sy

    A = ReferenceFrame()
    B = A.orient_new('Body', [0, 0, 90 * np.pi/180],  'XYZ')

    kdef = 0.8
    xlam = {
        'E1' : 1100/(1+kdef),
        'E2' : 55/(1+kdef),
        'G12' : 60.0/(1+kdef),
        'G23' : 6.9/(1+kdef),
        'NU12' : 0.4,
        'stype' : 'membrane',
        }

    plank0 = Lamina(**xlam)
    plank0.expand()
    plank0.collapse()
    plank90 = plank0.rotZ(90)

    # symbolic transformed lamina
    sym = Hooke3D.symbolic()
    Tsym = Tensor(sym, A, 'sympy', dtype=object)
    angle = sy.symbols('theta')
    TsymR = Tsym.rotate('Body', [0, 0, angle], 'XYZ')
    M = sy.Matrix(TsymR.array)
    M.row_del(2)
    M.col_del(2)

    # membrane
    kdef = 0.8
    xlam = {
        'E1' : 1100 / (1 + kdef),
        'E2' : 55 / (1 + kdef),
        'G12' : 60.0 / (1 + kdef),
        'G23' : 6.9 / (1 + kdef),
        'NU12' : 0.4,
        }

    plank0 = HookeMembrane(**xlam)
    plank0.expand()
    plank0.collapse()
    plank90 = plank0.rotZ(90)
