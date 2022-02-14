# -*- coding: utf-8 -*-
import sympy as sy
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict

from dewloosh.core.tools.kwargtools import getasany

from dewloosh.math.function import Function
from dewloosh.math.optimize import BinaryGeneticAlgorithm as BGA
from dewloosh.math.linalg.tensor3333 import ComplianceTensor

from .sym import smat_sym_ortho_3d
from .utils import standard_keys_ortho, group_mat_params, keys_ortho_all


class Hooke3D(ComplianceTensor):

    _standard_keys_ = standard_keys_ortho

    def __init__(self, *args, density : float=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.constants = defaultdict(lambda : None)
        self.density = density
        self.update(**kwargs)

    @classmethod
    def symbolic(cls, imap : dict=None, **subs):
        imap = cls.__imap__ if imap is None else imap
        return smat_sym_ortho_3d(imap=imap, **subs)

    def modified(self):
        S = self.symbolic()
        subs = {s : self.constants[str(s)] for s in S.free_symbols}
        S = S.subs([(sym, val) for sym, val in subs.items()])
        S = np.array(S, dtype=np.float)
        self._array = np.linalg.inv(S)
        self.collapsed = True

    def lambdify(self):
        S = self.__class__.symbolic()
        C = S.inv()
        self._args = [str(s) for s in S.free_symbols]
        self._sfunc = sy.lambdify(self._args, S)
        self._cfunc = sy.lambdify(self._args, C)

    def evalf(self):
        return self._cfunc(*[self.constants[key]
                             for key in self._args])

    def is_defined(self):
        raise NotImplementedError
        #return is_proper_ortho(**self.constants)

    def update(self, **kwargs):
        E, NU, G = group_mat_params(**kwargs)
        nE, nNU, nG = len(E), len(NU), len(G)
        nParams = sum([nE, nNU, nG])

        #to avoid TypeError due to multiple keyword arguments
        for k in keys_ortho_all:
            if k in kwargs:
                del kwargs[k]

        if not self.is_defined():
            if nParams == 0:
                return None
            elif nParams > 9 and nParams <= 12:
                # overdetermined ortho 3D
                raise NotImplementedError
            elif nParams == 9:
                self.handle_proper_ortho_3D(**E, **NU, **G)
            elif nParams > 6 and nParams < 9:
                # overdetermined trans-iso 3D or underdetermined ortho 3D
                raise NotImplementedError
            elif nParams == 6:
                # nE == 2, nNU == 2, nG == 2
                raise NotImplementedError
            elif nParams == 5:
                if nG == 2:
                    isoplane = kwargs.get('isoplane', '23')
                    self.handle_proper_triso_2D_lamina(**E, **NU, **G,
                                                       isoplane=isoplane)
                else:
                    raise NotImplementedError
            elif nParams == 4:
                # overdetermined iso 2D or 3D
                raise NotImplementedError
            elif nParams == 3:
                pass
            elif nParams == 2:
                # proper iso 2D or 3D
                self.handle_proper_iso_3D(**E, **NU, **G)
            elif nParams == nE == 1:
                # proper 1D
                raise NotImplementedError
            else:
                raise NotImplementedError
        self._complete()
        self.modified()

    def handle_proper_ortho_3D(self, **kwargs):
        self.constants.update(**kwargs)
        # all shear moduli must be defined
        self.constants['G12'] = kwargs.get('G12', kwargs.get('G21', None))
        self.constants['G13'] = kwargs.get('G13', kwargs.get('G31', None))
        self.constants['G23'] = kwargs.get('G23', kwargs.get('G32', None))

    def handle_proper_triso_2D_lamina(self, isoplane = '23', **kwargs):
        if '1' not in isoplane:
            # 2 from the 3 shear moduli must be defined
            G23 = getasany(['G23', 'G32'], None, **kwargs)
            G13 = getasany(['G13', 'G31', 'G12', 'G21'], None, **kwargs)
            G12 = G13
            # three from the the rest must be defined
            NU12 = getasany(['NU12', 'NU13'], None, **kwargs)
            NU21 = getasany(['NU21', 'NU31'], None, **kwargs)
            NU23 = getasany(['NU23', 'NU32'], None, **kwargs)
            E2 = getasany(['E2', 'E3'], None, **kwargs)
            E1 = kwargs.get('E1', None)
            NU32 = NU23
            NU13 = NU12
            NU31 = NU21
            E3 = E2
        else:
            raise NotImplementedError
        self.constants['E1'] = E1
        self.constants['E2'] = E2
        self.constants['E3'] = E3
        self.constants['NU12'] = NU12
        self.constants['NU13'] = NU13
        self.constants['NU23'] = NU23
        self.constants['NU21'] = NU21
        self.constants['NU31'] = NU31
        self.constants['NU32'] = NU32
        self.constants['G12'] = G12
        self.constants['G13'] = G13
        self.constants['G23'] = G23

    def handle_proper_triso_2D_membrane(self, isoplane = '23', **kwargs):
        if '1' not in isoplane:
            # three from the the rest must be defined
            NU12 = getasany(['NU12', 'NU13'], None, **kwargs)
            NU21 = getasany(['NU21', 'NU31'], None, **kwargs)
            E2 = getasany(['E2', 'E3'], None, **kwargs)
            E1 = kwargs.get('E1', None)
            if NU21 is None:
                NU21 = NU12 * E2 / E1
            elif NU12 is None:
                NU12 = NU21 * E1 / E2
            elif E1 is None:
                E1 = NU12 * E2 / NU21
            elif E2 is None:
                E2 = NU21 * E1 / NU12
            E3 = E2
            NU32 = NU23 = np.sqrt(NU12 * NU21)
            NU13 = NU12
            NU31 = NU21
        else:
            raise NotImplementedError
        self.constants['E1'] = E1
        self.constants['E2'] = E2
        self.constants['E3'] = E3
        self.constants['NU12'] = NU12
        self.constants['NU13'] = NU13
        self.constants['NU23'] = NU23
        self.constants['NU21'] = NU21
        self.constants['NU31'] = NU31
        self.constants['NU32'] = NU32
        self.constants['G12'] = np.inf
        self.constants['G13'] = np.inf
        self.constants['G23'] = np.inf

    def handle_proper_iso_3D(self, **kwargs):
        if 'E' in kwargs:
            E = kwargs['E']
            if 'NU' in kwargs:
                NU = kwargs['NU']
                G = E/2/(1+NU)
            elif 'G' in kwargs:
                G = kwargs['G']
                NU = E/2/G - 1
        else:
            assert 'NU' in kwargs
            assert 'G' in kwargs
            NU = kwargs['NU']
            G = kwargs['G']
            E = 2*G*(1+NU)

        #assert that shear or hydrostatic loading does not produce
        #negative strain energy
        assert NU > -1
        assert NU < 0.5

        self.constants['E1'] = E
        self.constants['E2'] = E
        self.constants['E3'] = E
        self.constants['NU12'] = NU
        self.constants['NU13'] = NU
        self.constants['NU23'] = NU
        self.constants['NU21'] = NU
        self.constants['NU31'] = NU
        self.constants['NU32'] = NU
        self.constants['G12'] = G
        self.constants['G13'] = G
        self.constants['G23'] = G

    def _complete(self, method = 'NM'):
        """
        Given a minimal number of necessary known parameters, the function tries
        to infer the missing parameters from the known ones.
        """
        if self.is_defined():
            self.constants['NU21'] = self.constants['NU12'] * \
                self.constants['E2'] / self.constants['E1']
            self.constants['NU31'] = self.constants['NU13'] * \
                self.constants['E3'] / self.constants['E1']
            self.constants['NU32'] = self.constants['NU23'] * \
                self.constants['E3'] / self.constants['E2']
            return
        S = Hooke3D.symbolic()
        S_skew = S - S.T
        params = [(sym,val) for sym,val in self.constants.items()
                  if val is not None]
        residual = S_skew.subs(params)
        error = residual.norm()
        f = Function(error)
        v = f.variables

        if method == 'GA':
            ranges = [[-1.0, 1.0] for _ in range(len(v))]
            GA = BGA(f, ranges, length = 8, nPop = 140)
            r = GA.solve()
            self.constants.update({str(sym) : val for sym, val in zip(v, r)})
        elif method == 'NM':
            x0 = np.zeros((len(v),))
            res = minimize(f, x0, method = 'Nelder-Mead', tol = 1e-6)
            self.constants.update({str(sym) : val
                                   for sym, val in zip(v, res.x)})

    def stiffness_matrix(self):
        return self._array * self.density

    def compliance_matrix(self):
        return np.linalg.inv(self._array)

    def stresses_from_strains(self, strains : np.ndarray) -> np.ndarray:
        return self.stiffness_matrix() @ strains

    def strains_from_stresses(self, stresses : np.ndarray) -> np.ndarray:
        return self.compliance_matrix() @ stresses

    def rotZ(self, angle : float, deg=True):
        if deg:
            return self.rotate('Body', [0, 0, angle*np.pi/180], 'XYZ')
        else:
            return self.rotate('Body', [0, 0, angle], 'XYZ')


if __name__ == '__main__':
    from dewloosh.math.linalg.frame import ReferenceFrame

    """
    Test isotropic case.
    """
    A = ReferenceFrame()
    B = A.orient_new('Body', [0, 0, 90*np.pi/180],  'XYZ')

    E, nu = 1000, 0.4
    params = {
        'E1' : E,
        'E2' : E,
        'E3' : E,
        'NU12' : nu,
        'NU31' : nu,
        'NU23' : nu,
        'G12' : E/(2*(1+nu)),
        'G31' : E/(2*(1+nu)),
        'G23' : E/(2*(1+nu))
    }
    HookeA = Hooke3D(frame = A, **params)
    HookeB = HookeA.rotate('Body', [0, 0, 90*np.pi/180],  'XYZ')

    """
    Test orthotropic case (timber).
    """
    A = ReferenceFrame()
    B = A.orient_new('Body', [0, 0, 90*np.pi/180],  'XYZ')

    E, nu = 1000, 0.4
    params = {
        'E1' : 15700,
        'E2' : 1060,
        'E3' : 780,
        'NU12' : 0.29,
        'NU31' : 0.022,
        'NU23' : 0.39,
        'G12' : 880,
        'G31' : 880,
        'G23' : 88
    }
    HookeA = Hooke3D(frame = B, **params)
    HookeA.transform_to_frame(A)

    e11, e22, e33 = sy.symbols(r'\epsilon_1 \epsilon_2 \epsilon_3')
    g12, g23, g13 = sy.symbols(r'\gamma_{12} \gamma_{23} \gamma_{13}')
    strains = [e11, e22, e33, g23, g13, g12]
    C = Hooke3D.symbolic().inv()
    C3 = C[2,:]
    s33 = sum([ci*si for ci, si in zip(C3, strains)])

    kdef = 0.8
    xlam = {
        'E1' : 1100/(1+kdef),
        'E2' : 55/(1+kdef),
        'E3' : 91.66/(1+kdef),
        'G12' : 60.0/(1+kdef),
        'G13' : 69.0/(1+kdef),
        'G23' : 6.9/(1+kdef),
        'NU12' : 0.4,
        'NU13' : 0.35,
        'NU23' : 0.25,
        }
    N0 = ReferenceFrame()
    N90 = N0.orient_new('Body', [0, 0, 90*np.pi/180], 'XYZ')
    XLAM0 = Hooke3D(frame = N0, **xlam)
    XLAMD90 = Hooke3D(frame = N90, **xlam).transform_to_frame(N0)
    XLAMR90 = XLAM0.rotate('Body',
                           [0, 0, 90*np.pi/180], 'XYZ').transform_to_frame(N0)
