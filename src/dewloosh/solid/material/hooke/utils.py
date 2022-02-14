# -*- coding: utf-8 -*-
from scipy.optimize import minimize
from sympy import symbols, Eq, solve, false, true
from sympy.logic.boolalg import Boolean
import numpy as np

from dewloosh.core.tools.kwargtools import getasany

from dewloosh.math.function import Function

from .sym import smat_sym_ortho_3d


standard_keys_ortho = ['E1', 'E2', 'E3', 'NU12', 'NU13', 'NU23', 'G12', 'G13', 'G23']
keys_ortho_all = standard_keys_ortho + ['NU21', 'NU32', 'NU31']
bulk_keys_ortho = ['E1', 'E2', 'E3', 'NU12', 'NU13', 'NU23', 'NU21', 'NU32', 'NU31']


class HookeError(Exception): ...


def group_mat_params(**params) -> tuple:
    """
    Groups and returns all input material parameters as a 3-tuple
    of dictionaries.

    Returns
    -------
    tuple
        3 dictionaries for Young's moduli, Poisson's ratios 
        and shear moduli.
    """
    PARAMS = {key.upper() : value for key, value in params.items()}
    keys = list(PARAMS.keys())
    E = {key : PARAMS[key] for key in keys if key[0] == 'E'}
    NU = {key : PARAMS[key] for key in keys if key[:2] == 'NU'}
    G = {key : PARAMS[key] for key in keys if key[0] == 'G'}
    return E, NU, G


def missing_std_params_ortho(**params):
    """Returns the ones missing from the 9 standard orthotropic parameters."""
    return [k for k in standard_keys_ortho if k not in params]


def missing_params_ortho(**params):
    """Returns the ones missing from the 12 orthotropic parameters."""
    return [k for k in keys_ortho_all if k not in params]


def has_std_params_ortho(**params) -> bool:
    """
    Returns True, if all 9 standard keys are provided.
    Standard keys are the 3 Young's moduli, the 3 shear moduli
    and the minor Poisson's ratios.
    """
    return len(missing_std_params_ortho(**params)) == 0


def has_all_params_ortho(**params) -> bool:
    """ Returns True, if all 12 keys of an orthotropic material are provided."""
    return len(missing_params_ortho(**params)) == 0


def finalize_params_ortho(**params) -> dict:
    """
    Given at lest 3 bulk parameters (Ei, or NUij), the function 
    determines the remaining ones from symmetry conditions for
    a 3d orthotropic setup. Returns a dictionary that contains
    all 12 engineering constants.
    """
    if has_all_params_ortho(**params):
        return params
    E, NU, _ = group_mat_params(**params)
    subs = [(key, val) for key, val in {**E, **NU} if val is not None]
    assert len(subs) > 2, "There are too many missing parameters!"
    S = smat_sym_ortho_3d()[:3, :3]
    S_skew = S - S.T
    residual = S_skew.subs(subs)
    error = residual.norm()
    if len(error.free_symbols) == 0:
        return params
    f = Function(error)
    vars = f.variables
    x0 = np.zeros((len(vars),))
    res = minimize(f, x0, method='Nelder-Mead', tol=1e-6)
    params_ = {str(sym) : val for sym, val in zip(vars, res.x)}
    params.update(params_)
    return params


def finalize_params_triso(*args, isoplane='23', **params) -> dict:
    """
    Given exactly 5 independent parameters, the function determines the 
    remaining ones from symmetry conditions for a 3d transversely 
    isotropic setup. Returns a dictionary that contains all 12 
    engineering constants.
    """
    
    # get exactly 5 input parameters
    E, NU, G = group_mat_params(**params)
    str_to_sym = lambda k : symbols(k, real=True)
    subs = {str_to_sym(key) : val \
        for key, val in {**E, **NU, **G}.items() if val is not None}
    assert len(subs) == 5, "Exactly 5 independent material constants must be provided "\
        "for a transversely isotropic material!"

    E1, E2, E3, G23, G12, G13, NU12, NU21, NU13, NU31, NU23, NU32 = \
        symbols('E1 E2 E3 G23 G12 G13 NU12 NU21 NU13 NU31 NU23 NU32', real=True)
    eqs = []
    if '1' not in isoplane:
        _G13 = getasany(['G13', 'G31', 'G12', 'G21'], None, **params)
        assert _G13 is not None, "Out of plane shear modulus G13 (=G12) \
            must be provided as either G13, G31, G12 or G21!"
        eqs.append(Eq(E2 - E3, 0))
        eqs.append(Eq(NU12 - NU13, 0))
        eqs.append(Eq(NU21 - NU31, 0))
        eqs.append(Eq(NU23 - NU32, 0))
        eqs.append(Eq(G12 - G13, 0))
        eqs.append(Eq(G23 * 2 * (1 + NU23) - E2, 0))
        eqs.append(Eq(NU12*E2 - NU21*E1, 0))
        #eqs.append(Eq(NU13*E3 - NU31*E1, 0))  # not independent
        #eqs.append(Eq(NU23*E3 - NU32*E2, 0))  # not independent

    eqs = [eq.subs(subs) for eq in eqs]
    for eq in eqs:
        if isinstance(eq, Boolean):
            if eq is false:
                raise HookeError("There is contradiction in the input data!")
            elif eq is true:
                raise HookeError("Input parameters are not all independent!")
        else:
            assert isinstance(eq, Eq), "Unknown error in input data!"

    free_symbols = [eq.free_symbols for eq in eqs]
    vars = set.union(*free_symbols)
    sol = solve(eqs, vars, dict=True)[0]
    subs.update(sol)
    res = {str(key) : val for key, val in subs.items()}
    assert has_all_params_ortho(**res), "Unknown error!"
    return res


def get_iso_params(*args, **kwargs):
    """
    Returns all 12 orthotropic engineering constants for an 
    isotropic material. 
    Requires 2 independent constants to be provided.
    """
    KWARGS = {key.upper() : value for key, value in kwargs.items()}
    E = KWARGS.get('E', None)
    G = KWARGS.get('G', None)
    NU = KWARGS.get('NU', None)
    try:
        if E is None:
            E = 2*G*(1+NU)
        elif G is None:
            G = E/2/(1+NU)
        elif NU is None:
            NU = E/2/G - 1
    except TypeError:
        raise HookeError("At least 2 independent constants" \
            " must be defined for an isotropic material")
    params = \
        {'E1' : E, 'E2' : E, 'E3' : E, 
        'NU12' : NU, 'NU13' : NU, 'NU23' : NU,
        'NU21' : NU, 'NU31' : NU, 'NU32' : NU,
        'G12' : G, 'G13' : G, 'G23' : G}
    return params


def get_triso_params(*args, **kwargs) -> dict:
    """
    Returns all 12 orthotropic engineering constants for a 
    transversely isotropic material. 
    From the total of 12 constants, exactly 5 must to be provided, 
    from which 1 must be the out-of-plane shear moduli. 
    The remaining 4 necessary constants can be provided in any combination.
    """
    KWARGS = {key.upper() : value for key, value in kwargs.items()}
    G12 = getasany(['G12', 'G21'], None, **KWARGS)
    G23 = getasany(['G23', 'G32'], None, **KWARGS)
    G13 = getasany(['G13', 'G23'], None, **KWARGS)
    E1 = KWARGS.get('E1', None)
    E2 = KWARGS.get('E2', None)
    E3 = KWARGS.get('E3', None)
    NU12 = KWARGS.get('NU12', None)
    NU21 = KWARGS.get('NU21', None)
    NU23 = KWARGS.get('NU23', None)
    NU32 = KWARGS.get('NU32', None)
    NU13 = KWARGS.get('NU13', None)
    NU31 = KWARGS.get('NU31', None)
    params = \
        {'E1' : E1, 'E2' : E2, 'E3' : E3, 
        'NU12' : NU12, 'NU13' : NU13, 'NU23' : NU23,
        'NU21' : NU21, 'NU31' : NU31, 'NU32' : NU32,
        'G12' : G12, 'G13' : G13, 'G23' : G23}
    return finalize_params_triso(**params)


def get_ortho_params(*args, **kwargs) -> dict:
    """
    Returns all 12 orthotropic engineering constants for an 
    orthotropic material. 
    From the total of 12 constants, 9 must to be provided, from which
    3 must be shear moduli. The remaining 6 necessary constants 
    can be provided in any combination.
    """
    KWARGS = {key.upper() : value for key, value in kwargs.items()}
    G12 = getasany(['G12', 'G21'], None, **KWARGS)
    assert G12 is not None, "Shear modulus G12 \
        must be provided as either G12 or G21!"
    G23 = getasany(['G23', 'G32'], None, **KWARGS)
    assert G23 is not None, "Shear modulus G23 \
        must be provided as either G23 or G32!"
    G13 = getasany(['G13', 'G23'], None, **KWARGS)
    assert G13 is not None, "Shear modulus G13 \
        must be provided as either G13 or G31!"
    E1 = KWARGS.get('E1', None)
    E2 = KWARGS.get('E2', None)
    E3 = KWARGS.get('E3', None)
    NU12 = KWARGS.get('NU12', None)
    NU21 = KWARGS.get('NU21', None)
    NU23 = KWARGS.get('NU23', None)
    NU32 = KWARGS.get('NU32', None)
    NU13 = KWARGS.get('NU13', None)
    NU31 = KWARGS.get('NU31', None)
    params = \
        {'E1' : E1, 'E2' : E2, 'E3' : E3, 
        'NU12' : NU12, 'NU13' : NU13, 'NU23' : NU23,
        'NU21' : NU21, 'NU31' : NU31, 'NU32' : NU32,
        'G12' : G12, 'G13' : G13, 'G23' : G23}
    return finalize_params_ortho(**params)


if __name__ == '__main__':
    get_iso_params(E=1, NU=0.2)
