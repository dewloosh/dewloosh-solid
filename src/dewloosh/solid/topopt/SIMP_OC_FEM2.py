# -*- coding: utf-8 -*-
from pyoneer.mech.topopt.utils import compliances_bulk, cells_around, \
    get_filter_factors, get_filter_factors_csr, weighted_stiffness_bulk
from pyoneer.mech.fem.structure import Structure
from pyoneer.mech.topopt.filter import sensitivity_filter, \
    sensitivity_filter_csr
from pyoneer.math.linalg.sparse.csr import csr_matrix as csr
import numpy as np
from collections import namedtuple


OptRes = namedtuple('OptimizationResult', 'x obj vol')


def OC_SIMP_COMP(structure: Structure, *args,
                 miniter=50, maxiter=100, p_start=1.0, p_stop=4.0,
                 p_inc=0.2, p_step=5, q=0.5, vfrac=0.6, dtol=0.1,
                 r_min=None, summary=True, **kwargs):

    do_filter = r_min is not None

    structure.preprocess()
    gnum = structure._gnum
    K_bulk_0 = np.copy(structure._K_bulk)
    vols = structure.volumes()
    centers = structure.centers()

    # initial solution to set up parameters
    dens = np.zeros_like(vols)
    dens_tmp = np.zeros_like(dens)
    dens_tmp_ = np.zeros_like(dens)
    dCdx = np.zeros_like(dens)
    comps = np.zeros_like(dens)

    def compliance():
        structure._K_bulk[:, :, :] = weighted_stiffness_bulk(K_bulk_0, dens)
        structure.process(summary=True)
        U = structure._du
        comps[:] = compliances_bulk(K_bulk_0, U, gnum)
        np.clip(comps, 1e-7, None, out=comps)
        return np.sum(comps)

    # ------------------ INITIAL SOLUTION ---------------
    comp = compliance()
    vol = np.sum(vols)
    vol_start = vol
    vol_min = vfrac * vol_start

    # initialite filter
    if do_filter:
        neighbours = cells_around(centers, r_min, as_csr=False)
        if isinstance(neighbours, csr):
            factors = get_filter_factors_csr(centers, neighbours, r_min)
            fltr = sensitivity_filter_csr
        else:
            factors = get_filter_factors(centers, neighbours, r_min)
            fltr = sensitivity_filter

    # ------------- INITIAL FEASIBLE SOLUTION ------------
    dens[:] = vfrac
    vol = np.sum(dens * vols)
    comp = compliance()
    yield OptRes(dens, comp, vol)

    # ------------------- ITERATION -------------------

    p = p_start
    cIter = -1
    dt = 0
    terminate = False
    while not terminate:
        if (p < p_stop) and (np.mod(cIter, p_step) == 0):
            p += p_inc
        cIter += 1

        # estimate lagrangian
        lagr = p * comp / vol

        # set up boundaries of change
        _dens = dens * (1 - dtol)
        np.clip(_dens, 1e-5, 1.0, out=_dens)
        dens_ = dens * (1 + dtol)
        np.clip(dens_, 1e-5, 1.0, out=dens_)

        # sensitivity [*(-1)]
        dCdx[:] = p * comps * dens ** (p-1)

        # sensitivity filter
        if do_filter:
            dCdx[:] = fltr(dCdx, dens, neighbours, factors)

        # calculate new densities and lagrangian
        dens_tmp_[:] = dens * (dCdx / vols) ** q
        dens_tmp[:] = dens_tmp_
        _lagr = 0
        lagr_ = 2 * lagr
        while (lagr_ - _lagr) > 1e-3:
            _lagr_ = (_lagr + lagr_) / 2
            dens_tmp[:] = dens_tmp_ / (_lagr_ ** q)
            np.clip(dens_tmp, _dens, dens_, out=dens_tmp)
            vol_tmp = np.sum(dens_tmp * vols)
            if vol_tmp < vol_min:
                lagr_ = _lagr_
            else:
                _lagr = _lagr_
        lagr = lagr_
        dens[:] = dens_tmp

        # resolve equilibrium equations and calculate compliance
        comp = compliance()
        dt += structure.summary['proc']['time [ms]']
        vol = np.sum(dens * vols)
        yield OptRes(dens, comp, vol)

        if cIter < miniter:
            terminate = False
        elif cIter >= maxiter:
            terminate = True
        else:
            terminate = (p >= p_stop)

    if summary:
        structure.summary['opt'] = {
            'avg. time [ms]': dt / cIter,
            'niter': cIter
        }
    structure._K_bulk[:, :, :] = K_bulk_0
    # structure.postprocess()
    return OptRes(dens, comp, vol)


if __name__ == '__main__':
    pass
