# -*- coding: utf-8 -*-
from dewloosh.solid.navier import NavierProblem
from dewloosh.geom import grid, PolyData
from dewloosh.geom.topo.tr import Q4_to_T3
import numpy as np
from copy import deepcopy


def gen_grid(*args, Lx, Ly, rx=30, ry=40, proj='3d', **kwargs):
    # generate grid if necessary and prepare coordinates
    gridparams = {
        'size': (Lx, Ly),
        'shape': (rx, ry),
        'origo': (0, 0),
        'start': 0,
        'eshape': 'Q4'
    }
    coords_, topo = grid(**gridparams)
    coords = np.zeros((coords_.shape[0], 3))
    coords[:, :2] = coords_[:, :]
    del coords_
    if proj == '3d':
        coords, triangles = Q4_to_T3(coords, topo)
        return PolyData(coords=coords, topo=triangles)
    else:
        return coords


def gen_problem(*args, E, nu, t, Lx, Ly, x0, y0, w, h, q,
                nx=50, ny=50, **kwargs):
    # load
    loads = {
        'type': 'rect',
        'region': [x0, y0, w, h],
        'value': [0, 0, q],
    }
    # material
    G = E/2/(1+nu)
    D = np.array([[1, nu, 0], [nu, 1, 0],
                  [0., 0, (1-nu)/2]]) * t**3 * (E / (1-nu**2)) / 12
    S = np.array([[G, 0], [0, G]]) * t * 5 / 6
    # problem
    P = NavierProblem((Lx, Ly), (nx, ny), D=D, S=S)
    P.add_loads_from_dict(deepcopy(loads))
    return P


def solve(P: NavierProblem, coords: np.ndarray):
    res2d = []
    P.model = 'm'
    P.solve()
    P.postproc(coords[:, :2], cleanup=False)
    res2d.append(P.loads.res2d)
    P.model = 'k'
    P.solve()
    P.postproc(coords[:, :2], cleanup=False)
    res2d.append(P.loads.res2d)
    return np.stack(res2d, axis=0)
