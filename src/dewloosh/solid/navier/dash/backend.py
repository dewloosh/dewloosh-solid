# -*- coding: utf-8 -*-
from dewloosh.solid.navier import NavierProblem
from dewloosh.geom import grid, PolyData
from dewloosh.geom.topo.tr import Q4_to_T3
import numpy as np
from copy import deepcopy


def calc3d(*args, E, nu, t, Lx, Ly, x0, y0, w, h, q, 
           nx=50, ny=50, rx=30, ry=40, **kwargs):
    global coords, triangles, Mindlin, Kirchhoff
    # load
    loads = {
        'LG1' : {
            'LC1' : {
                'type' : 'rect',
                'points' : [[0, 0], [Lx, Ly]],
                'value' : [0, 0, -0.01],
                    },
            'LC2' : {
                'type' : 'rect',
                'region' : [x0, y0, w, h],
                'value' : [0, 0, q],
                    }
                },
        'LG2' : {
            'LC3' : {
                'type' : 'point',
                'point' : [Lx/3, Ly/2],
                'value' : [0, 0, -10],
                    },
            'LC4' : {
                'type' : 'point',
                'point' : [2*Lx/3, Ly/2],
                'value' : [0, 0, 10],
                    }
                },
        'dummy1' : 10
        }
    # grid
    gridparams = {
        'size' : (Lx, Ly),
        'shape' : (rx, ry),
        'origo' : (0, 0),
        'start' : 0,
        'eshape' : 'Q4'
        }
    coords_, topo = grid(**gridparams)
    coords = np.zeros((coords_.shape[0], 3))
    coords[:, :2] = coords_[:, :]
    del coords_
    coords, triangles = Q4_to_T3(coords, topo)
    Mesh = PolyData(coords=coords, topo=triangles)
    centers = Mesh.centers()
    #
    G = E/2/(1+nu)
    D = np.array([[1, nu, 0], [nu, 1, 0], 
                  [0., 0, (1-nu)/2]]) * t**3 * (E / (1-nu**2)) / 12
    S = np.array([[G, 0], [0, G]]) * t * 5 / 6
    #
    Mindlin = NavierProblem((Lx, Ly), (nx, ny), D=D, S=S, model='mindlin')
    Mindlin.add_loads_from_dict(deepcopy(loads))
    Mindlin.solve()
    Mindlin.postproc(centers[:, :2], cleanup=False)
    #
    Kirchhoff = NavierProblem((Lx, Ly), (nx, ny), D=D, S=S, model='kirchhoff')
    Kirchhoff.add_loads_from_dict(deepcopy(loads))
    Kirchhoff.solve()
    Kirchhoff.postproc(centers[:, :2], cleanup=False)
    
    res2d = np.stack([Mindlin.loads['LG1', 'LC2'].res2d, 
                      Kirchhoff.loads['LG1', 'LC2'].res2d], axis=0)
    return coords, triangles, res2d 