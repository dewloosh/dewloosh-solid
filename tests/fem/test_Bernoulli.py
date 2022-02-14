# -*- coding: utf-8 -*-
from dewloosh.math.linalg.vector import Vector
from dewloosh.math.linalg import linspace, Vector
from dewloosh.solid.fem.lib import Bernoulli2 as Bernoulli
from dewloosh.geom.space import StandardFrame, \
    PointCloud, frames_of_lines
from dewloosh.solid.fem import LineMesh
from dewloosh.optimus.fem import Structure
from dewloosh.math.array import repeat
from hypothesis import given, settings, HealthCheck, \
    strategies as st
import unittest
import numpy as np
from numpy import pi as PI


settings.register_profile(
    "fem_test",
    max_examples=200,
    deadline=20 * 1000,  # Allow 20s per example (deadline is specified in milliseconds)
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
)


def console_tip_displacement(*args, 
                             L=3000,  # length of the console 
                             R=25.0, # radius of the tube
                             nElem=10,  # number of finite elements to use 
                             F=-1000,  # value of the vertical load at the free end
                             E=210000.0,  # Young's modulus
                             nu=0.3,  # Poisson's ratio
                             angles=None
                             ):    
    # cross section
    A = PI * R**2
    Ix = PI * R**4 / 2
    Iy = PI* R**4 / 4
    Iz = PI * R**4 / 4

    # model stiffness matrix
    G = E / (2 * (1 + nu))
    Hooke = np.array([
        [E*A, 0, 0, 0],
        [0, G*Ix, 0, 0],
        [0, 0, E*Iy, 0],
        [0, 0, 0, E*Iz]
        ])
    
    # space
    GlobalFrame = StandardFrame(dim=3)
    if angles is None:
        angles = np.random.rand(3) * PI
    local_frame = GlobalFrame.fork('Body', angles, 'XYZ')
    
    # mesh
    p0 = np.array([0., 0., 0.])
    p1 = np.array([L, 0., 0.])
    coords = linspace(p0, p1, nElem+1)
    coords = PointCloud(coords, frame=local_frame).show()
    topo = np.zeros((nElem, 2), dtype=int)
    topo[:, 0] = np.arange(nElem)
    topo[:, 1] = np.arange(nElem) + 1
    
    # support at the leftmost, load at the rightmost node
    loads = np.zeros((coords.shape[0], 6))
    fixity = np.zeros((coords.shape[0], 6)).astype(bool)
    global_load_vector = \
        Vector([0., 0., F], frame=local_frame).show()
    loads[-1, :3] = global_load_vector
    fixity[0, :] = True
            
    # set up objects
    mesh = LineMesh(coords=coords, topo=topo, loads=loads, fixity=fixity, 
                    celltype=Bernoulli, frame=GlobalFrame, model=Hooke)
    frames = repeat(local_frame.dcm(), nElem)
    frames = frames_of_lines(coords, topo)
    mesh.frames = frames
    structure = Structure(mesh=mesh)
        
    # solution
    structure.linsolve()
    
    # postproc
    # 1) displace the mesh
    dofsol = structure.nodal_dof_solution()[:, :3]
    coords_new = coords + dofsol
    structure.mesh.pointdata['x'] = coords_new
    local_dof_solution = \
        Vector(dofsol[-1, :3], frame=GlobalFrame).show(local_frame)
    sol_fem = local_dof_solution[2]
    
    # Bernoulli solution
    EI = E * Iy
    sol_exact = F * L**3 / (3 * EI)
    diff = 100 * (sol_fem - sol_exact) / sol_exact
    return np.abs(diff)
    


class TestEncoding(unittest.TestCase):
    
    def test_console_Bernoulli_1(self):
        assert console_tip_displacement() < 1e-3
                
    @given(st.integers(min_value=5, max_value=20), 
           st.floats(min_value=3000., max_value=3500.),
           st.floats(min_value=0., max_value=2*PI),
           st.floats(min_value=0., max_value=2*PI),
           st.floats(min_value=0., max_value=2*PI))
    @settings(settings.load_profile("fem_test"))
    def test_console_Bernoulli_2(self, N, L, a1, a2, a3):
        angles = np.array([a1, a2, a3])
        assert console_tip_displacement(L=L, nElem=N, angles=angles) < 1e-3
        
    
if __name__ == "__main__":
    
    print(console_tip_displacement())
    #unittest.main()