# -*- coding: utf-8 -*-
import numpy as np

from ..mesh import FemMesh


class LineMesh(FemMesh):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def plot(self, *args, as_tubes=True, radius=0.1, **kwargs):
        if not as_tubes:
            return super().plot(*args, **kwargs)
        else:
            self.to_pv(as_tubes=True, radius=radius).plot(smooth_shading=True)
            
    def to_pv(self, *args, as_tubes=True, radius=0.1, **kwargs):
        import pyvista as pv
        if not as_tubes:
            return super().to_pv(*args, **kwargs)
        else:
            poly = pv.PolyData()
            poly.points = self.coords()
            topo = self.topology()
            lines = np.full((len(topo), 3), 2, dtype=int)
            lines[:, 1:] = topo
            poly.lines = lines
            return poly.tube(radius=radius)
            
              
if __name__ == '__main__':
    from dewloosh.solid.fem.structure.structure import Structure
    from dewloosh.math.linalg import linspace, Vector
    from dewloosh.solid.fem.cells import B2
    from dewloosh.geom.space import StandardFrame, \
        PointCloud, frames_of_lines
    from dewloosh.math.array import repeat
    import numpy as np
    from numpy import pi as PI
    
    # geometry
    L = 3000.  #length
    R = 25.0  # radius
    
    # material
    E = 210000.0
    nu = 0.3 
    
    # mesh
    nElem = 30  # number of finite elements
    
    # load
    F = -1000.  # vertical load at the end
    
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
    angles = [0, 0, 2 * np.random.rand() * PI]
    #angles = [0., PI/2, 0.]
    #angles = [0., 0., PI]
    #angles = [0., 0., 0.]
    angles = np.random.rand(3) * PI * 2
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
                    celltype=B2, frame=GlobalFrame, model=Hooke)
    frames = repeat(local_frame.dcm(), nElem)
    frames = frames_of_lines(coords, topo)
    mesh.frames = frames
    structure = Structure(mesh=mesh)
    
    # intermediate plots
    #mesh.plot()
    #structure.plot(radius=R)
    
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
    print("Analytic Bernoulli Solution : {}".format(sol_exact))
    print("FEM Solution : {}".format(sol_fem))
    diff = 100 * (sol_fem - sol_exact) / sol_exact
    print("Difference : {} %".format(diff))
    
    # final plot
    import pyvista as pv
    p = pv.Plotter(notebook=False)
    load_size = 50
    pvobj = mesh.to_pv(as_tubes=False)
    forces = loads[:, :3]
    forces /= np.abs(forces).max()
    forces *= load_size
    pvobj["loads"] = forces
    pvobj.set_active_vectors("loads")
    p.add_mesh(pvobj, show_edges=True, color='black')
    p.add_mesh(pvobj.arrows, color='red')
    p.add_mesh(mesh.to_pv(as_tubes=True, radius=R), opacity=0.5)
    p.add_points(coords_new, render_points_as_spheres=True,
                 point_size=10.0, color='blue')
    actor = p.show_bounds(grid='front', location='outer',
                          all_edges=True)
    p.show_axes()
    p.show()