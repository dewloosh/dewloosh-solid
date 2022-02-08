# -*- coding: utf-8 -*-
from dewloosh.core import squeeze
from dewloosh.solid.fem.preproc import fem_coeff_matrix_coo
from dewloosh.solid.fem.elements.utils import stiffness_matrix_bulk2
from dewloosh.solid.fem.utils import topo_to_gnum, \
    approximation_matrix, nodal_approximation_matrix, \
    nodal_compatibility_factors, compatibility_factors_to_coo, \
    compatibility_factors,  penalty_factor_matrix, \
    transform_element_stiffness_matrices as transform_stiffness
from dewloosh.geom.utils import distribute_nodal_data, \
    collect_nodal_data, points_of_cells
from scipy.sparse import coo_matrix
import numpy as np
from collections import namedtuple


Quadrature = namedtuple('QuadratureRule', ['inds', 'pos', 'weight'])


class FiniteElement:
    # must be reimplemented
    NNODE: int = None  # number of nodes, normally inherited
    # from the mesh object
    NDOFN: int = None  # numper of dofs per node, normally
    # inherited form the model object
    NDIM: int = None  # number of geometrical dimensions,
    # normally inherited from the mesh object

    # optional
    qrule: str = None
    compatible = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # !TODO : this should be implemented at geometry
    @classmethod
    def lcoords(cls, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    @classmethod
    def lcenter(cls, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    # !TODO : can be reimplemented if the element is not IsoP
    def jacobian_matrix(self, *args, **kwargs):
        raise NotImplementedError
    
    # !TODO : this should be implemented at geometry
    # !TODO : can be reimplemented if the element is not IsoP
    def jacobian(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at model
    @classmethod
    def strain_displacement_matrix(cls, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    def shape_function_values(self, *args, **kwargs):
        raise NotImplementedError
    
    # !TODO : this should be implemented at geometry
    # !TODO : can be reimplemented if the element is not IsoP
    def shape_function_matrix(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    def shape_function_derivatives(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at model
    def model_stiffness_matrix(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at model
    def stresses_at_centers(self, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at model
    def stresses_at_cells_nodes(self, *args, **kwargs):
        raise NotImplementedError
    
    # !TODO : this should be implemented at model
    def element_dcm_matrices(self, *args, frames=None, **kwargs):
        return None

    def local_coordinates(self, *args, frames=None, _topo=None, **kwargs):
        frames = self.frames.to_numpy() if frames is None else frames
        _topo = self.nodes.to_numpy() if _topo is None else _topo
        coords = self.pointdata.x.to_numpy()
        return points_of_cells(coords, _topo, local_axes=frames)
        
    def global_dof_numbering(self, *args, topo=None, **kwargs):
        topo = self.nodes.to_numpy() if topo is None else topo
        return topo_to_gnum(topo, self.NDOFN)

    def approximation_matrix(self, *args, **kwargs):
        return approximation_matrix(self.ndf.to_numpy(), self.NDOFN)
    
    def approximation_matrix_coo(self, *args, **kwargs):
        N = len(self.pointdata) * self.NDOFN
        topo = self.nodes.to_numpy()
        appr = self.approximation_matrix()
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(appr, *args, inds=gnum, N=N, **kwargs)
    
    def nodal_approximation_matrix(self, *args, **kwargs):
        return nodal_approximation_matrix(self.ndf.to_numpy())
    
    def nodal_approximation_matrix_coo(self, *args, **kwargs):
        N = len(self.pointdata)
        topo = self.nodes.to_numpy()
        appr = self.nodal_approximation_matrix()
        return fem_coeff_matrix_coo(appr, *args, inds=topo, N=N, **kwargs)
    
    def distribute_nodal_data(self, data, key):
        topo = self.nodes.to_numpy()
        ndf = self.ndf.to_numpy()
        self._wrapped[key] = distribute_nodal_data(data, topo, ndf)

    def collect_nodal_data(self, key, *args, N=None, **kwargs):
        topo = self.nodes.to_numpy()
        N = len(self.pointdata) if N is None else N
        cellloads = self._wrapped[key].to_numpy()
        return collect_nodal_data(cellloads, topo, N)
    
    def compatibility_penalty_matrix_coo(self, *args, nam_csr_tot, p=1e12, **kwargs):
        """
        Parameters
        ----------
        nam_csr_tot : nodal_approximation matrix for the whole structure
                      in csr format.  
        """
        topo = self.nodes.to_numpy()
        nam = self.nodal_approximation_matrix()
        factors, nreg = nodal_compatibility_factors(nam_csr_tot, nam, topo)
        factors, nreg = compatibility_factors(factors, nreg, self.NDOFN)
        data, rows, cols = compatibility_factors_to_coo(factors, nreg)
        return coo_matrix((data*p, (rows, cols)))
        
    def penalty_factor_matrix(self, *args, **kwargs):
        cellfixity = self.fixity.to_numpy()
        shp = self.shape_function_values(self.lcoords())
        return penalty_factor_matrix(cellfixity, shp)

    def penalty_matrix(self, *args, p=1e12, **kwargs):
        return self.penalty_factor_matrix() * p

    def penalty_matrix_coo(self, *args, **kwargs):
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.nodes.to_numpy()
        K_bulk = self.penalty_matrix(*args, topo=topo, **kwargs)
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(K_bulk, *args, inds=gnum, N=N, **kwargs)

    @squeeze(True)
    def stiffness_matrix(self, *args, _topo=None, _ecoords=None, **kwargs):
        _topo = self.nodes.to_numpy()
        if 'frames' not in kwargs:
            if 'frames' in self._wrapped.fields: 
                frames = self._wrapped.frames.to_numpy()
            else:
                frames=None
        
        if _ecoords is None:
            if frames is not None:        
                _ecoords = self.local_coordinates(_topo=_topo, frames=frames)
            else:
                _ecoords = points_of_cells(self.root().coords(), _topo)
        
        q = kwargs.get('_q', None)
        if q is None:
            q = self.quadrature[self.qrule]
            if isinstance(q, dict):
                N = self.NNODE * self.NDOFN
                res = np.zeros((len(self), N, N))
                for qinds, qvalue in q.items():
                    if isinstance(qvalue, str):
                        qpos, qweight = self.quadrature[qvalue]
                    else:
                        qpos, qweight = qvalue
                    q = Quadrature(qinds, qpos, qweight)
                    res[:, :, :] += self.stiffness_matrix(*args, _topo=_topo, 
                                                          frames=frames, _q=q, 
                                                          _ecoords=_ecoords,
                                                          squeeze=False, **kwargs)
            else:
                qpos, qweight = self.quadrature[self.qrule]
                q = Quadrature(None, qpos, qweight)
                res = self.stiffness_matrix(*args, _topo=_topo, 
                                            frames=frames, _q=q,
                                            _ecoords=_ecoords, 
                                            squeeze=False, **kwargs)       
            if frames is not None:
                dcm = self.element_dcm_matrices(frames=frames)
                if dcm is not None:
                    return transform_stiffness(res, dcm)
                else:
                    return res
            else:
                return res
        
        dshp = self.shape_function_derivatives(q.pos)
        jac = self.jacobian_matrix(dshp=dshp, ecoords=_ecoords)
        B = self.strain_displacement_matrix(dshp=dshp, jac=jac)
        if q.inds is not None:
            # zero out unused indices, only for selective integration
            inds = np.where(~np.in1d(np.arange(self.NSTRE), q.inds))[0]
            B[:, :, inds, :] = 0.
        djac = self.jacobian(jac=jac)
        C = self.model_stiffness_matrix()
        return stiffness_matrix_bulk2(C, B, djac, q.weight)

    def stiffness_matrix_coo(self, *args, **kwargs):
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.nodes.to_numpy()
        K_bulk = self.stiffness_matrix(*args, _topo=topo, **kwargs)
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(K_bulk, *args, inds=gnum, N=N, **kwargs)
