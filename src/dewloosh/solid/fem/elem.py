# -*- coding: utf-8 -*-
from scipy.sparse import coo_matrix
import numpy as np
from collections import namedtuple, Iterable

from dewloosh.core import squeeze

from dewloosh.math.linalg import ReferenceFrame, Vector
from dewloosh.math.array import atleast1d, atleastnd, ascont
from dewloosh.math.utils import to_range

from dewloosh.geom.utils import distribute_nodal_data, \
    collect_nodal_data

from .preproc import fem_coeff_matrix_coo
from .postproc import approx_element_solution_bulk, calculate_internal_forces_bulk
from .cells.utils import stiffness_matrix_bulk2
from .utils import topo_to_gnum, approximation_matrix, nodal_approximation_matrix, \
    nodal_compatibility_factors, compatibility_factors_to_coo, \
    compatibility_factors, penalty_factor_matrix, assemble_load_vector
from .utils import tr_cells_1d_in_multi, tr_cells_1d_out_multi, element_dof_solution_bulk, \
    transform_stiffness


Quadrature = namedtuple('QuadratureRule', ['inds', 'pos', 'weight'])


UX, UY, UZ, ROTX, ROTY, ROTZ = FX, FY, FZ, MX, MY, MZ = list(range(6))
ulabels = {UX: 'UX', UY: 'UY', UZ: 'UZ',
           ROTX: 'ROTX', ROTY: 'ROTY', ROTZ: 'ROTZ'}
flabels = {FX: 'FX', FY: 'FY', FZ: 'FZ', MX: 'MX', MY: 'MY', MZ: 'MZ'}
ulabel_to_int = {v: k for k, v in ulabels.items()}
flabel_to_int = {v: k for k, v in flabels.items()}


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
    
    def direction_cosine_matrix(self):
        return None
       
    def transform_stiffness_matrix(self, K, *args, **kwargs):
        dcm = self.direction_cosine_matrix()
        return K if dcm is None else transform_stiffness(K, dcm)
    
    @classmethod
    def integrate_body_loads(cls, *args, **kwargs):
        raise NotImplementedError
    
    def local_coordinates(self, *args, frames=None, _topo=None, **kwargs):
        # implemented at PolyCell
        raise NotImplementedError
    
    def points_of_cells(self):
        # implemented at PolyCell
        raise NotImplementedError
    
    @squeeze(True)
    def dof_solution(self, *args, target='local', cells=None, points=None, 
                     rng=None, flatten=True, **kwargs):
        """
        Returns nodal displacements for the cells, wrt. their local frames.
        
        Parameters
        ----------
        points : scalar or array, Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng' parameter. 
                If not provided, results are returned for the nodes of the selected elements.
                Default is None.
                
        rng : array, Optional
            Range where the points of evauation are understood. Default is [0, 1].
            
        cells : integer or array, Optional.
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.
        
        target : frame, Optional.
            Reference frame for the output. A value of None or 'local' refers to the local system
            of the cells. Default is 'local'.
        
        ---
        Returns
        -------
        numpy array or dictionary
        (nE, nEVAB, nRHS) if flatten else (nE, nNE, nDOF, nRHS) 
    
        """
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id.to_numpy())
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
        else:
            cells = np.s_[:]
                                    
        dofsol = self.pointdata.dofsol.to_numpy()
        dofsol = atleastnd(dofsol, 3, back=True)
        nP, nDOF, nRHS = dofsol. shape
        dofsol = dofsol.reshape(nP* nDOF, nRHS)
        gnum = self.global_dof_numbering()[cells]
        dcm = self.direction_cosine_matrix()[cells]
        
        # transform values to cell-local frames
        values = element_dof_solution_bulk(dofsol, gnum)  # (nE, nEVAB, nRHS)
        values = ascont(np.swapaxes(values, 1, 2))
        values = tr_cells_1d_in_multi(values, dcm)
        values = ascont(np.swapaxes(values, 1, 2))
        
        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)
            
        # approximate at points
        # values : (nE, nEVAB, nRHS)
        points, rng = to_range(points, source=rng, target=[-1, 1]).flatten(), [-1, 1]
        N = self.shape_function_matrix(points, rng=rng)[cells]  # (nE, nP, nDOF, nDOF * nNODE)
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = approx_element_solution_bulk(values, N)  # (nE, nRHS, nP, nDOF)
        
        if target is not None:
            # transform values to a destination frame, otherwise return
            # the results in the local frames of the cells
            if isinstance(target, str):
                if target == 'local':
                    pass
                elif target == 'global':
                    nDOF = values.shape[2]
                    target = ReferenceFrame(dim=nDOF)
            if isinstance(target, ReferenceFrame):    
                nE, nRHS, nP, nDOF = values.shape
                values = values.reshape(nE, nRHS, nP * nDOF)
                dcm = self.direction_cosine_matrix(N=nP)[cells]
                values = tr_cells_1d_out_multi(values, dcm)
                values = values.reshape(nE, nRHS, nP, nDOF)
                            
        values = np.moveaxis(values, 1, -1)   # (nE, nP, nDOF, nRHS)
            
        if flatten:
            nE, nP, nDOF, nRHS = values.shape
            values = values.reshape(nE, nP * nDOF, nRHS)  
        
        # values : (nE, nP, nDOF, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements 
            data = values
        elif isinstance(cells, Iterable):
            data = {c : values[i] for i, c in enumerate(cells)}                    
        else:
            raise TypeError("Invalid data type <> for cells.".format(type(cells)))    
        
        return data
    
    @squeeze(True)
    def strains(self, *args, cells=None, points=None, rng=None, **kwargs):
        """
        Returns strains for the cells.
        
        Parameters
        ----------
        points : scalar or array, Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng' parameter. 
                If not provided, results are returned for the nodes of the selected elements.
                Default is None.
                
        rng : array, Optional
            Range where the points of evauation are understood. Default is [0, 1].
            
        cells : integer or array, Optional.
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.
                
        ---
        Returns
        -------
        numpy array or dictionary
        (nE, nP * nSTRE, nRHS) if flatten else (nE, nP, nSTRE, nRHS) 
    
        """
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id.to_numpy())
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
        else:
            cells = np.s_[:]
                                    
        dofsol = self.pointdata.dofsol.to_numpy()
        dofsol = atleastnd(dofsol, 3, back=True)
        nP, nDOF, nRHS = dofsol. shape
        dofsol = dofsol.reshape(nP* nDOF, nRHS)
        gnum = self.global_dof_numbering()[cells]
        dcm = self.direction_cosine_matrix()[cells]
        ecoords = self.local_coordinates()[cells]
        
        # transform values to cell-local frames
        values = element_dof_solution_bulk(dofsol, gnum)  # (nE, nEVAB, nRHS)
        values = ascont(np.swapaxes(values, 1, 2))
        values = tr_cells_1d_in_multi(values, dcm)
        values = ascont(np.swapaxes(values, 1, 2))
        
        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)
            
        # approximate at points
        # values : (nE, nEVAB, nRHS)
        points, rng = to_range(points, source=rng, target=[-1, 1]).flatten(), [-1, 1]
                
        dshp = self.shape_function_derivatives(points, rng=rng)[cells] 
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)  # (nE, nP, 1, 1)
        B = self.strain_displacement_matrix(dshp=dshp, jac=jac)  # (nE, nP, 4, nNODE * 6)
        
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = approx_element_solution_bulk(values, B)  # (nE, nRHS, nP, 4)
        values = ascont(np.moveaxis(values, 1, -1))   # (nE, nP, nDOF, nRHS)
            
        # values : (nE, nP, 4, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements 
            data = values
        elif isinstance(cells, Iterable):
            data = {c : values[i] for i, c in enumerate(cells)}                    
        else:
            raise TypeError("Invalid data type <> for cells.".format(type(cells)))    
        
        return data
    
    @squeeze(True)
    def internal_forces(self, *args, cells=None, points=None, rng=None, 
                        flatten=True, target='local', **kwargs):
        """
        Returns strains for the cells.
        
        Parameters
        ----------
        points : scalar or array, Optional
                Points of evaluation. If provided, it is assumed that the given values
                are wrt. the range [0, 1], unless specified otherwise with the 'rng' parameter. 
                If not provided, results are returned for the nodes of the selected elements.
                Default is None.
                
        rng : array, Optional
            Range where the points of evauation are understood. Default is [0, 1].
            
        cells : integer or array, Optional.
            Indices of cells. If not provided, results are returned for all cells.
            Default is None.
                
        ---
        Returns
        -------
        numpy array or dictionary
        (nE, nP * nSTRE, nRHS) if flatten else (nE, nP, nSTRE, nRHS) 
    
        """
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id.to_numpy())
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
        else:
            cells = np.s_[:]
                                    
        dofsol = self.pointdata.dofsol.to_numpy()
        dofsol = atleastnd(dofsol, 3, back=True)
        nP, nDOF, nRHS = dofsol. shape
        dofsol = dofsol.reshape(nP* nDOF, nRHS)
        gnum = self.global_dof_numbering()[cells]
        dcm = self.direction_cosine_matrix()[cells]
        ecoords = self.local_coordinates()[cells]
        
        # transform dofsol to cell-local frames
        values = element_dof_solution_bulk(dofsol, gnum)  # (nE, nEVAB, nRHS)
        values = ascont(np.swapaxes(values, 1, 2))
        values = tr_cells_1d_in_multi(values, dcm)
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nEVAB, nRHS)
        
        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)
            
        # approximate at points
        # values : (nE, nEVAB, nRHS)
        points, rng = to_range(points, source=rng, target=[-1, 1]).flatten(), [-1, 1]
                
        dshp = self.shape_function_derivatives(points, rng=rng)[cells] 
        jac = self.jacobian_matrix(dshp=dshp, ecoords=ecoords)  # (nE, nP, 1, 1)
        B = self.strain_displacement_matrix(dshp=dshp, jac=jac)  # (nE, nP, 4, nNODE * 6)
        
        values = ascont(np.swapaxes(values, 1, 2))  # (nE, nRHS, nEVAB)
        values = approx_element_solution_bulk(values, B)  # (nE, nRHS, nP, 4)        
        D = self.model_stiffness_matrix()[cells]
        values = calculate_internal_forces_bulk(values, D) # (nE, nRHS, nP, 4) 
        values = ascont(np.moveaxis(values, 1, -1))   # (nE, nP, 4, nRHS)
        
        if target is not None:
            # transform values to a destination frame, otherwise return
            # the results in the local frames of the cells
            assert target == 'local'
                   
        if flatten:
            nE, nP, nSTRE, nRHS = values.shape
            values = values.reshape(nE, nP * nSTRE, nRHS)  
                                 
        # values : (nE, nP, 4, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements 
            data = values
        elif isinstance(cells, Iterable):
            data = {c : values[i] for i, c in enumerate(cells)}                    
        else:
            raise TypeError("Invalid data type <> for cells.".format(type(cells)))    
        
        return data
                    
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

    def penalty_matrix_coo(self, *args, p=1e12, **kwargs):
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.nodes.to_numpy()
        K_bulk = self.penalty_matrix(*args, topo=topo, p=p, **kwargs)
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
                _ecoords = self.points_of_cells(topo=_topo)
        
        q = kwargs.get('_q', None)
        if q is None:
            # main loop
            q = self.quadrature[self.qrule]
            if isinstance(q, dict):
                # many side loops
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
                # one side loop
                qpos, qweight = self.quadrature[self.qrule]
                q = Quadrature(None, qpos, qweight)
                res = self.stiffness_matrix(*args, _topo=_topo, 
                                            frames=frames, _q=q,
                                            _ecoords=_ecoords, 
                                            squeeze=False, **kwargs)
            # end of main cycle
            self._wrapped['K'] = res
            return self.transform_stiffness_matrix(res)
        
        # in side loop
        dshp = self.shape_function_derivatives(q.pos)
        jac = self.jacobian_matrix(dshp=dshp, ecoords=_ecoords)
        B = self.strain_displacement_matrix(dshp=dshp, jac=jac)
        if q.inds is not None:
            # zero out unused indices, only for selective integration
            inds = np.where(~np.in1d(np.arange(self.NSTRE), q.inds))[0]
            B[:, :, inds, :] = 0.
        djac = self.jacobian(jac=jac)
        D = self.model_stiffness_matrix()
        return stiffness_matrix_bulk2(D, B, djac, q.weight)

    def stiffness_matrix_coo(self, *args, **kwargs):
        nP = len(self.pointdata)
        N = nP * self.NDOFN
        topo = self.nodes.to_numpy()
        K_bulk = self.stiffness_matrix(*args, _topo=topo, **kwargs)
        gnum = self.global_dof_numbering(topo=topo)
        return fem_coeff_matrix_coo(K_bulk, *args, inds=gnum, N=N, **kwargs)
    
    @squeeze(True)
    def body_load_vector(self, values=None, *args, source=None, 
                         constant=False, **kwargs):
        nNE = self.__class__.NNODE
        # prepare data to shape (nE, nNE * nDOF, nRHS)
        if values is None:
            values = self.loads.to_numpy()
            values = atleastnd(values, 4, back=True)  # (nE, nNE, nDOF, nRHS)
        else:
            if constant:
                values = atleastnd(values, 3, back=True)  # (nE, nDOF, nRHS)
                #np.insert(values, 1, values, axis=1)
                nE, nDOF, nRHS = values.shape
                values_ = np.zeros((nE, nNE, nDOF, nRHS), dtype=values.dtype)
                for i in range(nNE):
                    values_[:, i, :, :] = values
                values = values_
            values = atleastnd(values, 4, back=True)  # (nE, nNE, nDOF, nRHS)
        nE, _, nDOF, nRHS = values.shape
        values = values.reshape(nE, nNE * nDOF, nRHS) # (nE, nNE, nDOF, nRHS) -> (nE, nNE * nDOF, nRHS)
        values = ascont(values)
        
        if source is not None:
            raise NotImplementedError
            nE, nNE, nDOF, nRHS = values.shape
            dcm = self.direction_cosine_matrix(source=source)
            values = np.swapaxes(values, 1, 2)  # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
            values = ascont(values)
            values = tr_cells_1d_in_multi(values, dcm)
            values = np.swapaxes(values, 1, 2)  # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)
            values = ascont(values)
                
        # integrate
        nodal_loads = self.integrate_body_loads(values)  # -> (nE, nNE * nDOF, nRHS)
        # transform
        dcm = self.direction_cosine_matrix()
        nodal_loads = np.swapaxes(nodal_loads, 1, 2)  # (nE, nNE * nDOF, nRHS) -> (nE, nRHS, nNE * nDOF)
        nodal_loads = ascont(nodal_loads)
        nodal_loads = tr_cells_1d_out_multi(nodal_loads, dcm)
        nodal_loads = np.swapaxes(nodal_loads, 1, 2)  # (nE, nRHS, nNE * nDOF) -> (nE, nNE * nDOF, nRHS)
        #assemble  
        topo = self.nodes.to_numpy()
        gnum = self.global_dof_numbering(topo=topo)
        nX = len(self.pointdata) * self.__class__.NDOFN
        return assemble_load_vector(nodal_loads, gnum, nX)  # (nX, nRHS)
