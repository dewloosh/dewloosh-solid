# -*- coding: utf-8 -*-
from abc import abstractmethod
from inspect import signature, Parameter

from dewloosh.core.abc.meta import ABCMeta_Weak

from dewloosh.geom.cell import PolyCell

from copy import deepcopy
from functools import partial
from typing import Callable, Any


class FemMixin:    
    # must be reimplemented
    NNODE: int = None  # number of nodes, normally inherited

    # from the mesh object
    NDOFN: int = None  # numper of dofs per node, normally

    # inherited form the model object
    NDIM: int = None  # number of geometrical dimensions,
    # normally inherited from the mesh object

    # inherited form the model object
    NSTRE: int = None  # number of internal force components,

    # optional
    qrule: str = None
    quadrature = None

    # advanced settings
    compatible = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property    
    def geometry(self):
        pd, cd = self.pdb, self.db 
        return self.__class__.Geometry(pointdata=pd, celldata=cd)
    
    @property
    def pdb(self):
        return self.pointdata
    
    @property
    def db(self):
        return self._wrapped
    
    # !TODO : this should be implemented at geometry
    @classmethod
    def lcoords(cls, *args, **kwargs):
        raise NotImplementedError

    # !TODO : this should be implemented at geometry
    @classmethod
    def lcenter(cls, *args, **kwargs):
        raise NotImplementedError
    
    def local_coordinates(self, *args, frames=None, _topo=None, **kwargs):
        # implemented at PolyCell
        raise NotImplementedError

    def points_of_cells(self):
        # implemented at PolyCell
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
    def strain_displacement_matrix(self, *args, **kwargs):
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
    
    def integrate_body_loads(self, *args, **kwargs):
        raise NotImplementedError
    
    def masses(self, *args, **kwargs):
        raise NotImplementedError
    
    def volumes(self, *args, **kwargs):
        raise NotImplementedError
    
    def densities(self, *args, **kwargs):
        raise NotImplementedError
        
    def weights(self, *args, **kwargs):
        raise NotImplementedError
    


class FemModel(FemMixin):
    __dofs__ = ()
        

class FemModel1d(FemModel):
    ...


class MetaFiniteElement(ABCMeta_Weak):
    """
    Python metaclass for safe inheritance. Throws a TypeError
    if a method tries to shadow a definition in any of the base
    classes.
    """

    def __init__(self, name, bases, namespace, *args, **kwargs):
        super().__init__(name, bases, namespace, *args, **kwargs)

    def __new__(metaclass, name, bases, namespace, *args, **kwargs): 
        """cls_methods = metaclass._get_cls_methods(namespace)
        cls_abc = metaclass._get_cls_abstracts(namespace)
        base_methods = metaclass._get_base_methods(bases)
        base_abc = metaclass._get_base_abstracts(bases)"""
        
        # check if all abstract methods are implemented
        base_abc = metaclass._get_base_abstracts(bases)            
        
        # check if all abstract methods of base classes are implemented
        # by this class or any other class in the MRO hierarchy.
        
                
        cls = super().__new__(metaclass, name, bases, namespace, *args,
                              **kwargs)
        
        for base in bases:                
            if issubclass(base, PolyCell):
                cls.Geometry = base  
            elif issubclass(base, FemModel):
                cls.Model = base    
        return cls
    

class ABCFiniteElement(metaclass=MetaFiniteElement):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()
    
    
