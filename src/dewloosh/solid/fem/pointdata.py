# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.math.array import atleastnd

from dewloosh.geom.pointdata import PointData as BasePointData


class PointData(BasePointData):
    
    NDOFN = 6

    def __init__(self, *args, fixity=None, loads=None, mass=None, fields=None, 
                 **kwargs):
        
        fixity = fields.pop('fixity', fixity)
        loads = fields.pop('loads', loads)
        mass = fields.pop('mass', mass)
        
        super().__init__(*args, fields=fields, **kwargs)
        
        if self.db is not None:
            nP = len(self)
            NDOFN = self.__class__.NDOFN
            
            if fixity is None:
                fixity = np.zeros((nP, NDOFN), dtype=bool)
            else:
                assert isinstance(fixity, np.ndarray) and len(
                    fixity.shape) == 2
            self.db['fixity'] = fixity

            if loads is None:
                loads = np.zeros((nP, NDOFN, 1))
            if loads is not None:
                assert isinstance(loads, np.ndarray)
                if loads.shape[0] == nP:
                    self.db['loads'] = loads
                elif loads.shape[0] == nP * NDOFN:
                    loads = atleastnd(loads, 2, back=True)
                    self.db['loads'] = loads.reshape(
                        nP, NDOFN, loads.shape[-1])

            if mass is not None:
                if isinstance(mass, float) or isinstance(mass, int):
                    raise NotImplementedError
                elif isinstance(mass, np.ndarray):
                    self.db['mass'] = mass
