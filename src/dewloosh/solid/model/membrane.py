# -*- coding: utf-8 -*-
from dewloosh.solid.model.metashell import Surface, Layer
import numpy as np

__all__ = ['Membrane']


class Membrane(Surface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, layertype=MembraneLayer, **kwargs)
        self.thinness = 1.0

    def stiffness_matrix(self):
        self.ABDS = super().stiffness_matrix()
        self.SDBA = np.linalg.inv(self.ABDS)
        return self.ABDS

    def stresses_from_forces(self, *args, forces: np.ndarray = None,
                             flatten: bool = False, dtype=np.float32):
        """
        Calculates stresses from a vector of generalized forces.
        """
        if forces is None:
            forcesT = np.vstack(args).T
        else:
            assert isinstance(forces, np.ndarray)
            shp = forces.shape
            forcesT = forces if shp[0] == 3 else forces.T
        assert forcesT.shape[0] == 3, "All 3 internal forces mut be specified!"

        # calculate strains
        strainsT = np.matmul(self.SDBA, forcesT)
        assert forcesT.shape == strainsT.shape

        # calculate stresses
        e_126 = strainsT
        layers = self.layers()
        res = []
        for layer in layers:
            res_layer = []
            C_126 = layer.C_126
            for i, z in enumerate(layer.zi):
                s_126 = np.matmul(C_126, e_126)
                res_layer.append(s_126)
            res.append(res_layer)

        #reshape (nForce, nPoints, nStress)
        if flatten:
            res = np.reshape(np.array(res, dtype=dtype),
                             (forcesT.shape[1], len(layers)*2, 3))
        else:
            res = np.reshape(np.array(res, dtype=dtype),
                             (forcesT.shape[1], len(layers), 2, 3))

        if forcesT.shape[1] == 1:
            return res[0]
        else:
            return res


class MembraneLayer(Layer):

    __loc__ = [-1., 1.]
    __shape__ = (3, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # locations of discrete points along the thickness
        self.zi = [self.loc_to_z(loc) for loc in self.__loc__]

    def material_stiffness_matrix(self):
        """
        Returns and stores the transformed material stiffness matrix.
        """
        Cm_126 = self.material.stiffness_matrix()
        T_126 = self.rotation_matrix()
        R_126 = np.diag([1, 1, 2])
        C_126 = np.matmul(T_126,
                          np.matmul(Cm_126,
                                    np.matmul(np.linalg.inv(R_126),
                                              np.matmul(np.transpose(T_126),
                                                        R_126))))
        C_126[np.abs(C_126) < 1e-12] = 0.0
        self.C_126 = C_126
        return C_126

    def rotation_matrix(self):
        """
        Returns the transformation matrix.
        """
        angle = self.angle * np.pi / 180
        T_126 = np.zeros([3, 3])
        T_126[0, 0] = np.cos(angle)**2
        T_126[0, 1] = np.sin(angle)**2
        T_126[0, 2] = -np.sin(2 * angle)
        T_126[1, 0] = T_126[0, 1]
        T_126[1, 1] = T_126[0, 0]
        T_126[1, 2] = -T_126[0, 2]
        T_126[2, 0] = np.cos(angle) * np.sin(angle)
        T_126[2, 1] = -T_126[2, 0]
        T_126[2, 2] = np.cos(angle)**2 - np.sin(angle)**2
        return T_126

    def stiffness_matrix(self):
        """
        Returns the stiffness contribution to the layer.
        """
        return self.material_stiffness_matrix() * (self.tmax - self.tmin)

    def approxfunc(self, values):
        z0, z1 = self.zi
        z = np.array([[1, z0], [1, z1]])
        a, b = np.matmul(np.linalg.inv(z), np.array(values))
        return lambda z: a + b*z


if __name__ == "__main__":
    pass
