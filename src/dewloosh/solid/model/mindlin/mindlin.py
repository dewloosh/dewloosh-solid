from dewloosh.solid.model.metashell import PreShell, PreLayer

import numpy as np

__all__ = ['MindlinShell']


class MindlinShell(PreShell):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, layertype=MindlinShellLayer, **kwargs)
        self.thinness = 1.0
        self.shear_lag = 0.0

    def stiffness_matrix(self):
        ABDS = super().stiffness_matrix()
        layers = self.layers()
        A11 = ABDS[0, 0]
        B11 = ABDS[0, 3]
        D11 = ABDS[3, 3]
        S55 = ABDS[6, 6]
        A22 = ABDS[1, 1]
        B22 = ABDS[1, 4]
        D22 = ABDS[4, 4]
        S44 = ABDS[7, 7]
        eta_x = 1/(A11*D11-B11**2)
        eta_y = 1/(A22*D22-B22**2)

        # Create shear factors. These need to be multiplied with the shear
        # force in order to obtain shear stress at a given height. Since the
        # shear stress distribution is of 2nd order, the factors are
        # determined at 3 locations per layer.
        for i, layer in enumerate(layers):
            zi = layer.zi
            Exi = layer.C_126[0, 0]
            Eyi = layer.C_126[1, 1]

            # first point through the thickness
            layer.shear_factors_x[0] = layers[i-1].shear_factors_x[-1]
            layer.shear_factors_y[0] = layers[i-1].shear_factors_y[-1]

            # second point through the thickness
            layer.shear_factors_x[1] = layer.shear_factors_x[0] - \
                eta_x*Exi*(0.5*(zi[1]**2-zi[0]**2)*A11-(zi[1]-zi[0])*B11)
            layer.shear_factors_y[1] = layer.shear_factors_y[0] - \
                eta_y*Eyi*(0.5*(zi[1]**2-zi[0]**2)*A22-(zi[1]-zi[0])*B22)

            # third point through the thickness
            layer.shear_factors_x[2] = layer.shear_factors_x[0] - \
                eta_x*Exi*(0.5*(zi[2]**2-zi[0]**2)*A11-(zi[2]-zi[0])*B11)
            layer.shear_factors_y[2] = layer.shear_factors_y[0] - \
                eta_y*Eyi*(0.5*(zi[2]**2-zi[0]**2)*A22-(zi[2]-zi[0])*B22)

        # remove numerical junk from the end
        layers[-1].shear_factors_x[-1] = 0.
        layers[-1].shear_factors_y[-1] = 0.

        # prepare data for interpolation of shear stresses in a layer
        for layer in layers:
            layer.compile_shear_factors()

        # potential energy using constant stress distribution
        # and unit shear force
        pot_c_x = 0.5/S55
        pot_c_y = 0.5/S44

        # positions and weights of Gauss-points
        gP = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        gW = np.array([5/9, 8/9, 5/9])

        # potential energy using parabolic stress distribution
        # and unit shear force
        pot_p_x, pot_p_y = 0., 0.
        for layer in layers:
            dJ = 0.5*(layer.tmax - layer.tmin)
            Gxi = layer.C_45[0, 0]
            Gyi = layer.C_45[1, 1]
            for loc, weight in zip(gP, gW):
                sfx, sfy = layer.loc_to_shear_factors(loc)
                pot_p_x += 0.5*(sfx**2)*dJ*weight/Gxi
                pot_p_y += 0.5*(sfy**2)*dJ*weight/Gyi
        kx = self.thinness*pot_c_x/pot_p_x
        ky = self.thinness*pot_c_y/pot_p_y

        ABDS[6, 6] = kx * S55
        ABDS[7, 7] = ky * S44
        self.kx = kx
        self.ky = ky
        self.ABDS = ABDS
        self.SDBA = np.linalg.inv(ABDS)
        return ABDS

    def stresses_from_forces(self, *args, forces: np.ndarray = None,
                             flatten: bool = False, separate: bool = False,
                             dtype=np.float32):
        """
        Calculates stresses from a vector of generalized forces.
        """
        if separate:
            return self.stresses_from_forces_NM(*args, forces=forces,
                                                flatten=flatten,
                                                dtype=dtype)
        if forces is None:
            forcesT = np.vstack(args).T
        else:
            assert isinstance(forces, np.ndarray)
            shp = forces.shape
            forcesT = forces if shp[0] == 8 else forces.T
        assert forcesT.shape[0] == 8, "All 8 internal forces mut be specified!"

        # calculate strains
        strainsT = np.matmul(self.SDBA, forcesT)
        assert forcesT.shape == strainsT.shape

        # separate effects
        v = forcesT[6:8, :]
        e_126_0 = strainsT[0:3, :]
        curvatures = strainsT[3:6, :]
        def e_126(z): return e_126_0 + z*curvatures

        # calculate stresses
        layers = self.layers()
        res = []
        for layer in layers:
            res_layer = []
            C_126 = layer.C_126  # 3x3 matrix
            for i, z in enumerate(layer.zi):
                s_126 = np.matmul(C_126, e_126(z))
                s_45_0 = [layer.shear_factors_x[i]*v[0, :]]
                s_45_1 = [layer.shear_factors_y[i]*v[1, :]]
                stresses = np.concatenate((s_126, s_45_0, s_45_1), axis=0)
                res_layer.append(stresses)
            res.append(res_layer)

        #reshape (nForce,nPoints,5)
        if flatten:
            res = np.reshape(np.array(res, dtype=dtype),
                             (forcesT.shape[1], len(layers)*3, 5))
        else:
            res = np.reshape(np.array(res, dtype=dtype),
                             (forcesT.shape[1], len(layers), 3, 5))

        if forcesT.shape[1] == 1:
            return res[0]
        else:
            return res

    def stresses_from_forces_NM(self, *args, forces: np.ndarray = None,
                                flatten: bool = False, dtype=np.float32):
        """
        Calculates stresses from a vector of generalized forces for a number
        of records.
        """
        if forces is None:
            forcesT = np.vstack(args).T
        else:
            assert isinstance(forces, np.ndarray)
            shp = forces.shape
            forcesT = forces if shp[0] == 8 else forces.T
        assert forcesT.shape[0] == 8, "All 8 internal forces mut be specified!"

        # calculate strains
        strainsT = np.matmul(self.SDBA, forcesT)
        assert forcesT.shape == strainsT.shape

        # separate effects
        v = forcesT[6:8, :]
        e_126_n = strainsT[0:3, :]
        curvatures = strainsT[3:6, :]
        def e_126_m(z): return z * curvatures

        # calculate stresses
        layers = self.layers()
        res = []
        for layer in layers:
            res_layer = []
            C_126 = layer.C_126  # 3x3 matrix
            for i, z in enumerate(layer.zi):
                s_126_n = np.matmul(C_126, e_126_n)
                s_126_m = np.matmul(C_126, e_126_m(z))
                s_45_0 = [layer.shear_factors_x[i]*v[0, :]]
                s_45_1 = [layer.shear_factors_y[i]*v[1, :]]
                stresses = np.concatenate((s_126_n, s_126_m, s_45_0, s_45_1),
                                          axis=0)
                res_layer.append(stresses)
            res.append(res_layer)

        #reshape (nForce,nPoints,5)
        if flatten:
            res = np.reshape(np.array(res, dtype=dtype),
                             (forcesT.shape[1], len(layers) * 3, 8))
        else:
            res = np.reshape(np.array(res, dtype=dtype),
                             (forcesT.shape[1], len(layers), 3, 8))

        # eliminate dummy index
        if forcesT.shape[1] == 1:
            return res[0]
        else:
            return res


class MindlinShellLayer(PreLayer):

    __loc__ = [-1., 0., 1.]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # locations of discrete points along the thickness
        self.zi = [self.loc_to_z(loc) for loc in MindlinShellLayer.__loc__]

        # Shear factors have to be multiplied by shear force to obtain shear
        # stress. They are determined externaly at discrete points, which
        # are later used for interpolation.
        self.shear_factors_x = np.array([0., 0., 0.], dtype=np.float32)
        self.shear_factors_y = np.array([0., 0., 0.], dtype=np.float32)

        # polinomial coefficients for shear factor interpoaltion
        self.sfx = None
        self.sfy = None

    def material_stiffness_matrices(self):
        """
        Returns and stores transformed material stiffness matrices.
        """
        Cm = self.material.stiffness_matrix()
        Cm_126 = Cm[0:3, 0:3]
        Cm_45 = Cm[3:, 3:]
        T_126, T_45 = self.rotation_matrices()
        R_126 = np.diag([1, 1, 2])
        R_45 = np.diag([2, 2])
        C_126 = np.matmul(T_126,
                          np.matmul(Cm_126,
                                    np.matmul(np.linalg.inv(R_126),
                                              np.matmul(np.transpose(T_126),
                                                        R_126))))
        C_45 = np.matmul(T_45,
                         np.matmul(Cm_45,
                                   np.matmul(np.linalg.inv(R_45),
                                             np.matmul(np.transpose(T_45),
                                                       R_45))))
        C_126[np.abs(C_126) < 1e-12] = 0.0
        C_45[np.abs(C_45) < 1e-12] = 0.0
        self.C_126 = C_126
        self.C_45 = C_45
        return C_126, C_45

    def rotation_matrices(self):
        """
        Produces transformation matrices T_126 and T_45.
        """
        T_126 = np.zeros([3, 3])
        T_45 = np.zeros([2, 2])
        #
        angle = self.angle * np.pi/180
        #
        T_126[0, 0] = np.cos(angle)**2
        T_126[0, 1] = np.sin(angle)**2
        T_126[0, 2] = -np.sin(2*angle)
        T_126[1, 0] = T_126[0, 1]
        T_126[1, 1] = T_126[0, 0]
        T_126[1, 2] = -T_126[0, 2]
        T_126[2, 0] = np.cos(angle)*np.sin(angle)
        T_126[2, 1] = -T_126[2, 0]
        T_126[2, 2] = np.cos(angle)**2 - np.sin(angle)**2
        #
        T_45[0, 0] = np.cos(angle)
        T_45[0, 1] = -np.sin(angle)
        T_45[1, 1] = T_45[0, 0]
        T_45[1, 0] = -T_45[0, 1]
        #
        return T_126, T_45

    def stiffness_matrix(self):
        """
        Returns the uncorrected stiffness contribution to the layer.
        """
        C_126, C_45 = self.material_stiffness_matrices()
        tmin = self.tmin
        tmax = self.tmax
        A = C_126*(tmax - tmin)
        B = (1/2)*C_126*(tmax**2 - tmin**2)
        D = (1/3)*C_126*(tmax**3 - tmin**3)
        S = C_45*(tmax - tmin)
        S[0, 0] = S[0, 0]
        S[1, 1] = S[1, 1]
        ABDS = np.zeros([8, 8])
        ABDS[0:3, 0:3] = A
        ABDS[0:3, 3:6] = B
        ABDS[3:6, 0:3] = B
        ABDS[3:6, 3:6] = D
        ABDS[6:8, 6:8] = S
        return ABDS

    def compile_shear_factors(self):
        """
        Prepares data for continuous interpolation of shear factors. Should
        be called if shear factors are already set.
        """
        coeff_inv = np.linalg.inv(np.array([[1, z, z**2] for z in self.zi],
                                           dtype=np.float32))
        self.sfx = np.matmul(coeff_inv, self.shear_factors_x)
        self.sfy = np.matmul(coeff_inv, self.shear_factors_y)

    def loc_to_shear_factors(self, loc: float):
        """
        Returns shear factor for local z direction by quadratic interpolation.
        Local coordinate is expected between -1 and 1.
        """
        z = self.loc_to_z(loc)
        monoms = np.array([1, z, z**2], dtype=np.float32)
        return np.dot(monoms, self.sfx), np.dot(monoms, self.sfy)

    def approxfunc(self, values):
        z0, z1, z2 = self.zi
        z = np.array([[1, z0, z0**2], [1, z1, z1**2], [1, z2, z2**2]])
        a, b, c = np.matmul(np.linalg.inv(z), np.array(values))
        return lambda z: a + b*z + c*z**2


if __name__ == "__main__":
    from dewloosh.solidanics.material import Lamina

    material = Lamina(E=1000, nu=0.2, stype='shell')
    print(material.compliance_matrix())
    print(material.stiffness_matrix())

    model = MindlinShell()
    layer = model.Layer(material=material, angle=0,
                        tmin=-0.1, tmax=0.1)
    print(layer.stiffness_matrix())
