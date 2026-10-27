import numpy as np
 




# def rotation_matrix_y(theta):
#     """
#     Computes the 3D rotation matrix about the y-axis for a given angle theta.

#     Parameters:
#     theta (float): Rotation angle in radians.

#     Returns:
#     numpy.ndarray: 3x3 rotation matrix.
#     """
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)

#     return np.array([
#         [cos_theta,  0, sin_theta],
#         [0,          1, 0],
#         [-sin_theta, 0, cos_theta]
#     ])


# def transformation_T(Q):
#     """
#     Computes the transformation matrix T (6x6) in Voigt notation
#     from a 3D rotation matrix Q.

#     Parameters:
#     Q (numpy.ndarray): 3x3 rotation matrix.

#     Returns:
#     numpy.ndarray: 6x6 transformation matrix.
#     """
#     T = np.array([
#         [Q[0,0]**2, Q[0,1]**2, Q[0,2]**2, 2*Q[0,1]*Q[0,2], 2*Q[0,0]*Q[0,2], 2*Q[0,0]*Q[0,1]],
#         [Q[1,0]**2, Q[1,1]**2, Q[1,2]**2, 2*Q[1,1]*Q[1,2], 2*Q[1,0]*Q[1,2], 2*Q[1,0]*Q[1,1]],
#         [Q[2,0]**2, Q[2,1]**2, Q[2,2]**2, 2*Q[2,1]*Q[2,2], 2*Q[2,0]*Q[2,2], 2*Q[2,0]*Q[2,1]],
#         [2*Q[1,0]*Q[2,0], 2*Q[1,1]*Q[2,1], 2*Q[1,2]*Q[2,2],
#          Q[1,1]*Q[2,2] + Q[1,2]*Q[2,1], Q[1,0]*Q[2,2] + Q[1,2]*Q[2,0], Q[1,0]*Q[2,1] + Q[1,1]*Q[2,0]],
#         [2*Q[0,0]*Q[2,0], 2*Q[0,1]*Q[2,1], 2*Q[0,2]*Q[2,2],
#          Q[0,1]*Q[2,2] + Q[0,2]*Q[2,1], Q[0,0]*Q[2,2] + Q[0,2]*Q[2,0], Q[0,0]*Q[2,1] + Q[0,1]*Q[2,0]],
#         [2*Q[0,0]*Q[1,0], 2*Q[0,1]*Q[1,1], 2*Q[0,2]*Q[1,2],
#          Q[0,1]*Q[1,2] + Q[0,2]*Q[1,1], Q[0,0]*Q[1,2] + Q[0,2]*Q[1,0], Q[0,0]*Q[1,1] + Q[0,1]*Q[1,0]],
#     ], dtype=float)
    
#     S = np.diag([1,1,1,2,2,2])
#     return np.linalg.inv(S) @ T @ S



def rotated_elasticityTensor(C, theta, dim=3):
    """
    Rotates the elasticity tensor C (6x6 in Voigt notation) by angle theta
    about the y-axis.

    Parameters:
    C (numpy.ndarray): 6x6 elasticity tensor in Voigt notation.
    theta (float): Rotation angle in radians.

    Returns:
    numpy.ndarray: Rotated 6x6 elasticity tensor in Voigt notation.
    """
    c = np.cos(theta); s = np.sin(theta)

    L = np.array([
        [c*c, 0,   s*s, 0,     2*c*s, 0],
        [0,   1,   0,   0,     0,     0],
        [s*s, 0,   c*c, 0,    -2*c*s, 0],
        [0,   0,   0,   c,     0,    -s],
        [-c*s,0,   c*s, 0,  c*c - s*s, 0],
        [0,   0,   0,   s,     0,     c]
    ], dtype=float)

    C_rot = L.dot(C).dot(L.T)   # rotated stiffness in global axes

    # Extract 2D xz plane (Voigt indices 0->xx, 2->zz, 4->xz)
    idx = [0, 2, 4]
    C2D = C_rot[np.ix_(idx, idx)]
    if dim == 2:
        return np.rint(C2D)
    return np.rint(C_rot)


  
def TTI_velocity_from_tensor(C, rho):
    """
    Computes velocities (VPV, VPH, VSV, VSH) and ETA (η) from a 3D TTI stiffness tensor.

    Parameters:
    C (numpy.ndarray): 6x6 stiffness tensor in Voigt notation.
    rho (float): Density of the material (kg/m^3).

    Returns:
    dict: Dictionary containing VPV, VPH, VSV, and ETA.
    """
    # Extract relevant stiffness coefficients
    C11 = C[0, 0]  # Horizontal P-wave stiffness
    C33 = C[2, 2]  # Vertical P-wave stiffness
    C44 = C[3, 3]  # Vertical S-wave stiffness
    C55 = C44      # Symmetry assumption: C55 = C44
    C66 = C[5,5]   # Horizontal S-wave stiffness
    C13 = C[0, 2]  # Coupling stiffness

    # Calculate velocities
    VPV = np.sqrt(C33 / rho)  # Vertical P-wave velocity
    VPH = np.sqrt(C11 / rho)  # Horizontal P-wave velocity
    VSV = np.sqrt(C55 / rho)  # Vertical S-wave velocity
    VSH = np.sqrt(C66 / rho)  # Horizontal S-wave velocity

    # Calculate epsilon
    epsilon = (C11 - C33) / (2 * C33)

    # Calculate delta
    delta = ((C13 + C44) ** 2 - (C33 - C44) ** 2) / (2 * C33 * (C33 - C44))

    # Calculate eta
    eta = (epsilon - delta) / (1 + 2 * delta)


    # Return results as a dictionary
    return {
        "RHO": rho,
        "VPV": VPV,
        "VPH": VPH,
        "VSV": VSV,
        "VSH": VSH,
        "ETA": eta
    }
    





class Austenite:
    def __init__(self):
        # Define parameters
        self.RHO = 8100  # Density in kg/m^3
        self.C11 = 217.1e9  # Stiffness in Pa
        self.C13 = 144.4e9  # Stiffness in Pa
        self.C33 = 263.2e9  # Stiffness in Pa
        self.C44 = 82.4e9   # Stiffness in Pa
        self.C66 = 128.4e9  # Stiffness in Pa

        # Derived parameters
        self.C12 = self.C11 - 2 * self.C66  # Derived from the relationship
        self.C22 = self.C11                # Symmetry in TTI media
        self.C23 = self.C13                # Symmetry in TTI media
        self.C55 = self.C44                # Symmetry in TTI media
        
        self.C = self.tensor()  

    

    def density(self):
        return self.RHO

    def tensor(self):
        C = np.zeros((6,6))
        C[0,0] = self.C11
        C[0,1] = self.C12
        C[0,2] = self.C13
        C[1,1] = self.C22
        C[1,2] = self.C23
        C[2,2] = self.C33
        C[3,3] = self.C44
        C[4,4] = self.C55
        C[5,5] = self.C66
        
        M = C + C.T                 # sum (not average)
        di = np.diag_indices_from(M)
        M[di] *= 0.5                # halve only the diagonal
        return M    
    
    
    def rotated_tensor(self, theta, dim=3):
        return rotated_elasticityTensor(self.tensor(), theta, dim=dim)
        
    def rotated_VTI_approx(self, theta):
        rotated_C = self.rotated_tensor(theta)
        params = {
            "rho": self.RHO, 
            "c11": rotated_C[0, 0],
            "c12": rotated_C[0, 0] - 2 * rotated_C[5, 5],
            "c13": rotated_C[0, 2],
            "c14": 0,
            "c15": 0,
            "c16": 0,
            "c22": rotated_C[0, 0],
            "c23": rotated_C[0, 2],
            "c24": 0,
            "c25": 0,
            "c26": 0,
            "c33": rotated_C[2, 2],
            "c34": 0,
            "c35": 0,
            "c36": 0,
            "c44": rotated_C[3, 3],
            "c45": 0,
            "c46": 0.0,
            "c55": rotated_C[3, 3],
            "c56": 0,
            "c66": rotated_C[5, 5],
        }
        return params

    def VTI_parameters(self):
        return {
        "rho": self.RHO, 
        "c11": self.C11,
        "c12": self.C12,
        "c13": self.C13,
        "c33": self.C33,
        "c44": self.C44,
        }
    
    def rotated_parameters(self, theta):
        rotated_C = self.rotated_tensor(theta)
        return {
        "rho": self.RHO, 
        "c11": rotated_C[0, 0],
        "c12": rotated_C[0, 1],
        "c13": rotated_C[0, 2],
        "c14": rotated_C[0, 3],
        "c15": rotated_C[0, 4],
        "c16": rotated_C[0, 5],
        "c22": rotated_C[1, 1],
        "c23": rotated_C[1, 2],
        "c24": rotated_C[1, 3],
        "c25": rotated_C[1, 4],
        "c26": rotated_C[1, 5],
        "c33": rotated_C[2, 2],
        "c34": rotated_C[2, 3],
        "c35": rotated_C[2, 4],
        "c36": rotated_C[2, 5],
        "c44": rotated_C[3, 3],
        "c45": rotated_C[3, 4],
        "c46": rotated_C[3, 5],
        "c55": rotated_C[4, 4],
        "c56": rotated_C[4, 5],
        "c66": rotated_C[5, 5],
        }





class Titanium:
    def __init__(self):
        # Define parameters
        self.RHO = 4500  # Density in kg/m^3
        self.C11 = 162.4e9  # Stiffness in Pa
        self.C12 = 92.9e9
        self.C13 = 69.0e9  # Stiffness in Pa
        self.C33 = 180.7e9  # Stiffness in Pa
        self.C44 = 46.7e9   # Stiffness in Pa
        self.C66 = (self.C11 - self.C12) / 2

        # Derived parameters
        self.C22 = self.C11                # Symmetry in TTI media
        self.C23 = self.C13                # Symmetry in TTI media
        self.C55 = self.C44                # Symmetry in TTI media
        
        self.C = self.tensor()  

    

    def density(self):
        return self.RHO

    def tensor(self):
        C = np.zeros((6,6))
        C[0,0] = self.C11
        C[0,1] = self.C12
        C[0,2] = self.C13
        C[1,1] = self.C22
        C[1,2] = self.C23
        C[2,2] = self.C33
        C[3,3] = self.C44
        C[4,4] = self.C55
        C[5,5] = self.C66
        
        M = C + C.T                 # sum (not average)
        di = np.diag_indices_from(M)
        M[di] *= 0.5                # halve only the diagonal
        return M    
    
    
    def rotated_tensor(self, theta, dim=3):
        return rotated_elasticityTensor(self.tensor(), theta, dim=dim)
        
    def rotated_VTI_approx(self, theta):
        rotated_C = self.rotated_tensor(theta)
        params = {
            "rho": self.RHO, 
            "c11": rotated_C[0, 0],
            "c12": rotated_C[0, 0] - 2 * rotated_C[5, 5],
            "c13": rotated_C[0, 2],
            "c14": 0,
            "c15": 0,
            "c16": 0,
            "c22": rotated_C[0, 0],
            "c23": rotated_C[0, 2],
            "c24": 0,
            "c25": 0,
            "c26": 0,
            "c33": rotated_C[2, 2],
            "c34": 0,
            "c35": 0,
            "c36": 0,
            "c44": rotated_C[3, 3],
            "c45": 0,
            "c46": 0.0,
            "c55": rotated_C[3, 3],
            "c56": 0,
            "c66": rotated_C[5, 5],
        }
        return params

    def VTI_parameters(self):
        return {
        "rho": self.RHO, 
        "c11": self.C11,
        "c12": self.C12,
        "c13": self.C13,
        "c33": self.C33,
        "c44": self.C44,
        }
    
    def rotated_parameters(self, theta):
        rotated_C = self.rotated_tensor(theta)
        return {
        "rho": self.RHO, 
        "c11": rotated_C[0, 0],
        "c12": rotated_C[0, 1],
        "c13": rotated_C[0, 2],
        "c14": rotated_C[0, 3],
        "c15": rotated_C[0, 4],
        "c16": rotated_C[0, 5],
        "c22": rotated_C[1, 1],
        "c23": rotated_C[1, 2],
        "c24": rotated_C[1, 3],
        "c25": rotated_C[1, 4],
        "c26": rotated_C[1, 5],
        "c33": rotated_C[2, 2],
        "c34": rotated_C[2, 3],
        "c35": rotated_C[2, 4],
        "c36": rotated_C[2, 5],
        "c44": rotated_C[3, 3],
        "c45": rotated_C[3, 4],
        "c46": rotated_C[3, 5],
        "c55": rotated_C[4, 4],
        "c56": rotated_C[4, 5],
        "c66": rotated_C[5, 5],
        }



