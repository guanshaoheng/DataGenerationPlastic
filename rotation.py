import numpy as np


def rotating2PiPlane(stress):
    stress = np.array([1, 2, 3])[:, np.newaxis]
    stress = np.array([1, -2, 3])[:, np.newaxis]
    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)
    r1 = np.array([[sqrt2/2., 0., sqrt2/2], [0., 1., 0.], [-sqrt2/2., 0., sqrt2/2.]], dtype=float)
    r2 = np.array([[1., 0., 0.], [0., sqrt2/sqrt3, 1./sqrt3], [0., -1./sqrt3, sqrt2/sqrt3]], dtype=float)
    r1_inv = np.linalg.inv(r1)
    r2_inv = np.linalg.inv(r2)
    np.linalg.det(r1)
    np.linalg.det(r2)
    stress_pi = r2_inv@r1_inv@stress
