import numpy as np
from skfem import MeshQuad1

def init_gappy():
    # location of vertices
    p = np.array([[0., .2, .4, .6,  .8, 1.,
                   0., .2, .4, .6,  .8, 1.,
                   0., .2, .4, .6,  .8, 1.,
                   0., .2, .4, .6,  .8, 1.],
                  [0., 0., 0.,  0., 0., 0.,
                   1/3, 1/3, 1/3, 1/3, 1/3, 1/3,
                   2/3, 2/3, 2/3, 2/3, 2/3, 2/3,
                   1., 1., 1., 1., 1., 1.
                   ]], dtype=np.float64)
    # indices of vertices of each element
    t = np.array([[0, 1, 7, 6], [1, 2, 8, 7], [2, 3, 9, 8], [4, 5, 11, 10],
                  [6, 7, 13, 12], [8, 9, 15, 14], [9, 10, 16, 15], [10, 11, 17, 16],
                  [12, 13, 19, 18], [13, 14, 20, 19], [14, 15, 21, 20], [16, 17, 23, 22]
                  ], dtype=np.int64).T
    return MeshQuad1(p, t)
