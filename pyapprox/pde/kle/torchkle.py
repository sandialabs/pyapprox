import numpy as np

from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.pde.kle._kle import MeshKLE, DataDrivenKLE


class TorchMeshKLE(MeshKLE, TorchLinAlgMixin):
    pass


class TorchDataDrivenKLE(DataDrivenKLE, TorchLinAlgMixin):
    pass


class TorchInterpolatedMeshKLE(MeshKLE, TorchLinAlgMixin):
    # TODO make this work for any linalgmix in and move to _kle.py
    # This requires larger changes to autopde
    def __init__(self, kle_mesh, kle, mesh):
        self._kle_mesh = kle_mesh
        self._kle = kle
        assert isinstance(self._kle, TorchMeshKLE)
        self._mesh = mesh

        self.matern_nu = self._kle._matern_nu
        self.nterms = self._kle._nterms
        self.lenscale = self._kle._lenscale

        self._basis_mat = self._kle_mesh._get_lagrange_basis_mat(
            self._kle_mesh._canonical_mesh_pts_1d,
            mesh._map_samples_to_canonical_domain(self._mesh.mesh_pts))

    def _fast_interpolate(self, values, xx):
        assert xx.shape[1] == self._mesh.mesh_pts.shape[1]
        assert np.allclose(xx, self._mesh.mesh_pts)
        interp_vals = self._la_multidot((self._basis_mat, values))
        return interp_vals

    def __call__(self, coef):
        use_log = self._kle._use_log
        self._kle._use_log = False
        vals = self._kle(coef)
        interp_vals = self._fast_interpolate(vals, self._mesh.mesh_pts)
        mean_field = self._fast_interpolate(
            self._kle._mean_field[:, None], self._mesh.mesh_pts)
        if use_log:
            interp_vals = self._la_exp(mean_field+interp_vals)
        self._kle._use_log = use_log
        return interp_vals
