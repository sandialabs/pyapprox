from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.kle._kle import MeshKLE, DataDrivenKLE


class NumpyMeshKLE(MeshKLE, NumpyLinAlgMixin):
    pass


class NumpyDataDrivenKLE(DataDrivenKLE, NumpyLinAlgMixin):
    pass
