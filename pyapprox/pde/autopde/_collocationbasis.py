from abc import abstractmethod


from pyapprox.surrogates.bases.basis import Basis


class CollocationBasis(Basis):
    @abstractmethod
    def nabla(self):
        raise NotImplementedError

    @abstractmethod
    def laplace(self):
        raise NotImplementedError

    @abstractmethod
    def interpolate(self):
        raise NotImplementedError


class ChebyshevCollocationBasis(CollocationBasis):
    def __init__(
            self,
            mesh, ChebyshevCollocationMesh,
    ):
        if not isinstance(mesh, ChebyshevCollocationMesh):
            raise ValueError(
                "transform must be an instance of "
                "ChebyshevCollocationMesh"
            )
        super().__init__(mesh._bkd)
        self._mesh = mesh
