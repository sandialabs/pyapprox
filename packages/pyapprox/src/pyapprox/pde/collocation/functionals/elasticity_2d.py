"""What the 2D elasticity density terms read from a stress
post-processor: pointwise hoop stress and strain energy density for
given Lame parameters, with their state and Lame derivatives.
"""
# TODO: this is specific to collocation, should it go in
# collocation module or in benchmark module

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array


@runtime_checkable
class HoopStressSourceProtocol(Protocol, Generic[Array]):
    """What the hoop-stress density term reads: the hoop stress for given
    Lame parameters, its state Jacobian and its pointwise Lame
    derivatives. Both 2D stress post-processors satisfy it."""

    def npts(self) -> int: ...

    def hoop_stress(self, state: Array, mu: Array, lamda: Array) -> Array: ...

    def hoop_stress_state_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Array: ...

    def hoop_stress_lame_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array]: ...


@runtime_checkable
class StrainEnergySourceProtocol(Protocol, Generic[Array]):
    """What the strain-energy density term reads, as for
    :class:`HoopStressSourceProtocol`."""

    def npts(self) -> int: ...

    def strain_energy_density(self, state: Array, mu: Array, lamda: Array) -> Array: ...

    def strain_energy_density_state_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Array: ...

    def strain_energy_density_lame_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array]: ...
