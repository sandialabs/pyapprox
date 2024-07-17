from abc import ABC, abstractmethod


class ValuesTransform(ABC):
    @abstractmethod
    def map_from_canonical(self, values):
        raise NotImplementedError

    @abstractmethod
    def map_to_canonical(self, values):
        raise NotImplementedError

    @abstractmethod
    def map_stdev_from_canonical(self, canonical_stdevs):
        raise NotImplementedError

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)


class IdentityValuesTransform(ValuesTransform):
    def map_from_canonical(self, values):
        return values

    def map_to_canonical(self, values):
        return values

    def map_stdev_from_canonical(self, canonical_stdevs):
        return canonical_stdevs


class StandardDeviationValuesTransform(ValuesTransform):
    def __init__(self):
        self._means = None
        self._stdevs = None

    def map_to_canonical(self, values):
        self._means = values.mean(axis=1)[None, :]
        self._stdevs = values.std(axis=1, ddof=1)[None, :]
        canonical_values = (values-self._means)/self._stdevs
        return canonical_values

    def map_from_canonical(self, canonical_values):
        values = canonical_values*self._stdevs + self._means
        return values

    def map_stdev_from_canonical(self, canonical_stdevs):
        return canonical_stdevs*self._stdevs


class SamplesTransform(ABC):
    @abstractmethod
    def map_from_canonical(self, values):
        raise NotImplementedError

    @abstractmethod
    def map_to_canonical(self, values):
        raise NotImplementedError


class IdentitySamplesTransform(SamplesTransform):
    def map_from_canonical(self, samples):
        return samples

    def map_to_canonical(self, samples):
        return samples
