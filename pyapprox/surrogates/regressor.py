from abc import abstractmethod

from pyapprox.interface.model import Model


class Regressor(Model):
    @abstractmethod
    def fit(self, train_samples, train_values):
        raise NotImplementedError

    @staticmethod
    def _check_training_data(train_samples, train_values):
        if train_samples.shape[1] != train_values.shape[0]:
            raise ValueError(
                (
                    "Number of cols of samples {0} does not match"
                    + "number of rows of values"
                ).format(train_samples.shape[1], train_values.shape[0])
            )
        return train_samples, train_values
