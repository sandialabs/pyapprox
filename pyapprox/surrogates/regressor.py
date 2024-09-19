from abc import abstractmethod
import pickle

from pyapprox.interface.model import Model
from pyapprox.surrogates.loss import LossFunction
from pyapprox.optimization.pya_minimize import MultiStartOptimizer
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.transforms import Transform, IdentityTransform
from pyapprox.util.visualization import get_meshgrid_samples, plot_surface


class Regressor(Model):
    def __init__(self, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        self._in_trans = IdentityTransform()
        self._out_trans = IdentityTransform()
        # canonical traning samples after transformation
        self._ctrain_samples = None
        # canonical traning values after transformation
        self._ctrain_values = None

        # TODO: consider adding transform also to OthogonalPolyBasis as
        # regressor should not have to know about constraints on
        # poly for instance Legendre samples must be in [-1, 1].
        # The regressor could apply another transform on top of this
        # but only for consistency with other regressors as
        # a second transform would not be necessary if poly
        # had its own.

        # TODO: Jacobian hessian and apply versions must be mapped from
        # canonical domain. Need to implement mappings in transforms.
        # For now make sure they are not called
        if not isinstance(self._out_trans, IdentityTransform):
            self._jacobian_implemented = False
            self._apply_jacobian_implemented = False
            self._hessian_implemented = False
            self._apply_hessian_implemented = False

    def set_input_transform(self, in_trans: Transform):
        if not isinstance(in_trans, Transform):
            raise ValueError("in_trans must be an instance of Transform")
        self._in_trans = in_trans

    def set_output_transform(self, out_trans: Transform):
        if not isinstance(out_trans, Transform):
            raise ValueError("out_trans must be an instance of Transform")
        self._out_trans = out_trans

    def _set_training_data(self, train_samples, train_values):
        if train_samples.shape[1] != train_values.shape[0]:
            raise ValueError(
                (
                    "Number of cols of samples {0} does not match"
                    + "number of rows of values"
                ).format(train_samples.shape[1], train_values.shape[0])
            )
        self._ctrain_samples = self._in_trans.map_to_canonical(train_samples)
        self._ctrain_values = self._out_trans.map_to_canonical(
            train_values.T
        ).T

    @abstractmethod
    def _fit(self, iterate):
        raise NotImplementedError

    def fit(self, train_samples, train_values, iterate=None):
        """fit the regressor"""
        self._set_training_data(train_samples, train_values)
        self._fit(iterate)

    def save(self, filename):
        '''
        To load, use pyapprox.sciml.network.load(filename)
        '''
        pickle.dump(self, open(filename, 'wb'))

    def __call__(self, samples):
        return self._out_trans.map_from_canonical(super().__call__(samples))

    def _plot_surface_1d(self, ax, qoi, plot_limits, npts_1d):
        plot_xx = self._bkd.linspace(*plot_limits, npts_1d[0])[None, :]
        ax.plot(plot_xx[0], self.__call__(plot_xx))

    def _plot_surface_2d(self, ax, qoi, plot_limits, npts_1d):
        X, Y, pts = get_meshgrid_samples(
            plot_limits, npts_1d, bkd=self._bkd
        )
        vals = self.__call__(pts)
        Z = self._bkd.reshape(vals[:, qoi], X.shape)
        plot_surface(X, Y, Z, ax)

    def plot_surface(self, ax, plot_limits, qoi=0, npts_1d=51):
        if self.nvars() > 3:
            raise RuntimeError("Cannot plot indices when nvars >= 3.")

        if not isinstance(npts_1d, list):
            npts_1d = [npts_1d]*self.nvars()

        if len(npts_1d) != self.nvars():
            raise ValueError("npts_1d must be a list")

        plot_surface_funs = {
            1: self._plot_surface_1d,
            2: self._plot_surface_2d,
        }
        plot_surface_funs[self.nvars()](ax, qoi, plot_limits, npts_1d)


class OptimizedRegressor(Regressor):
    def __init__(self, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        self._loss = None
        self._optimizer = None

    def set_loss(self, loss: LossFunction):
        if not isinstance(loss, LossFunction):
            raise ValueError(
                "loss {0} must be instance of LossFunction".format(loss)
            )
        self._loss = loss
        self._loss.set_model(self)
        self._optimizer.set_objective_function(loss)
        self._optimizer.set_bounds(self.hyp_list.get_active_opt_bounds())

    def set_optimizer(self, optimizer: MultiStartOptimizer):
        if not isinstance(optimizer, MultiStartOptimizer):
            raise ValueError(
                "optimizer {0} must be instance of MultiStartOptimizer".format(
                    optimizer
                )
            )
        self._optimizer = optimizer

    def _fit(self, iterate):
        if self._optimizer is None:
            raise RuntimeError("must call set_optimizer")
        if iterate is None:
            iterate = self._optimizer._initial_interate_gen()
        res = self._optimizer.minimize(iterate)
        active_opt_params = res.x[:, 0]
        self.hyp_list.set_active_opt_params(active_opt_params)


def QuadratureRule(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError


def TensorProductQuadratureRule(QuadratureRule):
    def __init__(self, basis):
        if not isinstance(basis, TensorProductBasis):
            raise ValueError("basis must be instance of MultiIndexBasis")
        self._basis = basis
        self._bkd = basis._bkd
        self._samples = None
        self._weights = None
        self._compute_rule()

    def _compute_rule(self):
        samples_1d, weights_1d = [], []
        for dd in range(self._basis.nvars()):
            xx, ww = self._basis.univariate_quadrature(dd)
            samples_1d.append(xx)
            weights_1d.append(ww)
        self._samples = self._bkd.cartesian_product(samples_1d)
        self._weights = self._bkd.outer_product(weights_1d)

    def __call__(self):
        return self._samples, self._weights
