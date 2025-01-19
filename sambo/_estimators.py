import warnings
from numbers import Integral, Real
from typing import Literal, Optional

import numpy as np

try:
    import sklearn  # noqa: F401
except ImportError:
    raise ImportError(
        'Missing package sklearn. Please install scikit-learn '
        '(`pip install scikit-learn`).') from None
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor as _GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.model_selection._search import BaseSearchCV

from sambo._util import _SklearnLikeRegressor, lru_cache


class _RegressorWithStdMixin:
    def predict(self, X, return_std=False):
        """
        Predict regression targets for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The mean predicted values.
        y_std : ndarray of shape (n_samples,)
            Standard deviation of predictive distribution of query points.
            Only returned when `return_std` is True.
        """
        if not return_std:
            return super().predict(X)
        preds = np.array([tree.predict(X) for tree in np.ravel(self.estimators_)])
        y_pred = preds.mean(axis=0)
        std = preds.std(axis=0, ddof=1)
        return y_pred, std


class GaussianProcessRegressor(_GaussianProcessRegressor):
    def fit(self, X, y):
        with warnings.catch_warnings(action='ignore', category=ConvergenceWarning):
            return super().fit(X, y)

    def predict(self, X, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Predicted variances smaller than 0', UserWarning)
            return super().predict(X, **kwargs)


class ExtraTreesRegressorWithStd(_RegressorWithStdMixin, ExtraTreesRegressor):
    """
    Like `ExtraTreesRegressor` from scikit-learn, but with
    `predict()` method taking an optional parameter `return_std=`.

    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
    """
    pass


class GradientBoostingRegressorWithStd(_RegressorWithStdMixin, GradientBoostingRegressor):
    """
    Like `GradientBoostingRegressor` from scikit-learn, but with
    `predict()` method taking an optional parameter `return_std=`.

    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    """
    pass


def _estimator_factory(estimator, bounds, rng):
    if isinstance(estimator, _SklearnLikeRegressor):
        return estimator
    if estimator is None:
        estimator = 'gp'

    if estimator == 'gp':
        return GaussianProcessRegressor(
            kernel=(ConstantKernel(constant_value=1, constant_value_bounds=(1e-1, 1e1)) *
                    RBF(length_scale=np.repeat(1, len(bounds)), length_scale_bounds=(1e-2, 1e2))),
            alpha=1e-14,
            copy_X_train=False,
            normalize_y=True,
            random_state=rng,
        )
    if estimator == 'et':
        return ExtraTreesRegressorWithStd(
            n_estimators=max(20, 10 * len(bounds)),
            ccp_alpha=.005,
            random_state=rng,
        )
    if estimator == 'gb':
        return GradientBoostingRegressorWithStd(
            n_estimators=20,
            max_depth=20,
            ccp_alpha=.01,
            random_state=rng,
        )
    assert False, f'Invalid estimator string: {estimator!r}'


class SamboSearchCV(BaseSearchCV):
    """
    SAMBO hyper-parameter search with cross-validation that can be
    used to **optimize hyperparameters of machine learning estimator pipelines**
    like those of scikit-learn.
    Similar to `GridSearchCV` from scikit-learn,
    but hopefully **much faster for large parameter spaces**.

    Parameters
    ----------
    estimator : BaseEstimator
        The base model or pipeline to optimize parameters for.
        It needs to implement `fit()` and `predict()` methods.

    param_grid : dict
        Dictionary with parameters names (str) as keys and lists of parameter
        choices to try as values. Supports both continuous parameter ranges and
        discrete/string parameter enumerations.

    max_iter : int, optional, default=100
        The maximum number of iterations for the optimization.

    method : {'shgo', 'sceua', 'smbo'}, optional, default='smbo'
        The optimization algorithm to use. See method `sambo.minimize()` for comparison.

    rng : int or np.random.RandomState or np.random.RandomGenerator or None, optional
        Random seed for reproducibility.

    **kwargs : dict, optional
        Additional parameters to pass to `BaseSearchCV`
        (`scoring=`, `n_jobs=`, `refit=` `cv=`, `verbose=`, `pre_dispatch=`,
        `error_score=`, `return_train_score=`). For explanation, see documentation
        on [`GridSearchCV`][skl_gridsearchcv].

        [skl_gridsearchcv]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    Attributes
    ----------
    opt_result_ : OptimizeResult
        The result of the optimization process.

    See Also
    --------
    1: https://scikit-learn.org/stable/modules/grid_search.html
    """
    def __init__(
            self,
            estimator,
            param_grid: dict,
            *,
            max_iter: int = 100,
            method: Literal['shgo', 'sceua', 'smbo'] = 'smbo',
            rng: Optional[int | np.random.RandomState | np.random.Generator] = None,
            **kwargs
    ):
        super().__init__(estimator=estimator, **kwargs)
        self.param_grid = param_grid
        self.max_iter = max_iter
        self.method = method
        self.rng = rng

    def _run_search(self, evaluate_candidates):
        import joblib

        @lru_cache(key=joblib.hash)  # TODO: lru_cache(max_iter) objective function calls always??
        def _objective(x):
            res = evaluate_candidates([dict(zip(self.param_grid.keys(), x))])
            y = -res['mean_test_score'][-1]
            nonlocal it
            it += 1
            if self.verbose:
                print(f'{self.__class__.__name__}: it={it}; y={y}; x={x}')
            return y

        bounds = [((sv := sorted(v))[0], sv[-1] + 1) if all(isinstance(i, Integral) for i in v) else
                  ((sv := sorted(v))[0], sv[-1]) if all(isinstance(i, Real) for i in v) else
                  list({i: 1 for i in v})
                  for v in self.param_grid.values()]
        kwargs = {}
        if self.max_iter is not None:
            kwargs = {'max_iter': self.max_iter}

        from ._minimize import minimize

        it = 0
        self.opt_result_ = minimize(
            _objective, bounds=bounds, method=self.method,
            disp=self.verbose, rng=0, **kwargs)
