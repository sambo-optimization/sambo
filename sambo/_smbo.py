from collections.abc import Iterable
from numbers import Integral, Real
from typing import Callable, Literal, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import NonlinearConstraint

from sambo._util import (
    INT32_MAX, FLOAT32_PRECISION, OptimizeResult, _ObjectiveFunctionWrapper, _SklearnLikeRegressor,
    _initialize_population,
    _sample_population,
    _check_bounds,
    _check_random_state, _sanitize_constraints, lru_cache, recompute_kde, weighted_uniform_sampling,
)


def _UCB(*, mean, std, kappa): return mean - np.outer(kappa, std)


class Optimizer:
    """
    A sequential optimizer that optimizes an objective function using a surrogate model.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float], optional
        Objective function to minimize. Must take a single array-like argument
        x (parameter combination) and return a scalar y (cost value).

        When unspecified, the Optimizer can be used iteratively in an ask-tell
        fashion using the methods named respectively.

    x0 : tuple | list[tuple], optional
        Initial guess(es) or starting point(s) for the optimization.

    args : tuple, optional
        Additional arguments to pass to the objective function and constraints.

    bounds : list[tuple], optional
        Bounds for the decision variables. A sequence of (min, max) pairs for each dimension.

    constraints : Callable[[np.ndarray], bool], optional
        Function representing constraints.
        Must return True iff the parameter combination x satisfies the constraints.

    max_iter : int, optional
        Maximum number of iterations allowed.

    n_init : int, optional
        Number of initial evaluations of the objective function before
        first fitting the surrogate model.

    n_candidates : int, optional
        Number of candidate solutions generated per iteration.

    n_iter_no_change : int, default 10
        Number of iterations with no improvement before stopping.

    n_models : int, default 1
        Number of most-recently-generated surrogate models to use for
        next best-point prediction. Useful for small and
        randomized estimators such as `"et"` with no fixed `rng=`.

    tol : float, default FLOAT32_PRECISION
        Tolerance for convergence. Optimization stops when
        found optimum improvements are below this threshold.

    estimator : {'gp', 'et', 'gb'} or scikit-learn-like regressor, default='gp'
        Surrogate model for the optimizer.
        Popular options include "gp" (Gaussian process), "et" (extra trees),
        or "gb" (gradient boosting).

        You can also provide your own regressor with a scikit-learn API,
        (namely `fit()` and `predict()` methods).

    y0 : float or tuple[float], optional
        Initial value(s) of the objective function corresponding to `x0`.

    callback : Callable[[OptimizeResult], bool], optional
        A callback function that is called after each iteration.
        The optimization stops If the callback returns True or
        raises `StopIteration`.

    n_jobs : int, default 1
        Number of objective function evaluations to run in parallel.
        Most applicate when n_candidates > 1.

    disp : bool, default False
        Display progress and intermediate results.

    rng : int or np.random.RandomState or np.random.Generator, optional
        Random number generator or seed for reproducibility.

    Examples
    --------
    >>> from sambo import Optimizer
    >>> def objective_func(x):
    ...     return sum(x**2)
    >>> optimizer = Optimizer(fun=objective_func, bounds=[(-5, 5), (-5, 5)])
    >>> result = optimizer.run()

    Using the ask-tell interface:
    >>> optimizer = Optimizer(fun=None, bounds=[(-5, 5), (-5, 5)])
    >>> suggested_x = optimizer.ask()
    >>> y = [objective_func(x) for x in suggested_x]
    >>> optimizer.tell(y, suggested_x)
    """
    def __init__(
            self,
            fun: Optional[Callable[[np.ndarray], float]],
            x0: Optional[tuple[float] | list[tuple[float]]] = None,
            *,
            args: tuple = (),
            bounds: Optional[list[tuple]] = None,
            constraints: Optional[Callable[[np.ndarray], bool] | NonlinearConstraint] = None,
            max_iter: int = INT32_MAX,
            n_init: Optional[int] = None,
            n_candidates: Optional[int] = None,
            n_iter_no_change: int = 5,
            n_models: int = 1,
            tol: float = FLOAT32_PRECISION,
            estimator: Literal['gp', 'et', 'gb'] | _SklearnLikeRegressor = None,
            y0: Optional[float | list[float]] = None,
            callback: Optional[Callable[[OptimizeResult], bool]] = None,
            n_jobs: int = 1,
            disp: bool = False,
            rng: Optional[int | np.random.RandomState | np.random.Generator] = None,
    ):
        assert fun is None or callable(fun), fun
        assert x0 is not None or bounds is not None, "Either x0= or bounds= must be provided"
        constraints = _sanitize_constraints(constraints)
        assert constraints is None or callable(constraints), constraints
        assert isinstance(max_iter, Integral) and max_iter > 0, max_iter
        assert isinstance(tol, Real) and 0 <= tol, tol
        assert isinstance(n_iter_no_change, int) and n_iter_no_change > 0, n_iter_no_change
        assert callback is None or callable(callback), callback
        assert isinstance(n_jobs, Integral) and n_jobs != 0, n_jobs
        assert isinstance(rng, (Integral, np.random.RandomState, np.random.Generator, type(None))), rng

        assert n_init is None or isinstance(n_init, Integral) and n_init >= 0, n_init
        assert n_candidates is None or isinstance(n_candidates, Integral) and n_candidates > 0, n_candidates
        assert estimator is None or isinstance(estimator, (str, _SklearnLikeRegressor)), estimator
        assert isinstance(n_models, Integral) and n_models > 0, n_models

        bounds, x0, y0 = _check_bounds(bounds, x0, y0)
        rng = _check_random_state(rng)

        if n_init is None:
            n_init = (0 if not callable(fun) else
                      min(max(1, max_iter - 20),
                          int(40 * len(bounds) * max(1, np.log2(len(bounds))))))
        assert max_iter >= n_init, (max_iter, n_init)

        if n_candidates is None:
            n_candidates = max(1, int(np.log10(len(bounds))))

        if estimator is None or isinstance(estimator, str):
            from sambo._estimators import _estimator_factory

            estimator = _estimator_factory(estimator, bounds, rng)
        assert isinstance(estimator, _SklearnLikeRegressor), estimator

        # Objective function can be None for the real-life function trials using ask-tell API
        fun = None if fun is None else _ParallelFuncWrapper(
            _ObjectiveFunctionWrapper(
                func=fun,
                max_nfev=max_iter,
                callback=callback,
                args=()),
            n_jobs, args,
        )

        self.fun = fun
        self.x0 = x0
        self.y0 = y0
        self.bounds = bounds
        self.constraints = constraints
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_candidates = n_candidates
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.estimator = estimator
        self.estimators = []
        self.n_models = n_models
        self.callback = callback
        self.n_jobs = n_jobs
        self.disp = disp
        self.rng = rng

        X, y = [], []
        if y0 is not None:
            y0 = np.atleast_1d(y0)
            assert x0 is not None and len(x0) == len(y0), (x0, y0)
            x0 = np.atleast_2d(x0)
            assert len(x0) == len(y0), (x0, y0)
            X, y = list(x0), list(y0)

        self._X_ask = []
        # Known points
        self._X = X
        self._y = y
        assert len(X) == len(y), (X, y)

        self._kde = None
        self._prev_y_min = np.inf

        # Cache methods on the _instance_
        self._init_once = lru_cache(1)(self._init_once)
        self.top_k = lru_cache(1)(self.top_k)

    def _init_once(self):
        assert not self.n_init or callable(self.fun), (self.n_init, self.fun)
        if not self.n_init:
            return
        x0, n_init = self.x0, self.n_init
        if self.y0 is not None:
            # x0, y0 already added to _X, _Y in __init__
            x0, n_init = None, max(0, self.n_init - len(self.x0))
        if n_init:
            X = _initialize_population(self.bounds, n_init, self.constraints, x0, self.rng)
            y = self.fun(X)
            self._X.extend(X)
            self._y.extend(y)
        self._fit()

    def _fit(self):
        from sklearn import clone

        estimator = self.estimator
        if self.n_models > 1 and hasattr(estimator, 'random_state'):
            estimator = clone(self.estimator)
            estimator.random_state = self.rng.randint(10000000)
        estimator.fit(self._X, self._y)

        self.estimators.append(estimator)
        if len(self.estimators) > self.n_models:
            self.estimators.pop(0)

        self.top_k.cache_clear()

    def _predict(self, X):
        means, stds, masks = [], [], []
        for estimator in self.estimators:
            X_batched = [X[i:i+10_000] for i in range(0, len(X), 10_000)]
            try:
                mean, std = np.concatenate(
                    [estimator.predict(X, return_std=True) for X in X_batched], axis=1)
            except TypeError as exc:
                if 'return_std' not in exc.args[0]:
                    raise
                mean, std = np.concatenate([estimator.predict(X) for X in X_batched]), 0
                mask = np.ones_like(mean, dtype=bool)
            else:
                # Only suggest new/unknown points
                mask = std != 0

            means.append(mean)
            stds.append(std)
            masks.append(mask)

        mask = np.any(masks, axis=0)
        mean = np.mean(means, axis=0)
        std = np.mean(stds, axis=0)

        if mask.any() and not mask.all():
            X, mean, std = X[mask], mean[mask], std[mask]

        return X, mean, std

    #: Acquisition functions for selecting the best candidates from the sample.
    #: Currently defined keys:
    #:     "UCB" for upper confidence bound (`mean - kappa * std`).
    #: [//]: # (No blank line here! bug in pdoc)
    #: .. note::
    #:      To make any use of the `kappa` parameter, it is important for the
    #:      estimator's `predict()` method to implement `return_std=` behavior.
    #:      All built-in estimators (`"gp"`, `"et"`, `"gb"`) do so.
    ACQ_FUNCS: dict = {
        'UCB': _UCB,
    }

    def ask(
            self,
            n_candidates: Optional[int] = None,
            *,
            acq_func: Optional[Callable] = ACQ_FUNCS['UCB'],
            kappa: float | list[float] = 0,
    ) -> np.ndarray:
        """
        Propose candidate solutions for the next objective evaluation based on
        the current surrogate model(s) and acquisition function.

        Parameters
        ----------
        n_candidates : int, optional
            Number of candidate solutions to propose.
            If not specified, the default value set during initialization is used.

        acq_func : Callable, default ACQ_FUNCS['UCB']
            Acquisition function used to guide the selection of candidate solutions.
            By default, upper confidence bound (i.e. `mean - kappa * std` where `mean`
            and `std` are surrogate models' predicted results).

            .. tip::
                [See the source][_ghs] for how `ACQ_FUNCS['UCB']` is implemeted.
                The passed parameters are open to extension to accommodate
                alternative acquisition functions.

                [_ghs]: https://github.com/search?q=repo%3Asambo-optimization%2Fsambo%20ACQ_FUNCS&type=code

        kappa : float or list[float], default 0
            The upper/lower-confidence-bound parameter, used by `acq_func`, that
            balances exploration (<0) vs exploitation (>0).

            Can also be an array of values to use sequentially for `n_cadidates`.

        Returns
        -------
        np.ndarray
            An array of shape `(n_candidates, n_bounds)` containing the proposed
            candidate solutions.

        Notes
        -----
        Candidates are proposed in parallel according to `n_jobs` when `n_candidates > 1`.

        Examples
        --------
        >>> candidates = optimizer.ask(n_candidates=2, kappa=2)
        >>> candidates
        array([[ 1.1, -0.2],
               [ 0.8,  0.1]])
        """
        if n_candidates is None:
            n_candidates = self.n_candidates
        assert isinstance(n_candidates, Integral) and n_candidates > 0, n_candidates
        assert isinstance(kappa, (Real, Iterable)), kappa
        self._init_once()

        n_points = min(self.MAX_POINTS_PER_ITER,
                       self.POINTS_PER_DIM * int(len(self.bounds)**2))  # TODO: Make this a param?
        nfev = len(self._X)
        if nfev < 10 * len(self.bounds)**2:
            X = _sample_population(self.bounds, n_points, self.constraints, self.rng)
        else:
            y_min = np.min(self._y)
            if self._kde is None or (nfev < 200 or nfev % 5 == 0 or y_min < self._prev_y_min):
                self._prev_y_min = y_min
                self._kde = recompute_kde(np.array(self._X), np.array(self._y))
            X = weighted_uniform_sampling(
                self._kde, self.bounds, n_points, self.constraints, self.rng)

        X, mean, std = self._predict(X)
        criterion = acq_func(mean=mean, std=std, kappa=kappa)
        n_candidates = min(n_candidates, criterion.shape[1])
        best_indices = np.take_along_axis(
            partitioned_inds := np.argpartition(criterion, n_candidates - 1)[:, :n_candidates],
            np.argsort(np.take_along_axis(criterion, partitioned_inds, axis=1)),
            axis=1).flatten('F')
        X = X[best_indices]
        X = X[:n_candidates]
        self._X_ask.extend(map(tuple, X))
        return X

    POINTS_PER_DIM = 20_000
    MAX_POINTS_PER_ITER = 80_000

    def tell(self, y: float | list[float],
             x: Optional[float | tuple[float] | list[tuple[float]]] = None):
        """
        Provide incremental feedback to the optimizer by reporting back the objective
        function values (`y`) at suggested or new candidate points (`x`).

        This allows the optimizer to refine its underlying model(s) and better
        guide subsequent proposals.

        Parameters
        ----------
        y : float or list[float]
            The observed value(s) of the objective function.

        x : float or list[float], optional
            The input point(s) corresponding to the observed objective function values `y`.
            If omitted, the optimizer assumes that the `y` values correspond
            to the most recent candidates proposed by the `ask` method (FIFO).

            .. warning::
                The function first takes `y`, then `x`, not the other way around!

        Examples
        --------
        >>> candidates = optimizer.ask(n_candidates=3)
        >>> ... # Evaluate candidate solutions IRL and tell it to the optimizer
        >>> objective_values = [1.7, 3, .8]
        >>> optimizer.tell(y=objective_values, x=candidates)
        """
        y = np.atleast_1d(y)
        assert y.ndim == 1, 'y= should be at most 1-dimensional'
        if x is None:
            if not self._X_ask:
                raise RuntimeError(
                    f'`{self.tell.__qualname__}(y, x=None)` only allowed as many '
                    f'times as `{self.ask.__qualname__}()` was called beforehand')
            for x, yval in zip(tuple(self._X_ask), y):
                self._X_ask.pop(0)
                self._X.append(x)
                self._y.append(yval)
        else:
            x = np.atleast_2d(x)
            assert len(x) == len(y), 'y= and x= (if provided) must contain the same number of items'
            for xi, yi in zip(x, y):
                try:
                    self._X_ask.pop(self._X_ask.index(tuple(xi)))
                except (ValueError, IndexError):
                    pass
                self._X.append(xi)
                self._y.append(yi)
        self._fit()

    def run(self, *,
            max_iter: Optional[int] = None,
            n_candidates: Optional[int] = None) -> OptimizeResult:
        """
        Execute the optimization process for (at most) a specified number of iterations
        (function evaluations) and return the optimization result.

        This method performs sequential optimization by iteratively proposing candidates using
        method `ask()`, evaluating the objective function, and updating the optimizer state
        with method `tell()`.
        This continues until the maximum number of iterations (`max_iter`) is reached or other
        stopping criteria are met.

        This method encapsulates the entire optimization workflow, making it convenient
        to use when you don't need fine-grained control over individual steps (`ask` and `tell`).
        It cycles between exploration and exploitation by random sampling `kappa` appropriately.

        Parameters
        ----------
        max_iter : int, optional
            The maximum number of iterations to perform. If not specified, the
            default value provided during initialization is used.

        n_candidates : int, optional
            Number of candidates to propose and evaluate in each iteration. If not specified,
            the default value provided during initialization is used.

        Returns
        -------
        OptimizeResult: OptimizeResult
            Results of the optimization process.

        Examples
        --------
        Run an optimization with a specified number of iterations:
        >>> result = optimizer.run(max_iter=30)
        >>> print(result.x, result.fun)  # Best x, y
        """
        max_iter = max_iter if max_iter is not None else 0 if self.fun is None else self.max_iter
        assert callable(self.fun) or max_iter == 0, "Can't run optimizer when fun==None. Can only use ask-tell API."
        assert n_candidates is None or isinstance(n_candidates, Integral) and n_candidates > 0, n_candidates
        assert max_iter is None or isinstance(max_iter, Integral) and max_iter >= 0, max_iter

        n_candidates = n_candidates or self.n_candidates
        success = True
        message = "Optimization hadn't been started"
        iteration = 0
        prev_best_value = np.inf
        no_change = 0
        try:
            for iteration in range(1, max_iter + 1):
                coefs = [self.rng.uniform(-2, 2) for i in range(n_candidates)]
                X = self.ask(n_candidates, kappa=coefs)
                y = self.fun(X)
                self.tell(y)

                best_value = min(self._y)
                if self.tol and prev_best_value - best_value < self.tol or prev_best_value == best_value:
                    no_change += 1
                    if no_change == self.n_iter_no_change:
                        message = 'Optimization converged (y_prev[n_iter_no_change] - y_best < tol)'
                        break
                else:
                    assert best_value < prev_best_value
                    no_change = 0
                    prev_best_value = best_value

                if self.disp:
                    print(f"{__package__}: {self.estimator.__class__.__name__} "
                          f"nit:{iteration}, nfev:{self.fun.func.nfev}, "
                          f"fun:{np.min(self._y):.5g}")
        except _ObjectiveFunctionWrapper.CallbackStopIteration:
            message = 'Optimization callback returned True'
        except _ObjectiveFunctionWrapper.MaximumFunctionEvaluationsReached:
            message = f'Maximum function evaluations reached (max_iter = {max_iter})'
            success = False
        except KeyboardInterrupt:
            message = 'KeyboardInterrupt'
            success = False

        if len(self._X) == 0 and self.fun is not None:
            # We were interrupted before ._init_once() could finish
            self._X = self.fun.func.xv
            self._y = self.fun.func.funv

        x, y = self.top_k(1)
        result = OptimizeResult(
            success=success,
            message=message,
            x=x,
            fun=y,
            nit=iteration,
            nfev=len(self._y) - (len(self.y0) if self.y0 is not None else 0),
            xv=np.array(self._X),
            funv=np.array(self._y),
            model=list(self.estimators),
        )
        return result

    def top_k(self, k: int = 1):
        """
        Based on their objective function values,
        retrieve the top-k best solutions found by the optimization process so far.

        Parameters
        ----------
        k : int, default 1
            The number of top solutions to retrieve.
            If `k` exceeds the number of evaluated solutions,
            all available solutions are returned.

        Returns
        -------
        X : np.ndarray
            A list of best points with shape `(k, n_bounds)`.
        y : np.ndarray
            Objective values at points of `X`.

        Examples
        --------
        Retrieve the best solution:
        >>> optimizer.run()
        >>> best_x, best_y = optimizer.top_k(1)
        """
        assert isinstance(k, Integral) and k > 0, k
        best_index = np.argsort(self._y)
        index = slice(0, k) if k > 1 else (k - 1)
        return self._X[best_index[index]], self._y[best_index[index]]


class _ParallelFuncWrapper:
    def __init__(self, func, n_jobs=1, args=()):
        self.parallel = Parallel(n_jobs=n_jobs, prefer='threads', require="sharedmem")
        self.func = func
        self.delayed_func = delayed(func)
        self.args = args

    def __call__(self, X, *args, **kwargs):
        X = np.atleast_2d(X)
        return np.array(self.parallel(self.delayed_func(x, *self.args, **kwargs) for x in X))


def smbo(
        fun: Callable[[np.ndarray], float],
        x0: Optional[tuple[float] | list[tuple[float]]] = None,
        *,
        args: tuple = (),
        bounds: Optional[list[tuple | list]] = None,
        constraints: Optional[Callable[[np.ndarray], bool] | NonlinearConstraint] = None,
        max_iter: int = INT32_MAX,
        n_init: Optional[int] = None,
        n_candidates: Optional[int] = None,
        n_iter_no_change: int = 5,
        n_models: int = 1,
        tol: float = FLOAT32_PRECISION,
        estimator: Optional[str | _SklearnLikeRegressor] = None,
        callback: Optional[Callable[[OptimizeResult], bool]] = None,
        y0: Optional[float | list[float]] = None,
        n_jobs: int = 1,
        disp: bool = False,
        rng: Optional[int | np.random.RandomState | np.random.Generator] = None,
        **kwargs,
):
    optimizer = Optimizer(
        fun=fun, x0=x0, y0=y0, args=args, bounds=bounds, constraints=constraints,
        max_iter=max_iter, n_init=n_init, n_candidates=n_candidates,
        n_iter_no_change=n_iter_no_change, n_models=n_models, tol=tol,
        estimator=estimator, callback=callback, n_jobs=n_jobs, disp=disp, rng=rng,
        **kwargs,
    )
    result = optimizer.run()
    return result


smbo.__doc__ == Optimizer.__doc__
