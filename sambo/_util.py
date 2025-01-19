import heapq
from functools import lru_cache as _lru_cache, wraps
from itertools import islice
from numbers import Integral, Real
from threading import Lock
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, OptimizeResult as _OptimizeResult
from scipy.stats import gaussian_kde
from scipy.stats.qmc import LatinHypercube

FLOAT32_PRECISION = 10**-np.finfo(np.float32).precision
INT32_MAX = np.iinfo(np.int32).max


def _bounds_to_tuples(bounds):
    return list(zip(bounds.lb, bounds.ub))


def _check_random_state(seed) -> np.random.RandomState | np.random.Generator:
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    if isinstance(seed, (type(None), Integral)):
        return np.random.RandomState(seed)
    raise ValueError(f"Cannot use seed={seed!r} to seed np.random.RandomState")


def _check_bounds(bounds, x0, y0, assert_numeric=True):
    assert bounds is not None or x0 is not None, \
        'Need either bounds= or x0= to minimally set the problem'
    x0 = None if x0 is None else np.atleast_2d(np.asarray(x0))
    if bounds is None:
        inf = np.finfo(np.float32).max / 2  # Avoid OverflowError https://github.com/numpy/numpy/issues/16695
        bounds = np.asarray([(-inf, inf)] * x0.shape[1])
    else:
        if isinstance(bounds, Bounds):
            bounds = _bounds_to_tuples(bounds)
        assert isinstance(bounds, (tuple, list, np.ndarray)) and len(bounds), bounds
    assert x0 is None or len(bounds) == x0.shape[1], \
        f"Shapes of x0= and bounds= mismatch ({x0.shape} vs. {bounds.shape})"
    if assert_numeric:
        bounds = np.asarray(bounds)
        assert bounds.ndim == 2 and bounds.shape[1] == 2 and np.all(bounds[:, 0] <= bounds[:, 1]), \
            "bounds= should be [(min_value, max_value), ...]"
    if y0 is not None:
        assert x0 is not None and len(x0) == len(y0), (y0, x0)
        y0 = np.atleast_1d(y0)
    return bounds, x0, y0


def lru_cache(maxsize: int = 128, key: Callable = None) -> Callable:
    """
    Decorator to LRU-cache (memoize) functions or methods based on `hash(key(*args, **kwargs))`.
    If `key` call returns None, the function result is not cached.
    """
    assert isinstance(maxsize, Integral) and maxsize > 0, maxsize
    assert key is None or callable(key), key

    if key is None:
        return _lru_cache(maxsize)

    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal key
            cache_key = key(*args, **kwargs)
            if cache_key in cache:
                return cache[cache_key]
            value = func(*args, **kwargs)
            if cache_key is not None:
                cache[cache_key] = value
                if len(cache) > maxsize:
                    cache.pop(next(iter(cache)))
            return value

        wrapper.cache_clear = cache.clear
        return wrapper

    return decorator


@runtime_checkable
class _SklearnLikeRegressor(Protocol):
    def fit(self, X, y): pass
    def predict(self, X, return_std=False): pass


class OptimizeResult(_OptimizeResult):
    """
    Optimization result. Most fields are inherited from
    `scipy.optimize.OptimizeResult`, with additional attributes: `xv`, `funv`, `model`.
    """
    success: bool  #: Whether or not the optimizer exited successfully.
    message: str  #: More detailed cause of optimization termination.
    x: np.ndarray  #: The solution of the optimization, `shape=(n_features,)`.
    fun: np.ndarray  #: Value of objective function at `x`, aka the observed minimum.
    nfev: int  #: Number of objective function evaluations.
    nit: int  #: Number of iterations performed by the optimization algorithm.
    xv: np.ndarray  #: All the parameter sets that have been tried, in sequence, `shape=(nfev, n_features)`.
    funv: np.ndarray  #: Objective function values at points `xv`.
    model: Optional[list[_SklearnLikeRegressor]]  #: The optimization model(s) used, if any.


class _ObjectiveFunctionWrapper:
    def __init__(self, func, *, args=(), max_nfev=0, callback=None, tol=None, n_iter_no_change=None):
        assert callable(func), func
        assert isinstance(args, tuple), args
        assert isinstance(max_nfev, Integral), max_nfev
        assert callback is None or callable(callback), callback
        self.func = func
        self.args = args
        self.max_nfev = max_nfev
        self.callback = callback
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        assert tol is None or n_iter_no_change is not None, (tol, n_iter_no_change)
        self.xv = []
        self.funv = []
        self._lock = Lock()

    @property
    def nfev(self):
        return len(self.funv)

    def __call__(self, *args, **kwargs):
        y = self.func(*args, **kwargs)
        assert np.ndim(y) == 0, (np.shape(y), y)
        x = args[0]
        with self._lock:
            self.xv.append(x)
            self.funv.append(y)
            if self.callback is not None:
                intermediate_result = OptimizeResult(
                    success=False,
                    message='Intermediate result',
                    x=x,
                    fun=y,
                    nfev=self.nfev,
                    # nit=n_iter,  # FIXME?
                    xv=np.array(self.xv),
                    funv=np.array(self.funv),
                )
        if self.callback is not None:
            try:
                if self.callback(intermediate_result):
                    raise self.CallbackStopIteration
            except StopIteration:
                raise self.CallbackStopIteration
        if self.nfev == self.max_nfev:
            raise self.MaximumFunctionEvaluationsReached
        if self.tol is not None and len(self.funv) >= self.n_iter_no_change:
            y_min = heapq.nsmallest(self.n_iter_no_change, self.funv)
            if y_min[0] < y_min[-1] and y_min[-1] - y_min[0] < self.tol:
                raise self.OptimizationToleranceReached
        return y

    class CallbackStopIteration(Exception):
        pass

    class MaximumFunctionEvaluationsReached(Exception):
        pass

    class OptimizationToleranceReached(Exception):
        pass


class _PrependY0:
    def __init__(self, func, x0, y0):
        self.func = func
        self.map_x0_y0 = {tuple(x): y for x, y in zip(np.atleast_2d(x0), np.atleast_1d(y0))}

    def __call__(self, x, *args, **kwargs):
        return self.map_x0_y0.get(tuple(x)) or self.func(x, *args, **kwargs)


class ObjectiveWithConstraints:
    def __init__(self, func, constraints, *, undefined_value=np.finfo(np.float32).max / 2):
        assert callable(func), func
        assert constraints is None or callable(constraints), constraints
        assert isinstance(undefined_value, Real)
        self.func = func
        self.constraints = constraints
        self.undefined_value = undefined_value

    def __call__(self, *args, **kwargs):
        cond = self.constraints is None or self.constraints(*args, **kwargs)
        return self.func(*args, **kwargs) if cond else self.undefined_value


class _Args0TransformingFunc:
    def __init__(self, func, transformer_func):
        assert callable(func), func
        assert callable(transformer_func), transformer_func
        self.func = func
        self.transformer_func = transformer_func

    def __call__(self, x, *args, **kwargs):
        x_new = self.transformer_func(x)
        res = self.func(x_new, *args, **kwargs)
        return res


def _filter_valid_points(constraints, *args) -> tuple[bool, Any]:
    if constraints is None:
        return True, *args
    valid_mask = np.apply_along_axis(lambda x: bool(constraints(x)), 1, args[0])
    if valid_mask.all():
        return True, *args
    return False, *[X[valid_mask] for X in args]


def _initialize_population(bounds, n, constraints, x0, rng):
    assert isinstance(n, Integral) and n >= 1, locals()
    population = []
    need_more = n
    if x0 is not None:
        x0 = np.atleast_2d(x0)
        if x0.size:
            assert not x0.shape[0] or x0.shape[1] == len(bounds), (x0, bounds)
            _, x0 = _filter_valid_points(constraints, x0)
            population = [x0]
            need_more -= len(x0)

    sampler = LatinHypercube(len(bounds), scramble=True, seed=rng)
    for _ in range(100_000):
        if need_more <= 0:
            # n already covered by x0
            break
        X = sampler.random(n=need_more)
        X = bounds[:, 0] + X * (bounds[:, 1] - bounds[:, 0])
        _, X = _filter_valid_points(constraints, X)
        if not X.size:
            continue
        population.append(X)
        if sum(map(len, population)) >= n:
            break
    else:
        raise RuntimeError('Constraints seemingly cannot be satisfied')
    return np.vstack(population)[:n]


def _sample_population(bounds, n_samples, constraints, rng: np.random.Generator):
    samples = []
    lo, hi = bounds.T
    for _ in range(100_000):
        # The docs for np.random.Generator.uniform() say:
        # "The high limit may be included in the returned array of floats
        # due to floating-point rounding in the equation ..."
        X = rng.uniform(lo, hi, size=(n_samples, len(bounds)))
        is_all_valid, X = _filter_valid_points(constraints, X)
        if is_all_valid:
            return X
        if not X.size:
            continue
        samples.append(X)
        if sum(map(len, samples)) >= n_samples:
            break
    else:
        raise RuntimeError('Constraints seemingly cannot be satisfied')
    samples = np.vstack(samples)[:n_samples]
    return samples


def _sanitize_constraints(constraints):
    """Return callable constraints(x) that returns >=0 where valid."""
    # Convert scipy constraints types
    if isinstance(constraints, NonlinearConstraint):
        def constraints(x, *, _c=constraints):
            return _c.lb < _c.fun(x) <= _c.ub
    elif isinstance(constraints, dict):
        match constraints.get('type'):
            case 'ineq':
                constraints = constraints['fun']
            case 'eq':
                def constraints(x, *, _c=constraints['fun']):
                    return _c(x) == 0
    return constraints


def weighted_uniform_sampling(kde, bounds, size, constraints, rng):
    """Sample points from a weighted density within given bounds"""
    rng = _check_random_state(rng)
    lb, ub = np.array(bounds).T
    if constraints is None:
        def constraints(_):
            return True
    try:
        points = np.array(
            list(islice(
                (x for _ in range(1000)
                 for x in kde.resample(size, seed=rng).T
                 if constraints(x) and np.all((lb <= x) & (x <= ub))),
                size)))
    except ValueError as ex:
        if 'too short' in str(ex):
            raise RuntimeError('Constraints seemingly cannot be satisfied.')
        raise
    return points


def recompute_kde(X, y):
    w = np.max(y) - y
    w /= np.sum(w)
    w **= 3
    kde = gaussian_kde(X.T, bw_method='silverman', weights=w)
    return kde
