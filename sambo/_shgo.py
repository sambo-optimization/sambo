from typing import Callable, Optional

import numpy as np
from scipy.optimize import shgo as _shgo, NonlinearConstraint
from scipy.stats.qmc import Halton

from sambo._util import (
    FLOAT32_PRECISION, INT32_MAX,
    OptimizeResult,
    _PrependY0, _ObjectiveFunctionWrapper, _check_bounds, _check_random_state,
    _sanitize_constraints,
)


class _ConstraintsBoolToFloat:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return float(self.func(*args, **kwargs))


class ConstrainedHaltonSampler:
    def __init__(self, bounds, constraints, x0, rng):
        self.bounds = np.asarray(bounds)
        self.constraints = constraints
        self.x0 = x0
        self.sampler = Halton(d=len(bounds), seed=rng, scramble=True)

    def __call__(self, n, _dim):
        return self.random(n)

    def random(self, n):
        Xs = []
        if self.x0 is not None:
            Xs.append(self.x0)
        for it in range(100_001):
            X = self.sampler.random(n=n)
            if self.constraints is None:
                return X
            _vals = np.apply_along_axis(
                self.constraints.fun, 1,
                self.bounds[:, 0] + X * (self.bounds[:, 1] - self.bounds[:, 0]))
            feasible = (self.constraints.lb <= _vals) & (_vals <= self.constraints.ub)  # FXIME: lb/ub as 1d ndarray?
            X = X[feasible]
            if not X.size:
                continue
            Xs.append(X)
            if sum(map(len, Xs)) >= n:
                break
        else:
            raise RuntimeError(f'Unable to sample {n} feasible points in {it} iterations')
        return np.vstack(Xs)[:n]


def shgo(
        func,
        x0: Optional[tuple[float] | list[tuple[float]]] = None,
        *,
        args: tuple = (),
        bounds: Optional[list[tuple[float]]] = None,
        constraints: Optional[Callable[[np.ndarray], bool] | NonlinearConstraint] = None,
        max_iter: int = INT32_MAX,
        n_init: Optional[int] = None,
        n_iter_no_change: int = 30,
        tol: float = FLOAT32_PRECISION,
        y0: Optional[float | list[float]] = None,
        callback: Optional[Callable[[OptimizeResult], bool]] = None,
        disp: bool = False,
        n_jobs: int = 1,
        rng: Optional[int | np.random.RandomState | np.random.Generator] = None,
        **kwargs
):
    """
    Optimize an objective function using simplical homology global optimization
    (SHGO, SciPy implementation).

    Parameters
    ----------
    fun : Callable[[np.ndarray], float], optional
        Objective function to minimize. Must take a single array-like argument
        x (parameter combination) and return a scalar y (cost value).

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

    n_iter_no_change : int, default 10
        Number of iterations with no improvement before stopping.

    tol : float, default FLOAT32_PRECISION
        Tolerance for convergence. Optimization stops when
        found optimum improvements are below this threshold.

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
    >>> from sambo import shgo
    >>> def objective_func(x):
    ...     return sum(x**2)
    >>> result = shgo(fun=objective_func, bounds=[(-5, 5), (-5, 5)])
    """
    bounds, x0, y0 = _check_bounds(bounds, x0, y0)
    rng = _check_random_state(rng)

    constraints = _sanitize_constraints(constraints)
    if callable(constraints):
        constraints = NonlinearConstraint(
            _ConstraintsBoolToFloat(constraints),
            # NOTE: NonlinearConstraint.__doc__ is lying. The condition is, in fact: lb < x <= ub !
            lb=.5, ub=np.inf)
    assert constraints is None or isinstance(constraints, NonlinearConstraint)

    wrapper = func = _ObjectiveFunctionWrapper(
        func, args=args, max_nfev=max_iter,
        callback=callback, tol=tol, n_iter_no_change=n_iter_no_change)
    if y0 is not None:
        func = _PrependY0(func, x0, y0)

    options = kwargs.pop('options', {})
    options.setdefault('disp', disp)
    options.setdefault('minhgrd', 0)
    options.setdefault('local_iter', 4)
    options.setdefault('minimize_every_iter', True)
    options.setdefault('infty_constraints', False)  # Don't use up RAM on infeasible points

    # Defaults from
    # https://github.com/scipy/scipy/blob/c7835c89f2ff593db4f68d8248f91738672d9bd4/scipy/optimize/_shgo.py#L568-L572
    minimizer_kwargs = kwargs.pop('minimizer_kwargs', {})
    minimizer_kwargs.setdefault('bounds', bounds)
    minimizer_kwargs.setdefault('method', 'SLSQP')
    minimizer_kwargs.setdefault('tol', tol)
    # minimizer_kwargs.setdefault('callback', callback)  # We have cb via _ObjectiveFunctionWrapper
    minimizer_kwargs.setdefault('options', {})
    minimizer_kwargs['options'].setdefault('ftol', tol)
    minimizer_kwargs['options'].setdefault('disp', disp)

    sampling_method = kwargs.pop('sampling_method', None)
    sampler = ConstrainedHaltonSampler(bounds=bounds, constraints=constraints, x0=x0, rng=rng)
    prefer_simplicial_sampling = len(bounds) <= 10
    if sampling_method is None:
        sampling_method = 'simplicial' if prefer_simplicial_sampling else sampler
    elif sampling_method == 'halton':
        sampling_method = sampler

    if n_init is None:
        if prefer_simplicial_sampling:
            n_init = min(int(max_iter / 2), max(80, 2**len(bounds) + 1))
        else:
            # Experientially determined
            # dims+2 <= n_init <= dims+3
            # optimal for qhull.Delaunay to work *quickly* in high-d
            n_init = len(bounds) + 3
    else:
        assert isinstance(n_init, int) and n_init >= 1, n_init
    assert max_iter is None or isinstance(max_iter, int) and n_init is None or max_iter > n_init, (max_iter, n_init)

    res = None
    try:
        res = _shgo(
            func, bounds,
            args=args,
            constraints=constraints,
            n=n_init,
            iters=1,
            options=options,
            minimizer_kwargs=minimizer_kwargs,
            sampling_method=sampling_method,
            workers=n_jobs,  # FIXME: Can't multiprocess with our unpicklable _ObjectiveFunctionWrapper
            **kwargs,
        )
    except _ObjectiveFunctionWrapper.MaximumFunctionEvaluationsReached:
        message = 'Maximum function evaluations reached'
        success = False
    except _ObjectiveFunctionWrapper.CallbackStopIteration:
        message = 'Callback returned True / raised StopIteration'
        success = True
    except _ObjectiveFunctionWrapper.OptimizationToleranceReached:
        message = 'Tolerance reached (dy < tol)'
        success = True
    except KeyboardInterrupt:
        message = 'KeyboardInterrupt'
        success = False
    else:
        message = res.message
        success = res.success

    i = np.argmin(wrapper.funv)
    res = OptimizeResult(
        success=success,
        message=message,
        x=np.array(wrapper.xv[i]),
        fun=np.array(wrapper.funv[i]),
        nit=0 if res is None else res.nit,
        nfev=len(wrapper.funv),
        xv=np.array(wrapper.xv),
        funv=np.array(wrapper.funv),
    )
    return res
