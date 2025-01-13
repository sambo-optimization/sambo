from typing import Callable, Literal, Optional

import numpy as np
from scipy.optimize import NonlinearConstraint

from sambo._util import (
    OptimizeResult, _Args0TransformingFunc, INT32_MAX, FLOAT32_PRECISION, _check_bounds,
    _check_random_state, _sanitize_constraints,
)


def minimize(
        fun: Callable[[np.ndarray], float],
        x0: Optional[tuple[float] | list[tuple[float]]] = None,
        *,
        args: tuple = (),
        bounds: Optional[list[tuple]] = None,
        constraints: Optional[Callable[[np.ndarray], bool] | NonlinearConstraint] = None,
        max_iter: int = INT32_MAX,
        method: Literal['shgo', 'sceua', 'smbo'] = 'shgo',
        tol: float = FLOAT32_PRECISION,
        # x_tol: float = FLOAT32_PRECISION,
        n_iter_no_change: Optional[int] = None,
        y0: Optional[float | list[float]] = None,
        callback: Optional[Callable[[OptimizeResult], bool]] = None,
        n_jobs: int = 1,
        disp: bool = False,
        rng: Optional[int | np.random.RandomState | np.random.Generator] = None,
        **kwargs,
):
    """
    Find approximate optimum of an objective function in the
    least number of evaluations.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float], optional
        Objective function to minimize. Must take a single array-like argument
        x (parameter combination) and return a scalar y (cost value).

    x0 : tuple or list[tuple], optional
        Initial guess(es) or starting point(s) for the optimization.

    args : tuple, optional
        Additional arguments to pass to the objective function and constraints.

    bounds : list[tuple], optional
        Bounds for parameter variables.
        Should be a sequence of (min, max) pairs for each dimension,
        or an enumeration of nominal values. For any dimension,
        if `min` and `max` are integers, the dimension is assumed to be _integral_.
        If `min` or `max` are floats, the dimension is assumed to be _real_.
        In all other cases including if more than two values are provided,
        the dimension is assumed to be an _enumeration_ of values.
        See _Examples_ below.

        .. note:: Nominals are represented as ordinals
            Categorical (nominal) enumerations, although often not inherently ordered,
            are internally represented as integral dimensions.
            If this appears to significantly affect your results
            (e.g. if your nominals span many cases),
            you may need to [one-hot encode] your nominal variables manually.

        [one-hot encode]: https://en.wikipedia.org/wiki/One-hot

        .. warning:: Mind the dot
            If optimizing your problem fails to produce expected results,
            make sure you're not specifying integer dimensions where real
            floating values would make more sense.

    constraints : Callable[[np.ndarray], bool], optional
        Function representing constraints.
        Must return True iff the parameter combination x satisfies the constraints.

            >>> minimize(..., constraints=lambda x: (lb < x <= ub))

    max_iter : int, optional
        Maximum number of iterations (objective function evaluations) allowed.

    method : {'shgo', 'sceua', 'smbo'}, default='shgo'
        Global optimization algorithm to use. Options are:

        * `"shgo"` – [simplicial homology global optimization] (SHGO; from SciPy),
          [assures quick convergence] to global minimum for Lipschitz-smooth functions;
        * `"smbo"` – [surrogate model-based optimization], for which you can pass
          your own `estimator=` (see `**kwargs`), robust, but slowest of the bunch;
        * `"sceua"` – [shuffled complex evolution (SCE-UA)] (with a few tweaks,
           marked in the source), a good, time-tested all-around algorithm
           similar to [Nelder-Mead],
           provided it's initialized with sufficient `n_complexes` and `complex_size`
           kwargs ([canonical literature] suggests reference values
           `n_complexes >= 3 * len(bounds)` and
           `complex_size = 2 * len(bounds) + 1`,
           but we find good performance using `complex_size=2`,
           allowing for more complexes and more complex evolutions for given `max_iter`).

        [simplicial homology global optimization]: http://doi.org/10.1007/s10898-018-0645-y
        [assures quick convergence]: https://shgo.readthedocs.io/en/latest/docs/README.html#simplicial-homology-global-optimisation-theory
        [surrogate model-based optimization]: https://en.wikipedia.org/wiki/Surrogate_model
        [shuffled complex evolution (SCE-UA)]: https://doi.org/10.1007/BF00939380
        [Nelder-Mead]: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
        [canonical literature]: https://doi.org/10.1016/0022-1694(94)90057-4

        .. caution:: Default method SHGO is only appropriate for Lipschitz-smooth functions
            Smooth functions have gradients that vary gradually, while non-smooth functions
            exhibit abrupt changes (e.g. neighboring values of categorical variables),
            sharp corners (e.g. function `abs()`),
            discontinuities (e.g. function `tan()`),
            or unbounded growth (e.g. function `exp()`).

            If your objective function is more of the latter kind,
            you might prefer to set one of the other methods.

    n_iter_no_change : int, optional
        Number of iterations with no improvement before stopping.
        Default is method-dependent.

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

    **kwargs : dict, optional
        Additional optional parameters to pass to optimization function.
        Popular options are:

        * for `method="shgo"`: `n_init` (number of initial points), `sampling_method="halton"`,
        * for `method="smbo"`: `n_init`, `n_candidates`, `n_models`, `estimator`
          (for explanation, see class `sambo.Optimizer`),
        * for `method="sceua"`: `n_complexes`, `complex_size` (as in [SCE-UA] algorithm),

        [SCE-UA]: https://doi.org/10.1007/BF00939380

    Examples
    --------
    Basic constrained 10-dimensional example:
    >>> from scipy.optimize import rosen
    >>> from sambo import minimize
    >>> result = minimize(rosen, bounds=[(-2, 2)] * 10,
    ...                   constraints=lambda x: sum(x) <= len(x))
    >>> result
     message: Optimization terminated successfully.
     success: True
         fun: 0.0
           x: [1 1 1 1 1 1 1 1 1 1]
        nfev: 1036
          xv: [[-2 -2 ... -2 1]
               [-2 -2 ... -2 1]
               ...
               [1 1 ... 1 1]
               [1 1 ... 1 1]]
        funv: [ 1.174e+04  1.535e+04 ...  0.000e+00  0.000e+00]

    A more elaborate example, minimizing an objective function of three variables:
    one integral, one real, and one nominal variable (see `bounds=`).
    >>> def demand(x):
    ...     n_roses, price, advertising_costs = x
    ...     # Ground truth model: Demand falls with price, but grows if you advertise
    ...     demand = 20 - 2*price + .1*advertising_costs
    ...     return n_roses < demand
    >>> def objective(x):
    ...     n_roses, price, advertising_costs = x
    ...     production_costs = 1.5 * n_roses
    ...     profits = n_roses * price - production_costs - advertising_costs
    ...     return -profits
    >>> bounds = [
    ...     (0, 100),  # From zero to at most roses per day
    ...     (.5, 9.),  # Price per rose sold
    ...     (10, 20, 100),  # Advertising budget
    ... ]
    >>> from sambo import minimize
    >>> result = minimize(fun=objective, bounds=bounds, constraints=demand)

    References
    ----------
    * Endres, S.C., Sandrock, C. & Focke, W.W. A simplicial homology algorithm for Lipschitz optimisation. J Glob Optim 72, 181–217 (2018). https://doi.org/10.1007/s10898-018-0645-y
    * Duan, Q.Y., Gupta, V.K. & Sorooshian, S. Shuffled complex evolution approach for effective and efficient global minimization. J Optim Theory Appl 76, 501–521 (1993). https://doi.org/10.1007/BF00939380
    * Koziel, Slawomir, and Leifur Leifsson. Surrogate-based modeling and optimization. New York: Springer, 2013. https://doi.org/10.1007/978-1-4614-7551-4
    * Head, T., Kumar, M., Nahrstaedt, H., Louppe, G., & Shcherbatyi, I. (2021). scikit-optimize/scikit-optimize (v0.9.0). Zenodo. https://doi.org/10.5281/zenodo.5565057
    """  # noqa: E501
    from sambo._space import Space
    constraints = _sanitize_constraints(constraints)
    rng = _check_random_state(rng)
    bounds, x0, y0 = _check_bounds(bounds, x0, y0, assert_numeric=False)
    space = Space(bounds, constraints, rng=rng)
    bounds = tuple(space)

    fun = _Args0TransformingFunc(fun, space.inverse_transform)
    if constraints is not None:
        constraints = _Args0TransformingFunc(constraints, space.inverse_transform)
    if callback is not None:
        callback = _Args0TransformingFunc(callback, space.inverse_transform_result)

    if method == 'shgo':
        from sambo._shgo import shgo as minimize_func
    elif method == 'sceua':
        from sambo._sceua import sceua as minimize_func
    elif method == 'smbo':
        from sambo._smbo import smbo as minimize_func
    else:
        assert False, f'Invalid method= parameter: {method!r}. Pls RTFM'

    if n_iter_no_change is not None:
        # Pass this iff specified b/c algos have different default values
        kwargs['n_iter_no_change'] = n_iter_no_change

    res = minimize_func(
        fun, x0, args=args, bounds=bounds, constraints=constraints,
        max_iter=max_iter, tol=tol, callback=callback, y0=y0,
        n_jobs=n_jobs, disp=disp, rng=rng, **kwargs
    )
    res = space.inverse_transform_result(res)
    res.space = space
    return res
