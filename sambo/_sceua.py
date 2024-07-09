from numbers import Integral, Real
from typing import Callable, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import NonlinearConstraint

from sambo._util import (
    INT32_MAX, FLOAT32_PRECISION, OptimizeResult, _ObjectiveFunctionWrapper, _initialize_population,
    _sample_population, _check_bounds, _check_random_state, _sanitize_constraints,
)


def _apply_constraints(x, bounds, constraints, centroid, rng):
    x = np.clip(x, bounds[:, 0], bounds[:, 1])
    while constraints is not None and not constraints(x):
        # doi:10.3389/feart.2022.1037173 § 2.3.3?
        for i in range(1, 1 + (T := 10)):
            x2 = x + (i / T) * (centroid - x)
            if constraints(x2):
                return x2
        x = rng.uniform(bounds[:, 0], bounds[:, 1], x.shape)
    return x


def _evolve_complex(func, args, population, func_values, complex_indices, bounds, constraints, rng):
    alpha = 1
    beta = .5
    theta = .2
    best_idx, worst_idx = 0, -1  # Index 0 is best in population. We use 1 which is best of complex

    complex_population = population[complex_indices]
    complex_values = func_values[complex_indices]
    centroid = np.mean(complex_population[:worst_idx], axis=0)

    reflection = centroid + alpha * (centroid - complex_population[worst_idx])
    reflection = (1 - theta) * reflection + theta * complex_population[best_idx]  # doi:10.1002/hyp.7082
    reflection = _apply_constraints(reflection, bounds, constraints, centroid, rng)
    reflection_value = func(reflection, *args)
    if reflection_value < complex_values[worst_idx]:
        complex_population[worst_idx] = reflection
        complex_values[worst_idx] = reflection_value
    else:
        contraction = beta * (centroid + complex_population[worst_idx])
        contraction = (1 - theta) * contraction + theta * complex_population[best_idx]  # doi:10.1002/hyp.7082
        contraction = _apply_constraints(contraction, bounds, constraints, centroid, rng)
        contraction_value = func(contraction, *args)
        if contraction_value < complex_values[best_idx]:
            complex_population[worst_idx] = contraction
            complex_values[worst_idx] = contraction_value
        else:
            n_worst = len(complex_indices) - 2  # This includes our +1 at index 0
            x = _sample_population(bounds, n_worst, constraints, rng)
            complex_population[-n_worst:] = x
            complex_values[-n_worst:] = np.apply_along_axis(func, 1, x, *args)

    sorted_indices = np.argsort(complex_values)
    complex_population = complex_population[sorted_indices]
    complex_values = complex_values[sorted_indices]

    population[complex_indices] = complex_population
    func_values[complex_indices] = complex_values


def sceua(
        fun: Callable[[np.ndarray], float],
        x0: Optional[tuple[float] | list[tuple[float]]] = None,
        *,
        args: tuple = (),
        bounds: Optional[list[tuple[float]]] = None,
        constraints: Optional[Callable[[np.ndarray], bool] | NonlinearConstraint] = None,
        n_complexes: Optional[int] = None,
        complex_size: Optional[int] = None,
        max_iter: int = INT32_MAX,
        n_iter_no_change: int = 30,
        tol: float = FLOAT32_PRECISION,
        x_tol: float = FLOAT32_PRECISION,
        y0: Optional[float | list[float]] = None,
        callback: Optional[Callable[[OptimizeResult], bool]] = None,
        n_jobs: int = 1,
        disp: bool = False,
        rng: Optional[int | np.random.RandomState | np.random.Generator] = None,
):
    assert callable(fun), fun
    assert x0 is not None or bounds is not None, "Either x0= or bounds= must be provided"
    constraints = _sanitize_constraints(constraints)
    assert constraints is None or callable(constraints), constraints
    assert n_complexes is None or isinstance(n_complexes, Integral) and n_complexes > 0, n_complexes
    assert complex_size is None or isinstance(complex_size, Integral) and complex_size > 1, complex_size
    assert isinstance(max_iter, Integral) and max_iter > 0, max_iter
    assert isinstance(tol, Real) and 0 <= tol, tol
    assert isinstance(x_tol, Real) and 0 <= x_tol, x_tol
    assert isinstance(n_iter_no_change, int) and n_iter_no_change > 0, n_iter_no_change
    assert callback is None or callable(callback), callback
    assert isinstance(n_jobs, Integral) and n_jobs != 0, n_jobs
    assert isinstance(rng, (Integral, np.random.RandomState, np.random.Generator, type(None))), rng

    bounds, x0, y0 = _check_bounds(bounds, x0, y0)
    rng = _check_random_state(rng)

    if complex_size is None:
        complex_size = 2
    if n_complexes is None:
        n_complexes = min(max(2, max_iter // complex_size - 1),
                          max(5, int(3 * np.log2(len(bounds)))))
    assert max_iter > n_complexes * complex_size, \
        ('Require `max_iter > (initial_population := n_complexes * complex_size)` '
         '-- ideally much larger!', max_iter, n_complexes, complex_size)

    parallel = Parallel(n_jobs=n_jobs, prefer='threads', require="sharedmem")
    population_size = n_complexes * complex_size
    success = True
    prev_best_value = np.inf
    no_change = 0
    iteration = -1
    fun = _ObjectiveFunctionWrapper(fun, args=args, max_nfev=max_iter, callback=callback)
    try:
        population = _initialize_population(bounds, population_size, constraints, x0, rng)
        assert len(population) == population_size, (population_size, population.shape, x0.shape)
        _n = population_size if y0 is None else population_size - len(y0)
        func_values = np.concatenate((
            [] if y0 is None else y0,
            np.array(parallel(delayed(fun)(x, *args) for x in population[-_n:])),
        ))

        for iteration in range(max_iter):
            sorted_indices = np.argsort(func_values)
            population[:] = population[sorted_indices]
            func_values[:] = func_values[sorted_indices]
            cur_best = func_values[0]
            if tol and prev_best_value - cur_best < tol or prev_best_value == cur_best:
                no_change += 1
                if no_change == n_iter_no_change:
                    message = 'Optimization converged (y_prev[n_iter_no_change] - y_best <= tol)'
                    break
            else:
                assert cur_best < prev_best_value
                no_change = 0
                prev_best_value = cur_best
            if np.sum(np.ptp(population, axis=0)) < x_tol:
                message = 'Optimization converged (sum |x_max - x_min| <= x_tol)'
                break
            if disp:
                print(f"{__package__}: SCE nit:{iteration}, nfev:{fun.nfev}, "
                      f"fun:{np.min(func_values):.5g}")
            parallel(
                delayed(_evolve_complex)(
                    fun, args, population, func_values,
                    # Our improvement: Best element is included in every complex!
                    (complex_indices := np.hstack((0, np.arange(k, population_size, n_complexes)))),  # noqa: F841
                    # np.arange(k, population_size, n_complexes),
                    bounds, constraints, rng,
                )
                for k in range(n_complexes)
            )
        else:
            assert False, 'Should not reache here'
    except _ObjectiveFunctionWrapper.CallbackStopIteration:
        message = 'Optimization callback returned True'
    except _ObjectiveFunctionWrapper.MaximumFunctionEvaluationsReached:
        message = f'Maximum function evaluations reached (max_iter = {max_iter})'
        success = False
    except KeyboardInterrupt:
        message = 'KeyboardInterrupt'
        success = False

    funv = np.asarray(fun.funv)
    xv = np.asarray(fun.xv)
    best_index = np.argmin(funv)
    result = OptimizeResult(
        x=xv[best_index],
        fun=funv[best_index],
        nit=iteration + 1,
        nfev=len(funv),
        xv=xv,
        funv=funv,
        success=success,
        message=message,
    )
    # assert np.all(result.x == result.xv[np.argmin(result.funv)]), result
    return result
