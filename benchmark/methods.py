import logging

import numpy as np
import skopt
from scipy.optimize import OptimizeResult
from skopt.callbacks import HollowIterationsStopper

SCIPY_OPTIMIZE_CONSTRAINED_METHODS = [
    'trust-constr',
    'SLSQP',
    'COBYLA',
    'COBYQA',
]

SCIPY_OPTIMIZE_NONCONSTRAINED_METHODS = [
    'Nelder-Mead',
    'Powell',
    'CG',
    'TNC',
]

SCIPY_OPTIMIZE_GLOBAL_OPTIMIZATION_CONSTRAINED_METHODS = [
    'differential_evolution',
    'shgo',
]
SCIPY_OPTIMIZE_GLOBAL_OPTIMIZATION_NONCONSTRAINED_METHODS = [
    'direct',
    'dual_annealing',
    'basinhopping',
]

# Additional non
IS_NONCONSTRAINED = {
    'dual_annealing',
    'direct',
}.__contains__

OUR_METHODS = {
    'sambo.minimize' + (i and f'({i})' or ''): i
    for i in (
        'shgo',
        'sceua',
        'smbo',
    )
}

OPTIMIZERS_3RD_PARTY = {
    'nevergrad': '_minimize_nevergrad',  # https://facebookresearch.github.io/nevergrad/
    'hyperopt': '_minimize_hyperopt',  # https://hyperopt.github.io/hyperopt/
    'scikit-optimize': '_minimize_skopt',  # https://github.com/scikit-optimize/scikit-optimize/
    'Optuna': '_minimize_optuna',  # https://optuna.org
    # The following packages were considered:
    # https://open-box.readthedocs.io -- too slow
    # https://github.com/SMTorg/smt   -- too complex
}


ALL_METHODS = (
    SCIPY_OPTIMIZE_CONSTRAINED_METHODS +
    SCIPY_OPTIMIZE_NONCONSTRAINED_METHODS +
    SCIPY_OPTIMIZE_GLOBAL_OPTIMIZATION_CONSTRAINED_METHODS +
    SCIPY_OPTIMIZE_GLOBAL_OPTIMIZATION_NONCONSTRAINED_METHODS +
    list(OPTIMIZERS_3RD_PARTY) +
    list(OUR_METHODS)
)


def _minimize_nevergrad(fun, x0, *, bounds=None, constraints=None):
    import nevergrad as ng
    if bounds is not None:
        bounds = np.asarray(bounds)
        instrumentation_params = ng.p.Array(shape=(len(x0),)).set_bounds(bounds[:, 0], bounds[:, 1])
    else:
        instrumentation_params = ng.p.Array(init=x0)
    instrumentation = ng.p.Instrumentation(instrumentation_params)

    optimizer = ng.optimizers.registry["NgIohTuned"](instrumentation, budget=3000)
    optimizer.register_callback('ask', ng.callbacks.EarlyStopping.no_improvement_stopper(300))

    def objective_with_nevergrad_expected_signature(*args):
        return fun(args[0], *args[1:])

    constraint_violation = constraints and [(lambda x, _c=constraints: not _c(x[0][0]))]
    recommendation = optimizer.minimize(
        objective_with_nevergrad_expected_signature, batch_mode=False,
        constraint_violation=constraint_violation)

    result = OptimizeResult(
        x=recommendation.value[0][0],
        fun=recommendation.loss,
        success=True,  # Assuming success as Nevergrad does not have a direct success flag
        message="Optimization terminated successfully.",
        nfev=optimizer.num_tell  # Number of function evaluations
    )
    return result


def _minimize_hyperopt(fun, x0, *, bounds=None, constraints=None, **kwargs):
    from hyperopt import fmin, tpe, hp, Trials
    from hyperopt.early_stop import no_progress_loss

    trials = Trials()  # Will store the results

    if bounds is not None:
        bounds = np.asarray(bounds)
        space = [hp.uniform(f'x{i}', b[0], b[1]) for i, b in enumerate(bounds)]
    else:
        space = hp.uniform('x', -np.inf, np.inf)

    def constrained_fun(x):
        x = np.asarray(x)
        if constraints is not None and not constraints(x):
            return np.inf
        return fun(x)

    best = fmin(fn=constrained_fun, space=space, trials=trials,
                algo=kwargs.get("algo", tpe.suggest),
                max_evals=kwargs.get("max_evals", 3000),
                early_stop_fn=no_progress_loss(500),
                rstate=np.random.default_rng(0))
    result = OptimizeResult(
        x=best,
        fun=trials.best_trial['result']['loss'],
        success=True,
        message="Optimization terminated successfully.",
        nfev=len(trials.trials),
    )
    return result


def _minimize_skopt(fun, x0, *, bounds=None, constraints=None, **kwargs):
    if bounds is not None:
        bounds = [skopt.space.Real(*b) for b in bounds]
    else:
        bounds = [skopt.space.Real(-np.inf, np.inf) for _ in range(len(x0))]
    # res = skopt.gp_minimize(
    #     lambda x: fun(np.array(x)), bounds, n_calls=1000,
    #     space_constraint=lambda x: constraints(np.array(x)),
    #     n_initial_points=20 * len(bounds),
    #     verbose=True,
    #     callback=[DeltaYStopper(1e-1, 5), DeltaXStopper(.01)])
    res = skopt.forest_minimize(
        lambda x: fun(np.array(x)), bounds, n_calls=3000,
        space_constraint=constraints and (lambda x: constraints(np.array(x))),
        n_initial_points=50 * len(bounds),
        verbose=True,
        callback=HollowIterationsStopper(150, .1))
    res['message'] = 'mock success'
    res['nfev'] = len(res.x_iters)
    res['success'] = True
    return res


def _minimize_optuna(fun, x0, *, bounds=None, constraints=None, **kwargs):
    if bounds is None:
        bounds = [(-np.inf, np.inf) for _ in range(len(x0))]

    def objective(trial):
        nonlocal bounds, fun
        x = np.array([trial.suggest_float(f'x{i}', *b)
                      for i, b in enumerate(bounds)])
        return fun(x)

    import optuna

    class StopWhenNoImprovementCallback:
        def __init__(self, thresh: int):
            self.thresh = thresh
            self._count = 0
            self._best_value = np.inf

        def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
            if trial.value < self._best_value:
                self._count = 0
                self._best_value = trial.value
            else:
                self._count += 1
            if self._count >= self.thresh:
                study.stop()

    logging.getLogger('optuna.study.study').disabled = True

    if constraints is not None:
        def constraints(trial: optuna.trial.FrozenTrial, _c=constraints):
            return [0] if _c(np.array(list(trial.params.values()))) else [np.inf]

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=0, constraints_func=constraints))
    study.optimize(objective, n_trials=3000, callbacks=[StopWhenNoImprovementCallback(150)], n_jobs=1)
    res = OptimizeResult(
        message='Ok',
        success=True,
        nfev=len(study.trials),
        x=list(study.best_params.values()),
        fun=study.best_value,
    )
    return res
