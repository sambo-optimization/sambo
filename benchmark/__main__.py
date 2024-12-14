import sys
import time
from contextlib import redirect_stdout
from itertools import groupby

import hyperopt
import nevergrad
import numpy as np
import scipy
import scipy.optimize as scipy_optimize
import sklearn
import skopt

import sambo
from sambo._util import ObjectiveWithConstraints

from .methods import (  # noqa: F401
    ALL_METHODS,
    IS_NONCONSTRAINED,
    OPTIMIZERS_3RD_PARTY,
    OUR_METHODS,
    SCIPY_OPTIMIZE_CONSTRAINED_METHODS,
    SCIPY_OPTIMIZE_GLOBAL_OPTIMIZATION_CONSTRAINED_METHODS,
    SCIPY_OPTIMIZE_GLOBAL_OPTIMIZATION_NONCONSTRAINED_METHODS,
    SCIPY_OPTIMIZE_NONCONSTRAINED_METHODS,
    _minimize_nevergrad, _minimize_hyperopt, _minimize_skopt,
)
from .funcs import TEST_FUNCTIONS


def benchmark(method, func, x0, bounds, constraints):
    start_time = time.perf_counter()
    scipy_nonlinearconstraint = (
        None if constraints is None else
        scipy_optimize.NonlinearConstraint(lambda x, _c=constraints: float(_c(x)), .5, np.inf))

    if method in OUR_METHODS:
        method = OUR_METHODS[method]
        res = sambo.minimize(func, x0, bounds=bounds, constraints=constraints, method=method,
                             disp=True, rng=0)
    elif method in SCIPY_OPTIMIZE_CONSTRAINED_METHODS:
        bounds = np.asarray(bounds)
        res = scipy_optimize.minimize(func, x0, method=method, bounds=bounds,
                                      constraints=scipy_nonlinearconstraint)
    elif method in SCIPY_OPTIMIZE_NONCONSTRAINED_METHODS:
        func = ObjectiveWithConstraints(func, constraints)
        if method == 'CG':  # Avoid warning
            bounds = None
        res = scipy_optimize.minimize(func, x0, method=method, bounds=bounds)
    elif (method in SCIPY_OPTIMIZE_GLOBAL_OPTIMIZATION_CONSTRAINED_METHODS or
          method in SCIPY_OPTIMIZE_GLOBAL_OPTIMIZATION_NONCONSTRAINED_METHODS):
        kwargs = {
            'bounds': bounds,
        }
        if method in SCIPY_OPTIMIZE_GLOBAL_OPTIMIZATION_NONCONSTRAINED_METHODS:
            # Compare apples to apples
            func = ObjectiveWithConstraints(func, constraints)
        elif scipy_nonlinearconstraint:
            kwargs['constraints'] = scipy_nonlinearconstraint
        if method == 'shgo' and len(bounds) < 3:
            kwargs['n'] = 10  # Init at least some points to give shgo a chance
        if method == 'basinhopping':
            del kwargs['bounds']
            kwargs['x0'] = np.linspace(0, 1, len(bounds))
            kwargs['niter'] = 2000
            kwargs['niter_success'] = 20
        args = ()
        if method == 'direct':
            del kwargs['bounds']
            args = (bounds,)
        minimize_func = getattr(scipy_optimize, method)
        res = minimize_func(func, *args, **kwargs)
    elif method in OPTIMIZERS_3RD_PARTY:
        _minimize = globals()[OPTIMIZERS_3RD_PARTY[method]]
        res = _minimize(func, x0, bounds=bounds, constraints=constraints)
    print(res.message)
    duration = time.perf_counter() - start_time
    return res, duration


def main():
    for pkg, version in [
        ('numpy', np.__version__),
        ('scipy', scipy.__version__),
        ('scikit-learn', sklearn.__version__),
        ('scikit-optimize', skopt.__version__),
        ('hyperopt', hyperopt.__version__),
        ('nevergrad', nevergrad.__version__),
        ('sambo', sambo.__version__),
    ]:
        print(pkg, version)
    print()

    results = []
    for test in TEST_FUNCTIONS:
        for method in ALL_METHODS:
            np.random.seed(0)
            x0 = [np.random.uniform(*b) for b in test['bounds']]
            from sambo._util import _ObjectiveFunctionWrapper
            func = _ObjectiveFunctionWrapper(test['func'])
            with redirect_stdout(sys.stderr):
                print(method, test['name'], file=sys.stderr)
                res, duration = benchmark(method, func, x0, bounds=test['bounds'], constraints=test['constraints'])
            if func.nfev != res.nfev:
                print('res.nfev != func.nfev!', (res.nfev, func.nfev), method, test['name'], file=sys.stderr)
            y_range = (test['codomain'][1] - test['codomain'][0])
            results.append({
                'func': f"{test['name']}/{len(test['bounds'])}",
                'method': method,
                'error': (error_pct := int(min(100, round(np.nan_to_num(abs(res.fun - test['codomain'][0]) / y_range * 100))))),  # noqa: E501
                'success': error_pct <= 2,
                'nfev': res.nfev,
                'res': res,
                'duration': round(duration, 2),
            })

    sys.stderr.flush()
    sys.stdout.flush()

    def method_note(method):
        is_nonconstrained = method in SCIPY_OPTIMIZE_NONCONSTRAINED_METHODS or IS_NONCONSTRAINED(method)
        return ' \N{DAGGER}' if is_nonconstrained else ''

    header = f'{"Test function":24s}\t{"Method":24s}\tN Evals\tError %\tDuration'.expandtabs(4)
    print(header, '—'*len(header), sep='\n')
    for r in sorted(results, key=lambda r: (r['func'], r['error'], r['nfev'], r['method'])):
        print(f"{r['func']:24s}\t{r['method'] + method_note(r['method']):24s}\t{str(r['nfev']):>6s}{('' if r['success'] else '*')}\t{r['error']:7d}\t{r['duration']:5.2f}".expandtabs(4))  # noqa: E501

    print('\n')

    header = f'{"Method":24s}\t{"Correct":7s}\tN Evals\tError %\tDuration'.expandtabs(4)
    print(header, '—'*len(header), sep='\n')

    def key_func(r):
        return r['method']

    for method, nfev, success_pct, error, duration in sorted(
            [(k,
              int(round(np.mean([r['nfev'] for r in g]))),
              int(np.round(100 * np.mean([r['success'] * 1. for r in g]))),
              int(round(np.mean([r['error'] for r in g]))),
              np.percentile([r['duration'] for r in g], 90),
              )
             for k, g in [(k, list(g)) for k, g in groupby(sorted(results, key=key_func), key=key_func)]],
            key=lambda g: (-g[2], g[1], g[3])):
        print(f"{method + method_note(method):24s}\t{success_pct:6d}%\t{nfev:7d}\t{error:7d}\t{duration:8.2f}".expandtabs(4))  # noqa:E501

    print('\n')
    print('* Did not finish / unexpected result.\n'
          '\N{DAGGER} Non-constrained method.')


if __name__ == '__main__':
    main()
