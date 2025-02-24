from typing import Callable, Optional

import numpy
import numpy as np
import scipy.spatial
import scipy.optimize

from multiprocessing import Pool

from scipy.optimize import NonlinearConstraint
from scipy.stats.qmc import Sobol

from sambo._shgo import ConstrainedHaltonSampler, _ConstraintsBoolToFloat
from sambo._util import (
    FLOAT32_PRECISION, INT32_MAX, OptimizeResult, _ObjectiveFunctionWrapper, _PrependY0,
    _check_bounds,
    _check_random_state, _sanitize_constraints,
)


def tgo(
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
    """
    bounds, x0, y0 = _check_bounds(bounds, x0, y0)
    rng = _check_random_state(rng)

    constraints = _sanitize_constraints(constraints)
    if callable(constraints):
        constraints = _ConstraintsBoolToFloat(constraints)
    assert constraints is None or callable(constraints)

    wrapper = func = _ObjectiveFunctionWrapper(
        func, args=args, max_nfev=max_iter,
        callback=callback, tol=tol, n_iter_no_change=n_iter_no_change)
    if y0 is not None:
        func = _PrependY0(func, x0, y0)

    options = kwargs.pop('options', {})
    options.setdefault('disp', disp)
    options.setdefault('maxiter', max_iter)
    options.setdefault('maxfev', max_iter)

    # Defaults from
    # https://github.com/scipy/scipy/blob/c7835c89f2ff593db4f68d8248f91738672d9bd4/scipy/optimize/_shgo.py#L568-L572
    minimizer_kwargs = kwargs.pop('minimizer_kwargs', {})
    minimizer_kwargs.setdefault('method', 'SLSQP')
    minimizer_kwargs.setdefault('tol', tol)
    minimizer_kwargs.setdefault('options', {})
    minimizer_kwargs['options'].setdefault('ftol', tol)
    minimizer_kwargs['options'].setdefault('disp', disp)

    if n_init is None:
        n_init = max(10, len(bounds) ** 2)
    else:
        assert isinstance(n_init, int) and n_init >= 1, n_init
    assert max_iter is None or isinstance(max_iter, int) and n_init is None or max_iter > n_init, (max_iter, n_init)

    res = None
    try:
        res = _tgo(
            func, bounds,
            args=args,
            g_cons=constraints,
            n=n_init,
            options=options,
            minimizer_kwargs=minimizer_kwargs,
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
        # nit=0 if res is None else res.nit,
        nfev=len(wrapper.funv),
        xv=np.array(wrapper.xv),
        funv=np.array(wrapper.funv),
    )
    return res


def _tgo(func, bounds, args=(), g_cons=None, g_args=(), n=100,
        k_t=None, callback=None, minimizer_kwargs=None, options=None,
        multiproc=False):
    """
    Finds the global minima of a function using topograhphical global
    optimisation.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.

    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
        Use ``None`` for one of min or max when there is no bound in that
        direction. By default bounds are ``(None, None)``.

    args : tuple, optional
        Any additional fixed parameters needed to completely specify the
        objective function.

    g_cons : sequence of callable functions, optional
        Function(s) used to define a limited subset to defining the feasible
        set of solutions in R^n in the form g(x) <= 0 applied as g : R^n -> R^m

        NOTE: If the ``constraints`` sequence used in the local optimization
              problem is not defined in ``minimizer_kwargs`` and a constrained
              method is used then the ``g_cons`` will be used.
              (Defining a ``constraints`` sequence in ``minimizer_kwargs``
               means that ``g_cons`` will not be added so if equality
               constraints and so forth need to be added then the inequality
               functions in ``g_cons`` need to be added to ``minimizer_kwargs``
               too).

    g_args : sequence of tuples, optional
        Any additional fixed parameters needed to completely specify the
        feasible set functions ``g_cons``.
        ex. g_cons = (f1(x, *args1), f2(x, *args2))
        then
            g_args = (args1, args2)

    n : int, optional
        Number of sampling points used in the construction of the topography
        matrix.

    k_t : int, optional
        Defines the number of columns constructed in the k-t matrix. The higher
        k is the lower the amount of minimisers will be used for local search
        routines. If None the empirical model of Henderson et. al. (2015) will
        be used. (Note: Lower ``k_t`` values increase the number of local
        minimisations that need need to be performed, but could potentially be
        more robust depending on the local solver used due to testing more
        local minimisers on the function hypersuface)

    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the minimizer
        ``scipy.optimize.minimize`` Some important options could be:

            method : str
                The minimization method (e.g. ``SLSQP``)
            args : tuple
                Extra arguments passed to the objective function (``func``) and
                its derivatives (Jacobian, Hessian).

            options : {ftol: 1e-12}

    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.

    options : dict, optional
        A dictionary of solver options. All methods in scipy.optimize.minimize
        accept the following generic options:

            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.

        The following options are also used in the global routine:

            maxfev : int
                Maximum number of iterations to perform in local solvers.
                (Note only methods that support this option will terminate
                tgo at the exact specified value)

    multiproc : boolean, optional
        If True the local minimizations of the minimizer points will be pooled
        and processed in parallel using the multiprocessing module. This could
        significantly speed up slow optimizations.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are:
        ``x`` the solution array corresponding to the global minimum,
        ``fun`` the function output at the global solution,
        ``xl`` an ordered list of local minima solutions,
        ``funl`` the function output at the corresponding local solutions,
        ``success`` a Boolean flag indicating if the optimizer exited
        successfully and
        ``message`` which describes the cause of the termination,
        ``nfev`` the total number of objective function evaluations including
        the sampling calls.
        ``nlfev`` the total number of objective function evaluations
        culminating from all local search optimisations.

    Notes
    -----
    Global optimization using the Topographical Global Optimization (TGO)
    method first proposed by T??rn (1990) [1] with the the semi-empirical
    correlation by Hendorson et. al. (2015) [2] for k integer defining the
    k-t matrix.

    The TGO is a clustering method that uses graph theory to generate good
    starting points for local search methods from points distributed uniformly
    in the interior of the feasible set. These points are generated using the
    Sobol (1967) [3] sequence.

    The local search method may be specified using the ``minimizer_kwargs``
    parameter which is inputted to ``scipy.optimize.minimize``. By default
    the ``SLSQP`` method is used. In general it is recommended to use the
    ``SLSQP`` or ``COBYLA`` local minimization if inequality constraints
    are defined for the problem since the other methods do not use constraints.

    Performance can sometimes be improved by either increasing or decreasing
    the amount of sampling points ``n`` depending on the system. Increasing the
    amount of sampling points can lead to a lower amount of minimisers found
    which requires fewer local optimisations. Forcing a low ``k_t`` value will
    nearly always increase the amount of function evaluations that need to be
    performed, but could lead to increased robustness.

    The primitive polynomials and various sets of initial direction numbers for
    generating Sobol sequences is provided by [4] by Frances Kuo and
    Stephen Joe. The original program sobol.cc is available and described at
    http://web.maths.unsw.edu.au/~fkuo/sobol/ translated to Python 3 by
    Carl Sandrock 2016-03-31

    Examples
    --------
    First consider the problem of minimizing the Rosenbrock function. This
    function is implemented in `rosen` in `scipy.optimize`

    >>> from scipy.optimize import rosen
    >>> from tgo import tgo
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = tgo(rosen, bounds)
    >>> result.x, result.fun
    (array([ 1.,  1.,  1.,  1.,  1.]), 2.9203923741900809e-18)

    Note that bounds determine the dimensionality of the objective
    function and is therefore a required input, however you can specify
    empty bounds using ``None`` or objects like numpy.inf which will be
    converted to large float numbers.

    >>> bounds = [(None, None), (None, None), (None, None), (None, None)]
    >>> result = tgo(rosen, bounds)
    >>> result.x
    array([ 0.99999851,  0.99999704,  0.99999411,  0.9999882 ])

    Next we consider the Eggholder function, a problem with several local
    minima and one global minimum.
    (https://en.wikipedia.org/wiki/Test_functions_for_optimization)

    >>> from tgo import tgo
    >>> import numpy as np
    >>> def eggholder(x):
    ...     return (-(x[1] + 47.0)
    ...             * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
    ...             - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
    ...             )
    ...
    >>> bounds = [(-512, 512), (-512, 512)]
    >>> result = tgo(eggholder, bounds)
    >>> result.x, result.fun
    (array([ 512.        ,  404.23180542]), -959.64066272085051)

    ``tgo`` also has a return for any other local minima that was found, these
     can be called using:

    >>> result.xl, result.funl
    (array([[ 512.        ,  404.23180542],
           [-456.88574619, -382.6233161 ],
           [ 283.07593402, -487.12566542],
           [ 324.99187533,  216.0475439 ],
           [-105.87688985,  423.15324143],
           [-242.97923629,  274.38032063],
           [-414.8157022 ,   98.73012628],
           [ 150.2320956 ,  301.31377513],
           [  91.00922754, -391.28375925],
           [ 361.66626134, -106.96489228]]),
           array([-959.64066272, -786.52599408, -718.16745962, -582.30628005,
           -565.99778097, -559.78685655, -557.85777903, -493.9605115 ,
           -426.48799655, -419.31194957]))

    Now suppose we want to find a larger amount of local minima, this can be
    accomplished for example by increasing the amount of sampling points...

    >>> result_2 = tgo(eggholder, bounds, n=1000)
    >>> len(result.xl), len(result_2.xl)
    (10, 60)

    ...or by lowering the k_t value:

    >>> result_3 = tgo(eggholder, bounds, k_t=1)
    >>> len(result.xl), len(result_2.xl), len(result_3.xl)
    (10, 60, 48)

    To demonstrate solving problems with non-linear constraints consider the
    following example from [5] (Hock and Schittkowski problem 18):

    Minimize: f = 0.01 * (x_1)**2 + (x_2)**2

    Subject to: x_1 * x_2 - 25.0 >= 0,
                (x_1)**2 + (x_2)**2 - 25.0 >= 0,
                2 <= x_1 <= 50,
                0 <= x_2 <= 50.

    Approx. Answer:
        f([(250)**0.5 , (2.5)**0.5]) = 5.0

    >>> from scipy.optimize import tgo
    >>> def f(x):
    ...     return 0.01 * (x[0])**2 + (x[1])**2
    ...
    >>> def g1(x):
    ...     return x[0] * x[1] - 25.0
    ...
    >>> def g2(x):
    ...     return x[0]**2 + x[1]**2 - 25.0
    ...
    >>> g = (g1, g2)
    >>> bounds = [(2, 50), (0, 50)]
    >>> result = tgo(f, bounds, g_cons=g)
    >>> result.x, result.fun
    (array([ 15.81138847,   1.58113881]), 4.9999999999996252)


    References
    ----------
    .. [1] T??rn, A (1990) "Topographical global optimization", Reports on
           Computer Science and Mathematics Ser. A, No 199, 8p. Abo Akademi
           University, Sweden
    .. [2] Henderson, N, de S?? R??go, M, Sacco, WF, Rodrigues, RA Jr. (2015) "A
           new look at the topographical global optimization method and its
           application to the phase stability analysis of mixtures",
           Chemical Engineering Science, 127, 151-174
    .. [3] Sobol, IM (1967) "The distribution of points in a cube and the
           approximate evaluation of integrals. USSR Comput. Math. Math. Phys.
           7, 86-112.
    .. [4] S. Joe and F. Y. Kuo (2008) "Constructing Sobol sequences with
           better  two-dimensional projections", SIAM J. Sci. Comput. 30,
           2635-2654
    .. [5] Hoch, W and Schittkowski, K (1981) "Test examples for nonlinear
           programming codes." Lecture Notes in Economics and mathematical
           Systems, 187. Springer-Verlag, New York.
           http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    """
    # Initiate TGO class
    TGOc = TGO(func, bounds, args=args, g_cons=g_cons, g_args=g_args, n=n,
               k_t=k_t, callback=callback, minimizer_kwargs=minimizer_kwargs,
               options=options, multiproc=multiproc)

    sample = True
    while sample:
        # Generate sampling points
        if TGOc.disp:
            print('Generating sampling points')

        TGOc.sampling()
        # Find subspace of feasible points
        if g_cons is not None:
            TGOc.subspace()
        else:
            TGOc.fn = TGOc.n

        # Find topograph
        if TGOc.disp:
            print('Constructing topograph')

        TGOc.topograph()

        # Check for tabletop and nan values
        if not TGOc.T.any() == True:
            if TGOc.disp:
                print('No minimizers found. Increasing sampling space.')

            n_add = 100
            if options is not None:
                if 'maxiter' in options.keys():
                    n_add = int((options['maxiter'] - TGOc.fn)/1.618)
                    if n_add < 1:
                        TGOc.res.message = ("Failed to find a minimizer "
                                            "within the maximum allowed "
                                            "function evaluations.")
                        if TGOc.disp:
                            print(TGOc.res.message + " Breaking routine...")

                        TGOc.break_routine = True
                        TGOc.res.success = False
                        sample = False

            TGOc.n += n_add

            # Include each sampling point as func evaluation:
            TGOc.res.nfev = TGOc.fn
        else:  # If good values are found stop while loop
            # Include each sampling point as func evaluation:
            TGOc.res.nfev = TGOc.fn
            sample = False

    if not TGOc.break_routine:
        if TGOc.disp:
            print("Succesfully completed construction of topograph, "
                  "starting local minimizations.")

        # Find the optimal k+ topograph
        # Find epsilon_i parameter for current system
        if k_t is None:
            TGOc.K_opt = TGOc.K_optimal()
        else:
            TGOc.K_opt = TGOc.k_t_matrix(TGOc.T, k_t)

    if not TGOc.break_routine:
        # Local Search: Find the minimiser float values and func vals.
        TGOc.l_minima()

    # Confirm the routine ran succesfully
    if not TGOc.break_routine:
        TGOc.res.message = 'Optimization terminated successfully.'
        TGOc.res.success = True

    # Add local func evals to sampling func evals
    TGOc.res.nfev += TGOc.res.nlfev

    return TGOc.res


# %% Define tgo class
class TGO(object):
    """
    This class implements the tgo routine
    """

    def __init__(self, func, bounds, args=(), g_cons=None, g_args=(), n=100,
                 k_t=None, callback=None, minimizer_kwargs=None,
                 options=None, multiproc=False):

        self.func = func
        self.bounds = bounds
        self.args = args
        if type(g_cons) is not tuple and type(g_cons) is not list:
            self.g_func = (g_cons,)
        else:
            self.g_func = g_cons

        self.g_args = g_args
        self.n = n
        self.n_sampled = 0  # To track sampling points already evaluated
        self.k_t = k_t
        if k_t is not None:
            self.k_opt = k_t

        self.callback = callback
        self.maxfev = None
        self.disp = False
        if options is not None:
            if 'maxfev' in options:
                self.maxfev = options['maxfev']
            if 'disp' in options:
                self.disp = options['disp']

        # set bounds
        abound = numpy.array(bounds, float)
        # Check if bounds are correctly specified
        if abound.ndim > 1:
            bnderr = numpy.where(abound[:, 0] > abound[:, 1])[0]
            # Set none finite values to large floats
            infind = ~numpy.isfinite(abound)
            abound[infind[:, 0], 0] = -1e50  # e308
            abound[infind[:, 1], 1] = 1e50  # e308
        else:
            bnderr = numpy.where(abound[0] > abound[1])[0]
            # Set none finite values to large floats
            infind = ~numpy.isfinite(abound)
            infind = numpy.asarray(infind, dtype=int)

            abound[infind[0]] = -1e50  # e308
            abound[infind[1]] = 1e50  # e308

        if bnderr.any():
            raise ValueError('Error: lb > ub in bounds %s.' %
                             ', '.join(str(b) for b in bnderr))

        self.bounds = abound

        # Define constraint function used in local minimisation
        if g_cons is not None:
            self.min_cons = []
            for g in self.g_func:
                self.min_cons.append({'type': 'ineq',
                                      'fun': g})

        # Define local minimization keyword arguments
        if minimizer_kwargs is not None:
            self.minimizer_kwargs = minimizer_kwargs
            if 'args' not in minimizer_kwargs:
                self.minimizer_kwargs['args'] = self.args

            if 'method' not in minimizer_kwargs:
                self.minimizer_kwargs['method'] = 'SLSQP'

            if 'bounds' not in minimizer_kwargs:
                self.minimizer_kwargs['bounds'] = self.bounds

            if 'options' not in minimizer_kwargs:
                minimizer_kwargs['options'] = {'ftol': 1e-12}

                if options is not None:
                    if 'ftol' in options:
                        self.minimizer_kwargs['options']['ftol'] = \
                            options['ftol']
                    if 'maxfev' in options:
                        self.minimizer_kwargs['options']['maxfev'] = \
                            options['maxfev']
                    if 'disp' in options:
                        self.minimizer_kwargs['options']['disp'] = \
                            options['disp']

            if 'callback' not in minimizer_kwargs:
                minimizer_kwargs['callback'] = self.callback

            if self.minimizer_kwargs['method'] == 'SLSQP' or \
                            self.minimizer_kwargs['method'] == 'COBYLA':
                if 'constraints' not in minimizer_kwargs:
                    if g_cons is not None:
                        minimizer_kwargs['constraints'] = self.min_cons
        else:
            self.minimizer_kwargs = {'args': self.args,
                                     'method': 'SLSQP',
                                     'bounds': self.bounds,
                                     'options': {'ftol': 1e-12},
                                     'callback': self.callback
                                     }

            if g_cons is not None:
                self.minimizer_kwargs['constraints'] = self.min_cons

            if options is not None:
                if 'ftol' in options:
                    self.minimizer_kwargs['options']['ftol'] = \
                        options['ftol']
                if 'maxfev' in options:
                    self.minimizer_kwargs['options']['maxfev'] = \
                        options['maxfev']
                if 'disp' in options:
                    self.minimizer_kwargs['options']['disp'] = options['disp']

        # Remove unknown solver options to avoid OptimizeWarning:
        if self.minimizer_kwargs['method'] == 'SLSQP' or \
                        self.minimizer_kwargs['method'] == 'COBYLA':
            try:
                del self.minimizer_kwargs['options']['maxfev']
            except KeyError:
                pass

        self.break_routine = False

        self.multiproc = multiproc

        # Pop unknown arguments in self.minimizer_kwargs
        method = self.minimizer_kwargs['method']
        meth = method.lower()
        print('='*100)
        print('meth = {}'.format(meth))
        if meth == '_custom':
            self.min_solver_args = ['fun', 'x0', 'args', 'jac', 'hess',
                                    'hessp', 'bounds',
                                    'constraints', 'callback']
        elif meth == 'nelder-mead':
            self.min_solver_args = ['fun', 'x0', 'args', 'callback']
        elif meth == 'powell':
            self.min_solver_args = ['fun', 'x0', 'args', 'callback']
        elif meth == 'cg':
            self.min_solver_args = ['fun', 'x0', 'args', 'jac', 'callback']
        elif meth == 'bfgs':
            self.min_solver_args = ['fun', 'x0', 'args', 'jac', 'callback']
        elif meth == 'newton-cg':
            self.min_solver_args = ['fun', 'x0', 'args', 'jac', 'hess',
                                    'hessp', 'callback']
        elif meth == 'l-bfgs-b':
            self.min_solver_args = ['fun', 'x0', 'args', 'jac', 'bounds',
                                    'callback']
        elif meth == 'tnc':
            self.min_solver_args = ['fun', 'x0', 'args', 'jac', 'bounds',
                                    'callback']
        elif meth == 'cobyla':
            self.min_solver_args = ['fun', 'x0', 'args', 'constraints']
        elif meth == 'slsqp':
            self.min_solver_args = ['fun', 'x0', 'args', 'jac', 'bounds',
                                    'constraints', 'callback']
        elif meth == 'dogleg':
            self.min_solver_args = ['fun', 'x0', 'args', 'jac', 'hess',
                                    'callback']
        elif meth == 'trust-ncg':
            self.min_solver_args = ['fun', 'x0', 'args', 'jac', 'hess',
                                    'hessp', 'callback']
        elif meth == 'trust-krylov':
            self.min_solver_args = ['fun', 'x0', 'args', 'jac', 'hess',
                                    'hessp', 'callback']
        elif meth == 'trust-exact':
            self.min_solver_args = ['fun', 'x0', 'args', 'jac', 'hess',
                                    'callback']

        self.min_solver_args.append('options')
        self.min_solver_args.append('method')
        kwarg_dict = self.minimizer_kwargs.copy()
        print(kwarg_dict)
        kwarg_opt_dict = self.minimizer_kwargs['options'].copy()
        f_tol_excluded_list = ['nelder-mead', 'cg', 'bfgs', 'newton-cg',
                               'dogleg', 'trust-ncg', 'trust-exact',
                               'trust-krylov', 'cobyla']
        for key in kwarg_dict:
            if key not in self.min_solver_args:
                self.minimizer_kwargs.pop(key, None)
                if key is 'ftol':
                    if meth in f_tol_excluded_list:
                        self.minimizer_kwargs['options']['fatol'] = \
                            self.minimizer_kwargs['options']['ftol']
                        self.minimizer_kwargs['options'].pop(key, None)
                else:
                    self.minimizer_kwargs['options'].pop(key, None)
                    self.minimizer_kwargs['options'].pop(key, None)

        for key in kwarg_opt_dict:
            if key not in self.min_solver_args:
                self.minimizer_kwargs.pop(key, None)
                if key is 'ftol':
                    if meth in f_tol_excluded_list:
                        if meth in ['nelder-mead']:
                            self.minimizer_kwargs['options']['fatol'] = \
                                self.minimizer_kwargs['options']['ftol']
                        self.minimizer_kwargs['options'].pop(key, None)
                else:
                    self.minimizer_kwargs['options'].pop(key, None)

        # Initialize return object
        self.res = scipy.optimize.OptimizeResult()
        self.res.nfev = 0
        self.res.nlfev = 0  # Local function evals for all minimisers
        self.res.nljev = 0  # Local jacobian evals for all minimisers

    def sampling(self):
        """
        Generates uniform sampling points in a hypercube and scales the points
        to the bound limits.
        """
        # Generate sampling points.
        #  TODO Assert if func output matches dims. found from bounds
        self.m = len(self.bounds)  # Dimensions

        # Generate uniform sample points in R^m
        self.C = Sobol(self.m).random(self.n)

        # Distribute over bounds
        # TODO: Find a better way to do this
        for i in range(len(self.bounds)):
            self.C[:, i] = (self.C[:, i] *
                            (self.bounds[i][1] - self.bounds[i][0])
                            + self.bounds[i][0])

        return self.C

    def subspace(self):
        """Find subspace of feasible points from g_func definition"""
        # Subspace of feasible points.
        for g in self.g_func:
            self.C = self.C[[g(x, *self.g_args) >= 0
                             for x in self.C]]
            if self.C.size == 0:
                if self.disp:
                    self.res.message = ('No sampling point found within the '
                                        'feasible set. Increasing sampling '
                                        'size.')

        self.fn = numpy.shape(self.C)[0]
        return

    def topograph(self):
        """
        Returns the topographical matrix with True boolean values indicating
        positive entries and False ref. values indicating negative values.
        """
        self.Y = scipy.spatial.distance.cdist(self.C, self.C, 'euclidean')
        self.Z = numpy.argsort(self.Y, axis=-1)
        # Topographical matrix without signs:
        self.A = numpy.delete(self.Z, 0, axis=-1)
        # Obj. function returns to be used as reference table.:
        if self.n_sampled > 0:  # Store old function evaluations
            Ftemp = self.F

        self.F = numpy.zeros(numpy.shape(self.C)[0])
        # New function evaluations
        for i in range(self.n_sampled, numpy.shape(self.C)[0]):
            self.F[i] = self.func(self.C[i,:], *self.args)

        if self.n_sampled > 0:  # Restore saved function evaluations
            self.F[0:self.n_sampled] = Ftemp

        self.n_sampled = numpy.shape(self.C)[0]# self.n
        # Create float value and bool topograph:
        # This replaces all index values in A with the function result:
        self.H = self.F[self.A]

        # Replace numpy inf, -inf and nan objects with floating point numbers
        # fixme: Find a better way to deal with numpy.nan values.
        # nan --> float
        self.H[numpy.isnan(self.H)] = numpy.inf
        self.F[numpy.isnan(self.F)] = numpy.inf
        # inf, -inf  --> floats
        self.H = numpy.nan_to_num(self.H)
        self.F = numpy.nan_to_num(self.F)

        # Topograph with Boolean entries:
        self.T = (self.H.T > self.F.T).T
        return self.T, self.H, self.F

    def k_t_matrix(self, T, k):
        """Returns the k-t topograph matrix"""
        return T[:, 0:k]

    def minimizers(self, K):
        """Returns the minimizer indexes of a k-t matrix"""
        Minimizers = numpy.all(K, axis=-1)
        # Find data point indexes of minimizers:
        return numpy.where(Minimizers)[0]

    def K_optimal(self):
        """
        Returns the optimal k-t topograph with the semi-empirical correlation
        proposed by Henderson et. al. (2015)
        """
        # TODO: Recheck correct implementation, compare with HS19
        K_1 = self.k_t_matrix(self.T, 1)  # 1-t topograph
        if len(self.minimizers(K_1)) == 1:
            if self.disp:
                print("Found k_c = {}".format(1))
            self.K_opt = K_1
            return self.K_opt

        k_1 = len(self.minimizers(K_1))
        k_i = k_1
        i = 2
        while k_1 == k_i:
            K_i = self.k_t_matrix(self.T, i)
            k_i = len(self.minimizers(K_i))
            i += 1

        ep = i * k_i / (k_1 - k_i)
        k_c = numpy.floor((-(ep - 1) + numpy.sqrt((ep - 1.0)**2 + 80.0 * ep))
                          / 2.0)

        if self.disp:
            print("Found k_c = {}".format(k_c))

        self.k_opt = int(k_c + 1)
        if self.k_opt > numpy.shape(self.T)[1]:
            # If size of k_opt exceeds t-graph size.
            self.k_opt = int(numpy.shape(self.T)[1])

        self.K_opt = self.k_t_matrix(self.T, self.k_opt)
        return self.K_opt

    def process_pool(self, ind):
        """
        This function is used to calculate the mimima of each starting point
        in the multiprocessing pool.

        Parameters
        ----------
        ind : int
            Index of current sampling point to access.

        Returns
        -------
        lres : OptimizeResult
            The local optimization result represented as a `OptimizeResult`
            object.
        """
        if self.callback is not None:
            print('Callback for multiprocess '
                  'minimizer starting at {}:'.format(self.C[ind, :], ))

        if self.disp:
            print('Starting multiprocess local '
                  'minimization at {}...'.format(self.C[ind, :]))

        lres = scipy.optimize.minimize(self.func, self.C[ind, :],
                                       **self.minimizer_kwargs)

        # Local function evals for all minimisers
        self.res.nlfev += lres.nfev
        return lres

    def l_minima(self):
        """
        Find the local minima using the chosen local minimisation method with
        the minimisers as starting points.
        """
        # Sort to start with lowest minimizer
        Min_ind = self.minimizers(self.K_opt)
        if len(Min_ind) == 0:  # No min index found
            while len(Min_ind) == 0:
                self.k_opt -= 1
                self.K_opt = self.k_t_matrix(self.T, self.k_opt)
                Min_ind = self.minimizers(self.K_opt)

        Min_fun = self.F[Min_ind]
        fun_min_ind = numpy.argsort(Min_fun)
        Min_ind = Min_ind[fun_min_ind]

        # Init storages
        self.x_vals = []
        self.Func_min = numpy.zeros_like(Min_ind, dtype=float)

        if self.maxfev is not None:  # Update number of sampling points
            self.maxfev -= self.n

        # Pool processes if multiprocessing
        if self.multiproc:
            p = Pool()
            lres_list = p.map(self.process_pool, Min_ind)


        for i, ind in zip(range(len(Min_ind)), Min_ind):
            if not self.multiproc:
                if self.callback is not None:
                    print('Callback for '
                          'minimizer starting at {}:'.format(self.C[ind, :], ))

                if self.disp:
                    print('Starting local '
                          'minimization at {}...'.format(self.C[ind, :]))

                # Find minimum x vals
                lres = scipy.optimize.minimize(self.func, self.C[ind, :],
                                               **self.minimizer_kwargs)

            elif self.multiproc:
                lres = lres_list[i]

            self.x_vals.append(lres.x)
            self.Func_min[i] = lres.fun

            # Local function evals for all minimisers
            self.res.nlfev += lres.nfev

            if self.maxfev is not None:
                self.maxfev -= lres.nfev
                self.minimizer_kwargs['options']['maxfev'] = self.maxfev
                if self.maxfev <= 0:
                    self.res.message = 'Maximum number of function' \
                                       ' evaluations exceeded'
                    self.res.success = False
                    self.break_routine = True

                    if self.disp:
                        print('Maximum number of function evaluations exceeded'
                              'breaking'
                              'minimizations at {}...'.format(self.C[ind, :]))

                        if not self.multiproc:
                            for j in range(i + 1, len(Min_ind)):
                                self.x_vals.append(self.C[Min_ind[j], :])
                                self.Func_min[j] = self.F[Min_ind[j]]

                    if not self.multiproc:
                        break

        self.x_vals = numpy.array(self.x_vals)
        # Sort and save
        ind_sorted = numpy.argsort(self.Func_min)  # Sorted indexes in Func_min

        # Save ordered list of minima
        self.res.xl = self.x_vals[ind_sorted]  # Ordered x vals
        self.res.funl = self.Func_min[ind_sorted]  # Ordered fun values

        # Find global of all minimisers
        if numpy.size(ind_sorted) == 0: #TODO: Check
            self.res.message = ("Local solver failed to converge.")
            self.break_routine = True
            self.res.success = False
            self.res.x = self.x_vals
            self.res.fun = self.Func_min
            x_global_min = self.x_vals[0]
            return x_global_min

        self.res.x = self.x_vals[ind_sorted[0]]  # Save global minima
        x_global_min = self.x_vals[ind_sorted[0]][0]
        self.res.fun = self.Func_min[ind_sorted[0]]  # Save global fun value
        return x_global_min
