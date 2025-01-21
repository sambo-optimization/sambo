import inspect
import os
import unittest
from contextlib import chdir
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import get_type_hints
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import (
    Bounds, NonlinearConstraint, rosen, minimize as scipy_minimize,
    shgo as scipy_shgo,
)
from scipy.stats import gaussian_kde

from sambo import minimize, Optimizer, SamboSearchCV
from sambo._space import Space
from sambo._sceua import sceua
from sambo._shgo import shgo
from sambo._smbo import smbo
from sambo.plot import plot_convergence, plot_evaluations, plot_objective, plot_regret
from sambo._util import OptimizeResult, recompute_kde, weighted_uniform_sampling

minimize = partial(minimize, rng=0)
sceua = partial(sceua, rng=0)
smbo = partial(smbo, rng=0)

ROSEN_TEST_PARAMS = dict(fun=rosen, bounds=[(-2, 3.)] * 3)
ROSEN_TEST_PARAMS_SPACE = dict(fun=rosen, bounds=[(-2, 3.),
                                                  (-2, 4),
                                                  list(range(-2, 4))])
ROSEN_TRUE_MIN = 0

BUILTIN_METHODS = ['shgo', 'sceua', 'smbo']
BUILTIN_ESTIMATORS = ['gp', 'et', 'gb']

Optimizer.POINTS_PER_DIM = 1_000
Optimizer.MAX_POINTS_PER_ITER = 10_000


def check_result(res, y_true, atol=1e-5):
    np.testing.assert_allclose(res.fun, y_true, atol=atol, err_msg=res)


check_rosen_result = partial(check_result, y_true=ROSEN_TRUE_MIN)


class TestSCEUA(unittest.TestCase):
    def test_rosen10d(self):
        res = sceua(**ROSEN_TEST_PARAMS)
        check_rosen_result(res, atol=3)


class TestSHGO(unittest.TestCase):
    def test_rosen10d(self):
        params = ROSEN_TEST_PARAMS.copy()
        res = shgo(func=params.pop('fun'), **params)
        check_rosen_result(res)


class TestSMBO(unittest.TestCase):
    def test_rosen(self):
        res = smbo(**ROSEN_TEST_PARAMS, max_iter=50, estimator='gp', n_models=3, rng=0)
        check_rosen_result(res, atol=8)

    def test_ask_tell_interface(self):
        optimizer = Optimizer(fun=None, x0=[0], n_init=0)
        optimizer.tell(0, [0])
        optimizer.tell(1, [1])
        x = optimizer.ask(1)
        optimizer.tell(2)
        optimizer.tell(3, x)
        res = optimizer.run(max_iter=0)
        np.testing.assert_array_equal(res.nfev, 4)
        np.testing.assert_array_equal(res.funv, np.arange(4))

    def test_multiple_runs_continue_ie_warm_start(self):
        optimizer = Optimizer(fun=sum, x0=[0]*3, n_init=1)
        optimizer.run(max_iter=1)
        optimizer.run(max_iter=1)
        res = optimizer.run(max_iter=1)
        np.testing.assert_array_equal(res.nfev, 4)


class TestPlot(unittest.TestCase):
    if os.environ.get('CI') == 'true':
        import matplotlib
        matplotlib.use('Agg')

    @classmethod
    def setUpClass(cls):
        cls.RESULT_SMBO = minimize(**ROSEN_TEST_PARAMS_SPACE, max_iter=50, method='smbo', estimator='gp')
        cls.RESULT_SCEUA = minimize(**ROSEN_TEST_PARAMS_SPACE, max_iter=50, method='sceua')
        cls.RESULT_SHGO = minimize(**ROSEN_TEST_PARAMS_SPACE, max_iter=50, method='shgo')

    def tearDown(self):
        plt.show()

    def test_plot_convergence(self):
        plot_convergence(
            self.RESULT_SMBO,  # no name
            ('SMBO', self.RESULT_SMBO),
            ('SCE', self.RESULT_SCEUA),
            ('SHGO', self.RESULT_SHGO),
            true_minimum=ROSEN_TRUE_MIN, yscale='log')

    def test_plot_regret(self):
        plot_regret(
            self.RESULT_SMBO,  # no name
            ('SMBO', self.RESULT_SMBO),
            ('SCE', self.RESULT_SCEUA),
            ('SHGO', self.RESULT_SHGO),
            true_minimum=ROSEN_TRUE_MIN)

    def test_plot_objective(self):
        _plot_objective = partial(plot_objective, resolution=4)
        _plot_objective(self.RESULT_SMBO)
        _plot_objective(self.RESULT_SMBO, plot_max_points=5)
        with self.assertWarns(UserWarning):
            _plot_objective(self.RESULT_SCEUA)

    def test_plot_evaluations(self):
        plot_evaluations(self.RESULT_SMBO)
        plot_evaluations(self.RESULT_SCEUA)
        plot_evaluations(self.RESULT_SHGO)


class TestSpace(unittest.TestCase):
    @staticmethod
    def fun(x):
        return np.sum(np.abs(x.astype(float)))

    BOUNDS = [
        (-2., 3.),
        (-2, 4),
        np.arange(-2, 4).astype(str),
    ]
    SPACE = Space(BOUNDS)

    @staticmethod
    def CONSTRAINTS(x):
        assert isinstance(x[0], float), x
        assert isinstance(x[1], int), x
        assert isinstance(x[2], str), x
        return True

    def test_space_dtypes_passed_down_to_fun_constraints_callback(self):

        def fun(x):
            nonlocal self
            self.CONSTRAINTS(x)
            return self.fun(x)

        def callback(res):
            nonlocal self
            assert self.CONSTRAINTS(res.x)
            assert self.CONSTRAINTS(res.xv[0])
            return True

        for method in BUILTIN_METHODS:
            with self.subTest(method=method):
                minimize(fun, constraints=self.CONSTRAINTS, bounds=self.BOUNDS,
                         callback=callback, method=method, max_iter=20)

    def test_minimize_result_space_dtypes(self):
        for method in BUILTIN_METHODS:
            with self.subTest(method=method):
                res = minimize(self.fun, bounds=self.BOUNDS, method=method, max_iter=40, rng=0)
                self.CONSTRAINTS(res.x)
                self.CONSTRAINTS(res.xv[0])

    def _check_points(self, X):
        for i, b in enumerate(self.SPACE._bounds):
            valid_values = (X[:, i] >= b[0]) & (X[:, i] <= b[1])
            self.assertTrue(np.all(valid_values), i)
        # All values including edges are present in a large-enough sample
        for num in range(*self.BOUNDS[1]):
            self.assertTrue(np.isin(num, X[:, 1]), (1, num))
        for num in range(len(self.BOUNDS[2])):
            self.assertTrue(np.isin(num, X[:, 2]), (2, num))

    def test_sample_init(self):
        X = self.SPACE.sample(1000, init=True)
        self._check_points(X)

    def test_sample(self):
        X = self.SPACE.sample(1000)
        self._check_points(X)


class TestMinimize(unittest.TestCase):
    def test_scipy_minimize_api(self):
        res = scipy_shgo(
            func=ROSEN_TEST_PARAMS['fun'],
            bounds=ROSEN_TEST_PARAMS['bounds'],
            args=(),
            constraints=NonlinearConstraint(lambda x: 0, 0, 1),
            callback=lambda x: None,
        )
        check_result(res, 0)

        _ = scipy_minimize(
            fun=ROSEN_TEST_PARAMS['fun'],
            x0=np.mean(ROSEN_TEST_PARAMS['bounds'], axis=1),
            bounds=ROSEN_TEST_PARAMS['bounds'],
            args=(),
            constraints=NonlinearConstraint(lambda x: 0, 0, 1),
            tol=.01,
            callback=lambda x: None,
        )

    def test_sceua(self):
        res = minimize(**ROSEN_TEST_PARAMS, method='sceua', n_complexes=20, complex_size=3)
        check_rosen_result(res, atol=1)

    def test_smbo(self):
        res = minimize(**ROSEN_TEST_PARAMS, method='smbo', max_iter=20, estimator='gp')
        check_result(res, 0, atol=11)

    def test_args(self):
        def f(x, a):
            assert a is True, locals()
            return 0

        for method in BUILTIN_METHODS:
            with self.subTest(method=method):
                _ = minimize(f, args=(True,), x0=[0], method=method, max_iter=10)

    def test_max_iter(self):
        def f(x):
            nonlocal counter
            counter += 1
            return 0

        MAX_ITER = 11
        _minimize = partial(minimize, x0=[0], max_iter=MAX_ITER)

        counter = 0
        _ = _minimize(f, method='sceua', n_complexes=2)
        self.assertEqual(counter, MAX_ITER)

        counter = 0
        _ = _minimize(f, method='smbo')
        self.assertLessEqual(counter, MAX_ITER)

    def test_constraints(self):
        def f(x):
            assert np.all(x > 0), x
            return 0

        def constraints(x):
            nonlocal was_called
            was_called = True
            return np.all(x > 0)

        _minimize = partial(minimize, x0=[0] * 2, constraints=constraints, max_iter=5)
        for method in BUILTIN_METHODS:
            with self.subTest(method=method):
                was_called = False
                _ = _minimize(f, method=method)
                self.assertTrue(was_called)

    def test_callback(self):
        def callback(res):
            assert isinstance(res, OptimizeResult)
            nonlocal was_called
            was_called = True
            return True

        _minimize = partial(minimize, fun=lambda x: 0, x0=[0], callback=callback, max_iter=10)

        for method in BUILTIN_METHODS:
            with self.subTest(method=method):
                was_called = False
                res = _minimize(method=method)
                self.assertEqual(res.nfev, 1)
                self.assertTrue(was_called)

    def test_bounds(self):
        bounds = Bounds([0.], [1.])
        _minimize = partial(minimize, fun=lambda x: x[0], bounds=bounds, max_iter=10)

        for method in BUILTIN_METHODS:
            with self.subTest(method=method):
                res = _minimize(method=method)
                check_result(res, 0, atol=.001)

    def test_y0(self):
        x0 = np.arange(3)
        for method in BUILTIN_METHODS:
            with self.subTest(method=method):
                minimize(
                    fun=lambda x: x[0],
                    x0=x0.reshape((len(x0), 1)),
                    y0=x0, method=method, max_iter=6)

    def test_our_params_match_scipy_optimize_params(self):
        scipy_sig = inspect.signature(scipy_minimize)
        sig = inspect.signature(minimize)
        OUR_PARAMS = {
            'max_iter',
            'n_iter_no_change',
            'y0',
            'n_jobs',
            'disp',
            'rng',
            'kwargs',
        }
        for param in sig.parameters:
            if param in OUR_PARAMS:
                continue
            self.assertIn(param, scipy_sig.parameters)


class TestSklearnEstimators(unittest.TestCase):
    def test_estimator_factory(self):
        DEFAULT_KWARGS = {'max_iter': 20, 'n_iter_no_change': 5, 'rng': 0}
        ESTIMATOR_KWARGS = {
            'gp': {},
            'et': {'max_iter': 40, 'n_iter_no_change': 10},
            'gb': {},
        }
        for estimator in BUILTIN_ESTIMATORS:
            with self.subTest(estimator=estimator):
                res = smbo(lambda x: sum((x-2)**2), bounds=[(-100, 100)], estimator=estimator,
                           **dict(DEFAULT_KWARGS, **ESTIMATOR_KWARGS[estimator]))
                self.assertLess(res.fun, 1, msg=res)

    def test_SamboSearchCV_large_param_grid(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.tree import DecisionTreeClassifier

        X, y = load_breast_cancer(return_X_y=True)
        clf = DecisionTreeClassifier(random_state=0)
        param_grid = {
            'max_depth': list(range(1, 30)),
            'min_samples_split': [2, 5, 10, 20, 50, 100],
            'min_samples_leaf': list(range(1, 20)),
            'criterion': ['gini', 'entropy'],
            'max_features': [None, 'sqrt', 'log2'],
        }
        # altogether 9 * 6 * 4 * 2 * 3 >= 1000 iterations in grid space, yet we ...
        max_iter = 40
        for method in BUILTIN_METHODS:
            with self.subTest(method=method):
                search = SamboSearchCV(
                    clf, param_grid, max_iter=max_iter,
                    method=method, cv=2, verbose=0, rng=0)
                search.fit(X, y)
                self.assertGreater(search.best_score_, .9)
                self.assertIsNotNone(search.best_params_)
                print(search.opt_result_)


class TestDocs(unittest.TestCase):
    def test_make_doc_plots(self):
        KWARGS = {
            'shgo': dict(n_init=30),
            'smbo': dict(n_init=30),
            'sceua': dict(n_complexes=3),
        }
        results = [
            minimize(
                rosen, bounds=[(-2., 2.)]*2,
                constraints=lambda x: sum(x**2) <= 2**len(x),
                max_iter=100, method=method, rng=2,
                **KWARGS.get(method, {}),
            )
            for method in BUILTIN_METHODS
        ]
        for res in results:
            self.assertAlmostEqual(res.fun, 0, places=0, msg=res)

        PLOT_FUNCS = (
            plot_regret,
            plot_convergence,
            plot_objective,
            plot_evaluations,
        )
        KWARGS = {
            plot_regret: dict(true_minimum=0),
            plot_convergence: dict(yscale='log', true_minimum=0),
            plot_objective: dict(resolution=12)
        }
        www_dir = Path(__file__).parent.parent / 'www'
        www_dir.mkdir(exist_ok=True)
        with chdir(www_dir):
            for plot_func in PLOT_FUNCS:
                name = plot_func.__name__.removeprefix("plot_")
                with self.subTest(plot=name):
                    try:
                        fig = plot_func(*zip((f'method={m!r}' for m in BUILTIN_METHODS), results),
                                        **KWARGS.get(plot_func, {}))
                    except TypeError:
                        fig = plot_func(results[0], **KWARGS.get(plot_func, {}))  # FIXME: plot (1, 3) subplots
                    fig.savefig(f'{name}.svg')
                    plt.show()

    def test_website_example1(self):
        res = minimize(
            rosen, bounds=[(-2., 2.), ] * 2,
            constraints=lambda x: sum(x**2) <= 2**len(x),
            n_init=7, method='shgo', rng=0,
        )
        print(type(res), res, sep='\n\n')
        self.assertAlmostEqual(res.fun, 0, places=0, msg=res)

    @patch.object(Optimizer, 'POINTS_PER_DIM', 20_000)
    @patch.object(Optimizer, 'MAX_POINTS_PER_ITER', 80_000)
    def test_website_example2(self):

        def evaluate(x):
            ...  # Some long and laborious process, e.g.
            return rosen(x)

        results = []
        for estimator in BUILTIN_ESTIMATORS:
            optimizer = Optimizer(fun=None, bounds=[(-2, 2)]*4, estimator=estimator, rng=0)

            for i in range(30):
                suggested_x = optimizer.ask(n_candidates=1)
                y = [evaluate(x) for x in suggested_x]
                optimizer.tell(y)

            result = optimizer.run()
            results.append(result)
        named_results = [(f"method='smbo', estimator={e!r}", r) for e, r in zip(BUILTIN_ESTIMATORS, results)]
        fig = plot_convergence(*named_results, true_minimum=0)
        www_dir = Path(__file__).parent.parent / 'www'
        www_dir.mkdir(exist_ok=True)
        with chdir(www_dir):
            fig.savefig('convergence2.svg')
        plt.show()

    def test_website_example3(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV

        X, y = load_breast_cancer(return_X_y=True)
        clf = DecisionTreeClassifier(random_state=0)
        param_grid = {
            'max_depth': list(range(1, 20)),
            'min_samples_split': [2, 5, 10, 50, 100],
            'min_samples_leaf': list(range(1, 10)),
            'criterion': ['gini', 'entropy'],
            'max_features': [None, 'sqrt', 'log2'],
        }
        search = GridSearchCV(clf, param_grid, cv=2, n_jobs=1)
        # Trying all ~6k combinations may take a long time ...
        search.fit(X, y)
        pprint(dict(sorted(search.best_params_.items())))
        print(search.best_score_)

        # Alternatively ...
        from sambo import SamboSearchCV

        search = SamboSearchCV(clf, param_grid, max_iter=100, cv=2, n_jobs=1, rng=0)
        search.fit(X, y)  # Fast, good enough
        pprint(dict(sorted(search.best_params_.items())))
        print(search.best_score_)
        print(search.opt_result_)

    def test_annotations(self):
        from sambo import minimize
        from sambo._shgo import shgo
        from sambo._sceua import sceua
        from sambo._smbo import smbo, Optimizer

        SKIP_ARGS = {
            'all': ('bounds', 'n_iter_no_change'),
            Optimizer.__init__: ('fun',),
        }
        annot_ref = get_type_hints(minimize)
        for fun in (
                shgo,
                sceua,
                smbo,
                Optimizer.__init__,
        ):
            annot = get_type_hints(fun)
            for arg in annot:
                if arg in SKIP_ARGS['all'] or arg in SKIP_ARGS.get(fun, ()):
                    continue
                if arg in annot_ref:
                    self.assertEqual(
                        annot[arg], annot_ref[arg], msg=f'{fun.__qualname__} / {arg}')


class TestUtil(unittest.TestCase):
    def test_weighted_uniform_sampling(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(-10, 10, (100, 2))
        y = rng.uniform(1, 10, 100)
        bounds = [(-10, 10), (-10, 10)]
        n_samples = 10000
        kde = recompute_kde(X, y)
        sampled_points = weighted_uniform_sampling(kde, bounds, n_samples, None, 0)

        # Verify results
        hist, xedges, yedges = np.histogram2d(*sampled_points.T, range=bounds)
        # Compare histogram density with weight distribution
        kde = gaussian_kde(X.T, weights=(np.max(y) - y) / np.sum(np.max(y) - y))
        test_grid = np.array(np.meshgrid(xedges[:-1], yedges[:-1])).T.reshape(-1, 2)
        pdf_values = kde(test_grid.T).reshape(hist.shape)
        # Normalize for direct comparison
        hist_normalized = hist / np.sum(hist)
        pdf_normalized = pdf_values / np.sum(pdf_values)
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        extent = np.array(bounds).flatten()
        axes[0].imshow(hist_normalized, extent=extent, origin='lower', cmap='Blues')
        axes[1].imshow(pdf_normalized, extent=extent, origin='lower', cmap='Reds')
        plt.show()

        diff = np.abs(hist_normalized - pdf_normalized).mean()
        self.assertLess(diff, .05)


if __name__ == '__main__':
    unittest.main()
