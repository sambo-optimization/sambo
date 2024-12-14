from numpy import arange, arctan2, array, cos, exp, pi, prod, sin, sqrt, sum
from scipy.optimize import rosen

# Codomains computed with:
# for t in test_functions:
#     n = 200 if len(t['bounds']) <= 3 else 4
#     x = linspace(*array(t['bounds']).T, n)
#     mesh = vstack([a.ravel() for a in meshgrid(*x.T, indexing="ij")]).T
#     y = array([t['func'](x) for x in mesh if (t['constraints'] or (lambda x: True))(x)])
#     print(t['name'], [float(y.min()), float(y.max())])
# TODO: Replace with https://github.com/nathanrooy/landscapes ?
TEST_FUNCTIONS = [
    # From https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    {
        'name': 'rosenbrock',
        'func': rosen,
        'constraints': lambda x: (x[0] - 1)**3 - x[1] + 1 <= 0 and x[0] + x[1] - 2 <= 0,
        'bounds': [[-2., 2.1], [-1.9, 2.]],
        'tol': .2,
        'solution': 0,
        'codomain': [0, 3500],
    },
    {
        'name': 'rosenbrock',
        'func': rosen,
        'constraints': lambda x: sum(x**2) <= 1.1 * len(x),
        'bounds': [[-2., 2.1]]*10,
        'tol': 1,
        'solution': 0,
        'codomain': [0, 3500],
    },
    {
        'name': 'bird',
        'func': lambda x: sin(x[1]) * exp((1-cos(x[0]))**2) + cos(x[0]) * exp((1-sin(x[1]))**2) + (x[0] - x[1])**2,
        'constraints': lambda x: (x[0] + 5)++2 + (x[1] + 5)**2 < 25,
        'bounds': [[-10., 0.], [-6.5, 0.]],
        'tol': 1,
        'solution': -106.77,
        'codomain': [-106.77, 97.7],
    },
    {
        'name': 'gomez-levy',
        'func': lambda x: 4*x[0]**2 - 2.1*x[0]**4 + 1/3*x[0]**6 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4,
        'constraints': lambda x: -sin(4*pi*x[0]) + 2*sin(2*pi*x[1])**2 <= 1.5,
        'bounds': [[-1., .75], [-1., 1.]],
        'tol': .1,
        'solution': -1.0317,
        'codomain': [-1.0317, 3.233],
    },
    {
        'name': 'simionescu',
        'func': lambda x: .1 * x[0] * x[1],
        'constraints': lambda x: x[0]**2 + x[1]**2 <= (1 + .2 * cos(8 * arctan2(x[0], x[1])))**2,
        'bounds': [[-1.25, 1.25], [-1.25, 1.25]],
        'tol': .03,
        'solution': -0.072,
        'codomain': [-0.072, 0.072],
    },
    {
        'name': 'rastrigin',
        'func': lambda x: 10 * len(x) + sum(x**2 - 10 * cos(2 * pi * x)),
        'constraints': lambda x: x[0]**2 + x[1]**2 <= (1 + .2 * cos(8 * arctan2(x[0], x[1])))**2,
        'bounds': [[-5., 5.12], [-5.12, 5.]],
        'tol': .1,
        'solution': 0,
        'codomain': [0, 40],
    },
    # Various sources
    {
        'name': 'eggholder',
        'func': lambda x: (-(x[1] + 47) * sin(sqrt(abs(x[0] / 2 + (x[1] + 47)))) -
                           x[0] * sin(sqrt(abs(x[0] - (x[1] + 47))))),
        'constraints': None,
        'bounds': [[-512., 512.], [-512., 512.]],
        'tol': 300,
        'solution': -960,
        'codomain': [-960, 1050],
    },
    {
        'name': '6-hump-camelback',
        'func': lambda x: (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2,
        'constraints': None,
        'bounds': [[-2., 2.], [-1., 1.]],
        'tol': 1,
        'solution': -1.0317,
        'codomain': [-1.0317, 891.9],
    },
    {
        'name': 'griewank',
        'func': lambda x: 1 + sum(x**2) / 4000 - prod(cos(x/sqrt(1 + arange(len(x))))),
        'constraints': None,
        'bounds': [[-600., 590.], [-590., 600.]],
        'tol': 300,
        'solution': 0,
        'codomain': [0, 181],
    },
    {
        'name': 'hartman',
        'func': (
            lambda x,
                   _alpha=array([1, 1.2, 3, 3.2]),  # noqa: E131
                   _A=array([[10, 3, 17, 3.5, 1.7, 8],  # noqa: E131
                             [.05, 10, 17, .1, 8, 14],
                             [3, 3.5, 1.7, 10, 17, 8],
                             [17, 8, .05, 10, .1, 14.0]]),
                   _P=10**-4 * array([[1312, 1696, 5569, 124, 8283, 5886],  # noqa: E131
                                      [2329, 4135, 8307, 3736, 1004, 9991],
                                      [2348, 1451, 3522, 2883, 3047, 6650],
                                      [4047, 8828, 8732, 5743, 1091, 381]]):
            -sum(_alpha * exp(-sum(_A * (x - _P)**2, axis=1)))),
        'constraints': None,
        'bounds': [[0., 1.]] * 6,
        'tol': .3,
        'solution': -3.3224,
        'codomain': [-3.3224, 0],
    },
    {
        'name': 'schwefel',
        'func': lambda x: 418.9828872724338 * len(x) - sum(x * sin(sqrt(abs(x)))),
        'constraints': None,
        'bounds': [[-490., 500.], [-500., 480.]],
        'tol': 1,
        'solution': 0,
        'codomain': [0, 1676],
    },
    {
        'name': 'branin-hoo',
        'func': (
            lambda x, _a=1, _b=5.1 / (4 * pi**2), _c=5 / pi, _r=6, _s=10, _t=1 / (8 * pi):
            (_a * (x[1] - _b * x[0]**2 + _c * x[0] - _r)**2 + _s * (1 - _t) * cos(x[0]) + _s)),
        'constraints': None,
        'bounds': [[-5., 10.], [0., 15.]],
        'tol': .3,
        'solution': 0.3978,
        'codomain': [0.3978, 308],
    },
]


assert all(all(isinstance(v, float) for v in b)
           for f in TEST_FUNCTIONS
           for b in f['bounds'])
