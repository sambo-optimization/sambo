"""
# SAMBO - **Sequential and Model-Based Optimization** [in Python]

Sambo is a **global optimization framework for finding approximate global optima**\N{DAGGER}
of arbitrary high-dimensional objective functions in the **least number of function evaluations**.
Function evaluations are considered the "expensive" resource
(it can sometimes take weeks to obtain results!),
so it's important to find good-enough solutions in
**as few steps as possible** (whence _sequential_).

The main tools in this Python optimization toolbox are:

* **function `sambo.minimize()`**, a near drop-in replacement for [`scipy.optimize.minimize()`][sp_opt_min],
* **class `Optimizer`** with an ask-and-tell user interface,
  supporting arbitrary scikit-learn-like surrogate models,
  with Bayesian optimization estimators like [gaussian process] and [extra trees],
  built in,
* **`SamboSearchCV`**, a much faster drop-in replacement for
  scikit-learn's [`GridSearchCV`][skl_gridsearchcv] and similar exhaustive
  machine-learning hyper-parameter tuning methods,
  but compared to unpredictable stochastic methods, _informed_.

The algorithms and methods implemented by or used in this package are:

* [simplical homology global optimization] (SHGO), customizing the [implementation from SciPy],
* surrogate machine learning model-based optimization,
* [shuffled complex evolution] (SCE-UA with improvements).

[simplical homology global optimization]: http://doi.org/10.1007/s10898-018-0645-y
[implementation from SciPy]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html
[shuffled complex evolution]: https://doi.org/10.1007/BF00939380

This open-source project was heavily **inspired by _scikit-optimize_** project,
which now seems helplessly defunct.

The project is one of the better optimizers around according to [benchmark].

\N{DAGGER} The contained algorithms seek to _minimize_ your objective `f(x)`.
If you instead need the _maximum_, simply minimize `-f(x)`. 💡

[gaussian process]: https://www.gaussianprocess.org/gpml/chapters/RW.pdf
[extra trees]: https://doi.org/10.1007/s10994-006-6226-1
[kernel ridge regression]: https://scikit-learn.org/stable/modules/kernel_ridge.html
[sp_opt_min]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
[skl_gridsearchcv]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
"""
from ._minimize import minimize
from ._smbo import Optimizer
from ._estimators import SamboSearchCV
from ._util import OptimizeResult
try:
    from ._version import __version__
except ImportError:
    __version__ = '0.0.0-dev'

__all__ = [
    'minimize',
    'Optimizer',
    'OptimizeResult',
    'SamboSearchCV',
]
