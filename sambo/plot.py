"""
The module contains **functions for plotting**
convergence, regret, partial dependence, sequence of evaluations ...

Example
-------
>>> import matplotlib.pyplot as plt
>>> from scipy.optimize import rosen
>>> result = minimize(rosen, bounds=[(-2, 2), (-2, 2)],
...                   constraints=lambda x: sum(x) <= len(x))
>>> plot_convergence(result)
>>> plot_regret(result)
>>> plot_objective(result)
>>> plot_evaluations(result)
>>> plt.show()
"""
import warnings
from functools import partial
from itertools import cycle
from numbers import Integral, Real
from typing import Iterable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import (
    FixedLocator, FormatStrFormatter, FuncFormatter, LogLocator, MaxNLocator,
    ScalarFormatter,
)

from ._space import Space
from ._util import OptimizeResult, _SklearnLikeRegressor

_MARKER_SEQUENCE = 'osxdvP^'
_DEFAULT_SUBPLOT_SIZE = 1.7


def plot_convergence(
        *results: OptimizeResult | tuple[str, OptimizeResult],
        true_minimum: Optional[float] = None,
        xscale: Literal['linear', 'log'] = 'linear',
        yscale: Literal['linear', 'log'] = 'linear',
) -> Figure:
    """
    Plot one or several convergence traces,
    showing how an error estimate evolved during the optimization process.

    Parameters
    ----------
    *results : OptimizeResult or tuple[str, OptimizeResult]
        The result(s) for which to plot the convergence trace.
        In tuple format, the string is used as the legend label
        for that result.

    true_minimum : float, optional
        The true minimum *value* of the objective function, if known.

    xscale, yscale : {'linear', 'log'}, optional, default='linear'
        The scales for the axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure.

    Example
    -------
    .. image:: /convergence.svg
    """
    assert results, results

    fig = plt.figure()
    _watermark(fig)
    ax = plt.gca()
    ax.set_title("Convergence")
    ax.set_xlabel("Number of function evaluations $n$")
    ax.set_ylabel(r"$\min\ f(x)$ after $n$ evaluations")
    ax.grid()
    _set_xscale_yscale(ax, xscale, yscale)
    fig.set_layout_engine('tight')

    MARKER = cycle(_MARKER_SEQUENCE)

    for i, result in enumerate(results, 1):
        name = f'#{i}' if len(results) > 1 else None
        if isinstance(result, tuple):
            name, result = result
        result = _check_result(result)

        nfev = _check_nfev(result)
        mins = np.minimum.accumulate(result.funv)

        ax.plot(range(1, nfev + 1), mins,
                label=name, marker=next(MARKER), markevery=(.05 + .05*i, .2),
                linestyle='--', alpha=.7, markersize=6, lw=2)

    if true_minimum is not None:
        ax.axhline(true_minimum, color="k", linestyle='--', lw=1, label="True minimum")

    if true_minimum is not None or name is not None:
        ax.legend(loc="upper right")

    return fig


def _set_xscale_yscale(ax, xscale, yscale):
    kw = {}
    if xscale in ('log', 'symlog'):
        xscale = 'symlog'
        kw = {'linthresh': 1}
    ax.set_xscale(xscale, **kw)
    kw = {}
    if yscale in ('log', 'symlog'):
        yscale = 'symlog'
        kw = {'linthresh': 1}
    ax.set_yscale(yscale, **kw)


def _check_nfev(result):
    nfev = max(result.nfev, len(result.xv))
    if nfev != len(result.funv):
        warnings.warn(
            'OptimizeResult.nfev != len(OptimizeResult.xv); '
            'plotted data might be incomplete.', stacklevel=3)
    return nfev


def plot_regret(
        *results: OptimizeResult | tuple[str, OptimizeResult],
        true_minimum: Optional[float] = None,
        xscale: Literal['linear', 'log'] = 'linear',
        yscale: Literal['linear', 'log'] = 'linear',
) -> Figure:
    """
    Plot one or several cumulative [regret] traces.
    Regret is the difference between achieved objective and its optimum.

    [regret]: https://en.wikipedia.org/wiki/Regret_(decision_theory)

    Parameters
    ----------
    *results : OptimizeResult or tuple[str, OptimizeResult]
        The result(s) for which to plot the convergence trace.
        In tuple format, the string is used as the legend label
        for that result.

    true_minimum : float, optional
        The true minimum *value* of the objective function, if known.
        If unspecified, minimum is assumed to be the minimum of the
        values found in `results`.

    xscale, yscale : {'linear', 'log'}, optional, default='linear'
        The scales for the axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure.

    Example
    -------
    .. image:: /regret.svg
    """
    assert results, results

    fig = plt.figure()
    _watermark(fig)
    ax = fig.gca()
    ax.set_title("Cumulative regret")
    ax.set_xlabel("Number of function evaluations $n$")
    ax.set_ylabel(r"Cumulative regret after $n$ evaluations: "
                  r"$\ \sum_t^n{\,\left[\,f\,\left(x_t\right) - f_{\mathrm{opt}}\,\right]}$")
    ax.grid()
    _set_xscale_yscale(ax, xscale, yscale)
    ax.yaxis.set_major_formatter(FormatStrFormatter('$%.3g$'))
    fig.set_layout_engine('tight')

    MARKER = cycle(_MARKER_SEQUENCE)

    if true_minimum is None:
        true_minimum = np.min([
            np.min((r[1] if isinstance(r, tuple) else r).funv)  # TODO ensure funv???
            for r in results
        ])

    for i, result in enumerate(results, 1):
        name = f'#{i}' if len(results) > 1 else None
        if isinstance(result, tuple):
            name, result = result
        result = _check_result(result)

        nfev = _check_nfev(result)
        regrets = [np.sum(result.funv[:i] - true_minimum)
                   for i in range(1, nfev + 1)]

        ax.plot(range(1, nfev + 1), regrets,
                label=name, marker=next(MARKER), markevery=(.05 + .05*i, .2),
                linestyle='--', alpha=.7, markersize=6, lw=2)

    if name is not None:
        ax.legend(loc="lower right")

    return fig


def _get_lim(axis: str, bounds, i):
    assert axis in 'xy', axis
    margin = plt.rcParams[f"axes.{axis}margin"]
    span = bounds[i][1] - bounds[i][0]
    return bounds[i] + np.array([-1, 1]) * span * margin


class _CategoricalFormatter(FuncFormatter):
    def __init__(self, func):
        super().__init__(func)
        self.scalar_formatter = ScalarFormatter()

    def set_axis(self, axis):
        super().set_axis(axis)
        self.scalar_formatter.set_axis(axis)

    def set_locs(self, locs):
        super().set_locs(locs)
        # ScalarFormatter needs locs in advance to get the common fmt right
        locs = [self.func(l, i) for i, l in enumerate(locs)]
        if all(isinstance(i, (Integral, Real)) for i in locs):
            self.scalar_formatter.set_locs(locs)

    def __call__(self, *args, **kwargs):
        formatted_value = super().__call__(*args, **kwargs)
        if isinstance(formatted_value, (Integral, Real)):
            formatted_value = self.scalar_formatter(formatted_value, *args[1:], **kwargs)
        return formatted_value


def _format_scatter_plot_axes(fig, axs, space, plot_dims=None, dim_labels=None, size=2):
    assert plot_dims is not None, plot_dims
    bounds = dict(zip(plot_dims, space._bounds[plot_dims]))
    n_dims = len(plot_dims)

    # Work out min, max of y axis for the diagonal so we can adjust
    # them all to the same value
    diagonal_ylim = np.array([axs[i, i].get_ylim() for i in range(n_dims)])
    diagonal_ylim = np.min(diagonal_ylim[:, 0]), np.max(diagonal_ylim[:, 1])

    if dim_labels is None:
        dim_labels = [fr"$\mathbf{{x_{{{i}}}}}$" for i in plot_dims]

    nticks = int((1 + np.log10(size / (base_figsize := _DEFAULT_SUBPLOT_SIZE))) * (base_nticks := 6))  # noqa: F841
    fontsize = 8

    _MaxNLocator = partial(MaxNLocator, nbins=nticks)
    _FixedLocator = partial(FixedLocator, nbins=nticks)

    for _i, i in enumerate(plot_dims):  # rows
        for _j, j in enumerate(plot_dims):  # columns
            ax: plt.Axes = axs[_i, _j]

            # Remove upper triangular
            if j > i:
                ax.remove()
                continue

            # Set xlim, ylim, locator, formatter
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
            ax.set_xlim(*_get_lim('x', bounds, j))
            ax.set_ylim(*_get_lim('y', bounds, i))
            if space._is_cat(j):
                ax.xaxis.set_major_locator(_FixedLocator(np.arange(len(space._encoders[j].index))))
                ax.xaxis.set_major_formatter(_CategoricalFormatter(partial(space._label_categorical, j)))
            else:
                ax.xaxis.set_major_locator(_MaxNLocator(integer=space._is_int(j)))
            if i == j:  # Diagonal plots have their own yaxis
                ax.set_ylim(*diagonal_ylim)
                ax.yaxis.set_major_locator(_MaxNLocator(prune='lower' if i < n_dims - 1 else None))
            elif space._is_cat(i):
                ax.yaxis.set_major_locator(_FixedLocator(np.arange(len(space._encoders[i].index))))
                ax.yaxis.set_major_formatter(_CategoricalFormatter(partial(space._label_categorical, i)))
            else:
                ax.yaxis.set_major_locator(_MaxNLocator(integer=space._is_int(i)))

            # Off-diagonal plots
            if i > j:
                ax.xaxis.set_ticks_position('both')
                if j < i - 1:
                    ax.yaxis.set_ticks_position('both')

                # Leftmost column gets ylabels
                if j == 0:
                    ax.tick_params(axis='y', pad=2)
                    ax.set_ylabel(dim_labels[_i], fontsize=fontsize, fontweight='bold', labelpad=1)
                else:
                    ax.set_yticklabels([])

                # Bottom row gets xlabels
                if i == n_dims - 1:
                    ax.tick_params(axis='x', labelrotation=45, pad=2)
                    ax.set_xlabel(dim_labels[_j], fontsize=fontsize, fontweight='bold', labelpad=1)
                else:
                    ax.set_xticklabels([])

            else:  # diagonal plots
                ax.yaxis.tick_right()
                ax.tick_params(axis='y', labelrotation=45, top=True, bottom=True, right=True)
                for label in ax.get_yticklabels():
                    label.set_rotation_mode('anchor')

                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()
                ax.tick_params(axis='x', labelrotation=45, bottom=True, labelbottom=i == n_dims - 1, pad=2)
                ax.set_xlabel(dim_labels[_i], fontsize=fontsize, fontweight='bold', labelpad=4)
                # Buggy in matplotlib<=3.10 (https://github.com/matplotlib/matplotlib/pull/28629),
                # but needed to ignore top xlabels in constrained layout computation,
                # except for the top-most xlabel which otherwise collides with figure title.
                ax.xaxis.set_in_layout(i == 0)
                ax.yaxis.set_in_layout(j == n_dims - 1)

                if i == n_dims - 1:
                    # Add invisible ticks twice the length to make the spacing between
                    # the last two subplots match other horizontal spacings.
                    ax.secondary_yaxis('left').tick_params(
                        axis='y', labelleft=False, color='#fff0',
                        length=plt.rcParams["xtick.major.size"],
                    )

    fig.align_ylabels(axs[:, 0])
    fig.align_xlabels(axs[-1, :])
    return axs


def _partial_dependence(space: Space, bounds, estimator, i, j, sample_points, resolution=16):
    """Calculate the partial dependence for dimensions `i` and `j` with
    respect to the objective value, as approximated by `estimator`.

    The partial dependence plot shows how the value of the dimensions
    `i` and `j` influence the `estimator` predictions after "averaging out"
    the influence of all other dimensions.

    Parameters
    ----------
    space : `Space`
        The parameter space over which the minimization was performed.

    bounds : dict[int, tuple]

    estimator
        Fitted surrogate model for the objective function.

    i : int
        The first dimension for which to calculate the partial dependence.

    j : int or None, default=None
        The second dimension for which to calculate the partial dependence.
        To calculate the 1D partial dependence on `i` alone set `j=None`.

    sample_points : array-like, shape=(n_points, n_dims), default=None
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `resolution` steps.

    resolution : int, default=16
        Number of points at which to evaluate the partial dependence
        along each dimension `i` and `j`.

    Returns
    -------
    xi : ndarray, shape=(n_points,)
        The points at which the partial dependence was evaluated.
    yi : ndarray, shape=(n_points,)
        The points at which the partial dependence was evaluated.
    zi : ndarray, shape=(n_points, n_points)
        The value of the model at each point `(xi, yi)`.

    For 1D case, `yi = zi`.
    """
    def _check_resolution(resolution, col):
        if space._is_cat(col):
            return len(space._encoders[col].index)
        return resolution

    # 1D case
    if j is None:
        xi = np.linspace(*_get_lim('x', bounds, i), _check_resolution(resolution, i))
        yi = []
        rvs = np.array(sample_points)
        for x_ in xi:
            rvs[:, i] = x_
            yi.append(np.mean(estimator.predict(rvs)))
        return xi, yi

    # 2D case
    xi = np.linspace(*_get_lim('x', bounds, j), _check_resolution(resolution, j))
    yi = np.linspace(*_get_lim('y', bounds, i), _check_resolution(resolution, i))

    zi = []
    for x_ in xi:
        row = []
        for y_ in yi:
            rvs = np.array(sample_points)
            rvs[:, (j, i)] = (x_, y_)
            row.append(np.mean(estimator.predict(rvs)))
        zi.append(row)

    return xi, yi, np.array(zi).T


def _check_plot_dims(plot_dims, bounds) -> list[int]:
    """Return unique, sorted dimension indices."""
    if plot_dims is None:
        plot_dims = np.where(bounds[:, 0] != bounds[:, 1])[0]
        if not plot_dims.size:
            raise ValueError(f'All dimensions are constant: {bounds[:, 0].tolist()}')
    plot_dims = np.unique(plot_dims).astype(int, casting='safe')
    return plot_dims.tolist()


def _check_result(result: OptimizeResult):
    for attr in ('x', 'fun', 'xv', 'funv', 'nfev'):
        if not hasattr(result, attr):
            raise TypeError(
                'Caller up the traceback will only work with OptimizeResult '
                'from this package or one with the same API. The passed object was:'
                f'{result.__class__.__mro__}, attrs:{dir(result)}.')
    return result


def _check_space(result: OptimizeResult):
    space = getattr(result, 'space', None)
    if space is None:
        space = Space([(-np.inf, np.inf)] * len(result.x))
    return space


def _subplots_grid(n_dims, size, title):
    add_figure_title = n_dims > 1
    title_padding = 1 / n_dims if add_figure_title else 0
    fig, axs = plt.subplots(
        n_dims, n_dims, figsize=(size * n_dims, size * n_dims + title_padding),
        squeeze=False,
        # layout='compressed',  # No/default layout makes for the tightest graph
        subplot_kw=dict(box_aspect=1),
    )
    _watermark(fig)
    if add_figure_title:
        fig.suptitle(title, fontsize=10)
    scale_factor = (2 / n_dims) * (size / _DEFAULT_SUBPLOT_SIZE)
    margins = dict(
        left=scale_factor * .12,
        bottom=scale_factor * .12,
        right=1 - scale_factor * .08,
        top=1 - (.19 if add_figure_title else .13) * scale_factor
    )
    fig.subplots_adjust(**margins, hspace=.1, wspace=.1)
    return fig, axs


def _watermark(fig: plt.Figure):
    plt.rcParams['svg.fonttype'] = 'none'
    fig.text(.01, .01, 'Created with SAMBO, https://sambo-optimization.github.io',
             alpha=.01, gid='watermark')


def __rand_jitter(x, dx=.02, _rng=np.random.default_rng()):
    return x + _rng.standard_normal(len(x)) * dx * np.ptp(x)


def _maybe_jitter(jitter_amount, *dims, space):
    dims = list(dims)
    return [__rand_jitter(x, jitter_amount) for _i, (i, x) in enumerate(dims)]


def plot_objective(
        result: OptimizeResult,
        *,
        levels: int = 10,
        resolution: int = 16,
        n_samples: int = 250,
        estimator: Optional[str | _SklearnLikeRegressor] = None,
        size: float = _DEFAULT_SUBPLOT_SIZE,
        zscale: Literal['linear', 'log'] = 'linear',
        names: Optional[list[str]] = None,
        true_minimum: Optional[list[float] | list[list[float]]] = None,
        plot_dims: Optional[list[int]] = None,
        plot_max_points: int = 200,
        jitter: float = .02,
        cmap: str = 'viridis_r',
) -> Figure:
    """Plot a 2D matrix of partial dependence plots that show the
    individual influence of each variable on the objective function.

    The diagonal plots show the effect of a single dimension on the
    objective function, while the plots below the diagonal show
    the effect on the objective function when varying two dimensions.

    Partial dependence plot shows how the values of any two variables
    influence `estimator` predictions after "averaging out"
    the influence of all other variables.

    Partial dependence is calculated by averaging the objective value
    for a number of random samples in the search-space,
    while keeping one or two dimensions fixed at regular intervals. This
    averages out the effect of varying the other dimensions and shows
    the influence of just one or two dimensions on the objective function.

    Black dots indicate the points evaluated during optimization.

    A red star indicates the best found minimum (or `true_minimum`,
    if provided).

    .. note::
          Partial dependence plot is only an estimation of the surrogate
          model which in turn is only an estimation of the true objective
          function that has been optimized. This means the plots show
          an "estimate of an estimate" and may therefore be quite imprecise,
          especially if relatively few samples have been collected during the
          optimization, and especially in regions of the search-space
          that have been sparsely sampled (e.g. regions far away from the
          found optimum).

    Parameters
    ----------
    result : OptimizeResult
        The optimization result.

    levels : int, default=10
        Number of levels to draw on the contour plot, passed directly
        to `plt.contourf()`.

    resolution : int, default=16
        Number of points at which to evaluate the partial dependence
        along each dimension.

    n_samples : int, default=250
        Number of samples to use for averaging the model function
        at each of the `n_points`.

    estimator
        Last fitted model for estimating the objective function.

    size : float, default=2
        Height (in inches) of each subplot/facet.

    zscale : {'linear', 'log'}, default='linear'
        Scale to use for the z axis of the contour plots.

    names : list of str, default=None
        Labels of the dimension variables. Defaults to `['x0', 'x1', ...]`.

    plot_dims : list of int, default=None
        List of dimension indices to be included in the plot.
        Default uses all non-constant dimensions of
        the search-space.

    true_minimum : list of floats, default=None
        Value(s) of the red point(s) in the plots.
        Default uses best found X parameters from the result.

    plot_max_points: int, default=200
        Plot at most this many randomly-chosen evaluated points
        overlaying the contour plots.

    jitter : float, default=.02
        Amount of jitter to add to categorical and integer dimensions.
        Default looks clear for categories of up to about 8 items.

    cmap: str or Colormap, default='viridis_r'
        Color map for contour plots, passed directly to
        `plt.contourf()`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A 2D matrix of partial dependence sub-plots.

    Example
    -------
    .. image:: /objective.svg
    """
    result = _check_result(result)
    space = _check_space(result)
    plot_dims = _check_plot_dims(plot_dims, space._bounds)
    n_dims = len(plot_dims)
    bounds = dict(zip(plot_dims, space._bounds[plot_dims]))

    assert names is None or isinstance(names, Iterable) and len(names) == n_dims, (n_dims, plot_dims, names)

    if true_minimum is None:
        true_minimum = result.x
    true_minimum = np.atleast_2d(true_minimum)
    assert true_minimum.shape[1] == len(result.x), (true_minimum, result)

    true_minimum = space.transform(true_minimum)

    assert isinstance(plot_max_points, Integral) and plot_max_points >= 0, plot_max_points
    rng = np.random.default_rng(0)
    # Sample points to plot, but don't include points exactly at res.x
    inds = np.setdiff1d(
        np.arange(len(result.xv)),
        np.where(np.all(result.xv == result.x, axis=1))[0],
        assume_unique=True)
    plot_max_points = min(len(inds), plot_max_points)
    inds = np.sort(rng.choice(inds, plot_max_points, replace=False))

    x_samples = space.transform(result.xv[inds])
    samples = space.sample(n_samples)

    assert zscale in ('log', 'linear', None), zscale
    locator = LogLocator() if zscale == 'log' else None

    fig, axs = _subplots_grid(n_dims, size, "Partial dependence")

    result_estimator = getattr(result, 'model', [None])[-1]
    from sambo._estimators import _estimator_factory

    if estimator is None and result_estimator is not None:
        estimator = result_estimator
    else:
        _estimator_arg = estimator
        estimator = _estimator_factory(estimator, bounds, rng=0)
        if result_estimator is None and _estimator_arg is None:
            warnings.warn(
                'The optimization result process does not appear to have been '
                'driven by a model. You can still still observe partial dependence '
                f'of the variables as modeled by estimator={estimator!r}',
                UserWarning, stacklevel=2)
        estimator.fit(space.transform(result.xv), result.funv)
    assert isinstance(estimator, _SklearnLikeRegressor), estimator

    for _i, i in enumerate(plot_dims):
        for _j, j in enumerate(plot_dims[:_i + 1]):
            ax = axs[_i, _j]
            # diagonal line plot
            if i == j:
                xi, yi = _partial_dependence(
                    space, bounds, estimator, i, j=None, sample_points=samples, resolution=resolution)
                ax.plot(xi, yi)
                for m in true_minimum:
                    ax.axvline(m[i], linestyle="--", color="r", lw=1)
            # lower triangle contour field
            elif i > j:
                xi, yi, zi = _partial_dependence(
                    space, bounds, estimator, i, j, sample_points=samples, resolution=resolution)
                ax.contourf(xi, yi, zi, levels, locator=locator, cmap=cmap,
                            alpha=(1 - .2 * int(bool(plot_max_points))))
                for m in true_minimum:
                    ax.scatter(m[j], m[i], c='#d00', s=200, lw=.5, marker='*')
                if plot_max_points:
                    x, y = x_samples[:, j], x_samples[:, i]
                    if jitter:
                        x, y = _maybe_jitter(jitter, (j, x), (i, y), space=space)
                    ax.scatter(x, y, c='k', s=12, lw=0, alpha=.5)

    _format_scatter_plot_axes(fig, axs, space, plot_dims=plot_dims, dim_labels=names, size=size)
    return fig


def plot_evaluations(
        result: OptimizeResult,
        *,
        bins: int = 10,
        names: Optional[list[str]] = None,
        plot_dims: Optional[list[int]] = None,
        jitter: float = .02,
        size: int = _DEFAULT_SUBPLOT_SIZE,
        cmap: str = 'summer',
) -> Figure:
    """Visualize the order in which points were evaluated during optimization.

    This creates a 2D matrix plot where the diagonal plots are histograms
    that show distribution of samples for each variable.

    Plots below the diagonal are scatter-plots of the sample points,
    with the color indicating the order in which the samples were evaluated.

    A red star shows the best found parameters.

    Parameters
    ----------
    result : `OptimizeResult`
        The optimization result.

    bins : int, default=10
        Number of bins to use for histograms on the diagonal. This value is
        used for real dimensions, whereas categorical and integer dimensions
        use number of bins equal to their distinct values.

    names : list of str, default=None
        Labels of the dimension variables. Defaults to `['x0', 'x1', ...]`.

    plot_dims : list of int, default=None
        List of dimension indices to be included in the plot.
        Default uses all non-constant dimensions of
        the search-space.

    jitter : float, default=.02
        Ratio of jitter to add to scatter plots.
        Default looks clear for categories of up to about 8 items.

    size : float, default=2
        Height (in inches) of each subplot/facet.

    cmap: str or Colormap, default='summer'
        Color map for the sequence of scatter points.

    .. todo::
        Figure out how to lay out multiple Figure objects side-by-side.
        Alternatively, figure out how to take parameter `ax=` to plot onto.
        Then we can show a plot of evaluations for each of the built-in methods
        (`TestDocs.test_make_doc_plots()`).

    Returns
    -------
    fig : matplotlib.figure.Figure
        A 2D matrix of subplots.

    Example
    -------
    .. image:: /evaluations.svg
    """
    result = _check_result(result)
    space = _check_space(result)
    plot_dims = _check_plot_dims(plot_dims, space._bounds)
    n_dims = len(plot_dims)
    bounds = dict(zip(plot_dims, space._bounds[plot_dims]))

    assert names is None or isinstance(names, Iterable) and len(names) == n_dims, \
        (names, n_dims, plot_dims)

    x_min = space.transform(np.atleast_2d(result.x))[0]
    samples = space.transform(result.xv)
    color = np.arange(len(samples))

    fig, axs = _subplots_grid(n_dims, size, "Sequence & distribution of function evaluations")

    for _i, i in enumerate(plot_dims):
        for _j, j in enumerate(plot_dims[:_i + 1]):
            ax = axs[_i, _j]
            # diagonal histogram
            if i == j:
                # if dim.prior == 'log-uniform':
                #     bins_ = np.logspace(*np.log10(bounds[i]), bins)
                ax.hist(
                    samples[:, i],
                    bins=(int(bounds[i][1] + 1) if space._is_cat(i) else
                          min(bins, int(bounds[i][1] - bounds[i][0] + 1)) if space._is_int(i) else
                          bins),
                    range=None if space._is_cat(i) else bounds[i]
                )
            # lower triangle scatter plot
            elif i > j:
                x, y = samples[:, j], samples[:, i]
                if jitter:
                    x, y = _maybe_jitter(jitter, (j, x), (i, y), space=space)
                ax.scatter(x, y, c=color, s=40, cmap=cmap, lw=.5, edgecolor='k')
                ax.scatter(x_min[j], x_min[i], c='#d009', s=400, marker='*', lw=.5, edgecolor='k')

    _format_scatter_plot_axes(fig, axs, space, plot_dims=plot_dims, dim_labels=names, size=size)
    return fig
