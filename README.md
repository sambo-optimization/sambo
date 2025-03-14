[![SAMBO - Sequential And Model-Based Optimization](logo.svg)](https://sambo-optimization.github.io/)
=====
[![Build Status](https://img.shields.io/github/actions/workflow/status/sambo-optimization/sambo/ci.yml?branch=master&style=for-the-badge)](https://github.com/sambo-optimization/sambo/actions)
[![Code Coverage](https://img.shields.io/badge/coverage-96%25-%2397ca00?style=for-the-badge&label=Covr)](https://github.com/sambo-optimization/sambo/actions/workflows/ci.yml)
[![Source lines of code](https://img.shields.io/endpoint?url=https%3A%2F%2Fghloc.vercel.app%2Fapi%2Fsambo-optimization%2Fsambo%2Fbadge?filter=.py%26format=human&style=for-the-badge&label=SLOC&color=green)](https://ghloc.vercel.app/sambo-optimization/sambo)
[![sambo on PyPI](https://img.shields.io/pypi/v/sambo.svg?color=blue&style=for-the-badge)](https://pypi.org/project/sambo)
[![package downloads](https://img.shields.io/pypi/dm/sambo.svg?style=for-the-badge&color=skyblue&label=D/L)](https://pypistats.org/packages/sambo)
[![total downloads](https://img.shields.io/pepy/dt/sambo?style=for-the-badge&label=%E2%88%91&color=skyblue)](https://pypistats.org/packages/sambo)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/kernc?color=pink&style=for-the-badge&label=%E2%99%A5)](https://github.com/sponsors/kernc)

SAMBO: Sequential And Model-Based (Bayesian) Optimization of black-box objective functions.

[**Project website**](https://sambo-optimization.github.io)

[Documentation]

[Documentation]: https://sambo-optimization.github.io/doc/sambo/


Installation
------------
```shell
$ pip install sambo
# or
$ pip install 'sambo[all]'   # Pulls in Matplotlib, scikit-learn
```


Usage
-----
See [examples on the project website](https://sambo-optimization.github.io/#examples).


Features
--------
* Python 3+
* Simple usage, standard API.
* Algorithms prioritize to minimize number of evaluations of the objective function: SHGO, SCE-UA and SMBO available.
* Minimal dependencies: NumPy, SciPy (scikit-learn & Matplotlib optional).
* State-of-the-art performance—see [benchmark results](https://sambo-optimization.github.io/#benchmark)
  against other common optimizer implementations.
* Integral, real (floating), and categorical dimensions.
* Fast approximate global black-box optimization.
* [Beautiful Matplotlib charts](https://sambo-optimization.github.io/#examples).





Development
-----------
Check [CONTRIBUTING.md](CONTRIBUTING.md) for hacking details.
