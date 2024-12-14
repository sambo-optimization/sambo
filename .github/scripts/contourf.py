from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark.funcs import TEST_FUNCTIONS
from sambo import minimize

test_funcs = [test for test in TEST_FUNCTIONS if len(test['bounds']) == 2]
test_funcs[0]['func'] = lambda x: np.sin(x[1]) * np.exp((1-np.cos(x[0]))**2) + np.cos(x[0]) * np.exp((1-np.sin(x[1]))**2) + (x[0] - x[1])**2

fig, axes = plt.subplots(1, len(test_funcs), figsize=(len(test_funcs) * 2, 2), constrained_layout=True)


for ax, test in zip(axes, test_funcs):
    b = test['bounds']
    x = np.linspace(*b[0], 20)
    y = np.linspace(*b[1], 20)
    X, Y = np.meshgrid(x, y)

    Z = np.array([test['func'](np.array(x)) for x in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
    ax: plt.Axes
    cnt = ax.contourf(X, Y, Z, levels=100, cmap='Greys')
    cnt.set_edgecolor('face')
    ax.axis('off')

    res = minimize(test['func'], bounds=test['bounds'], max_iter=500, n_init=300, disp=True)
    ax.scatter(res.x[0], res.x[1], s=16, c='k', alpha=.4, marker='x')

fig.savefig(Path(__file__).parent.parent.parent / 'www/contourf.jpg')
plt.show()
