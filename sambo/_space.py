from numbers import Integral

import joblib
import numpy as np

from sambo import OptimizeResult
from sambo._util import _check_random_state, _initialize_population, _sample_population, lru_cache


class Space:
    def __init__(self, bounds, constraints=None, rng=_check_random_state(0)):
        space = []
        dtypes = []
        object_encoders = {}
        for i, b in enumerate(bounds):
            b = np.asarray(b)
            assert not np.issubdtype(b.dtype, np.complexfloating), \
                'No support for complex dtypes. Model complex space as 2d float space yourself.'
            dtype = (int if len(b) == 2 and np.issubdtype(b.dtype, np.integer) else
                     float if len(b) == 2 and np.issubdtype(b.dtype, np.floating) else
                     object)
            b = b.astype(dtype)
            if dtype is object:
                object_encoders[i] = encoder = ObjectEncoder().fit(b)
                b = np.array([0, encoder.index.size - 1])
            if dtype is int:
                b = np.array([b[0], b[1] - 1])
            if dtype in (int, float):
                assert b[0] <= b[1], ('Numeric bounds require lb <= ul', i, b, bounds)
            space.append(b)
            dtypes.append(dtype)

        self._constraints = constraints
        self._rng = rng

        self._encoders = object_encoders
        self._bounds = space = np.array(space, order='F')
        space.flags.writeable = False

        self._dtypes = dtypes = np.array(dtypes)
        self._float_dtypes = np.where(dtypes == float)[0]
        self._object_dtypes = np.where(dtypes == object)[0]
        self._int_dtypes = np.where(dtypes == int)[0]
        self._is_all_floats = np.all(dtypes == float)

        # FIXME: Caching like this prevents pickling (e.g. joblib.Memory)
        def key_func(X, copy=False):
            nonlocal self
            return None if self._is_all_floats else (joblib.hash(X), copy)

        self.inverse_transform = lru_cache(1, key=key_func)(self.inverse_transform)

    def __iter__(self):
        return iter(self._bounds)

    def _is_int(self, col):
        return col in self._int_dtypes

    def _is_cat(self, col):
        return col in self._object_dtypes

    @staticmethod
    def _round_values(values):
        return np.round(values.astype(float)).astype(int)

    def _maybe_coerce_inplace(self, X):
        for inds in (self._int_dtypes, self._object_dtypes):
            if inds.size:
                X[:, inds] = self._round_values(X[:, inds])

    def sample(self, n, *, init=False, x0=None):
        """Sample transformed"""
        # TODO use this instead of the other function
        if init:
            X = _initialize_population(self._bounds, n, self._constraints, x0, self._rng)
        else:
            X = _sample_population(self._bounds, n, self._constraints, self._rng)
        self._maybe_coerce_inplace(X)
        return X

    def transform(self, X):
        """Transofrm X from user's original space to our representation (floats)."""
        if self._encoders:
            X = np.array(X, dtype=object)  # Make a copy!
            is_single_row = X.ndim == 1
            X = np.atleast_2d(X)
            for i, encoder in self._encoders.items():
                X[:, i] = encoder.transform(X[:, i])
            if is_single_row:
                X = X[0]
        return X

    def _label_categorical(self, col, value, *tick_pos):
        assert col in self._encoders, (col, self._encoders)
        assert isinstance(value, Integral) or int(value) == value, (col, value)
        return self._encoders[col].inverse_transform(int(value))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform our float-only representation to user's original space."""
        if self._is_all_floats:  # Fast path for the common case
            return np.clip(X, *self._bounds.T)

        # TODO: Use structured dtype instead?
        # dtype = [(str if dtype is object else dtype).__name__
        #          for i, (dtype, values) in enumerate(self._bounds)]
        # X = X.astype(', '.join(dtype))
        X = np.asarray(X, dtype=object)
        is_single_row = X.ndim == 1
        X = np.atleast_2d(X)
        assert X.ndim == 2, X

        # TODO remove all np.clip as no longer required
        for i in self._object_dtypes:
            X[:, i] = self._encoders[i].inverse_transform(self._round_values(X[:, i]))

        if (inds := self._int_dtypes).size:
            bounds = self._bounds[inds].T
            X[:, inds] = self._round_values(np.clip(X[:, inds], *bounds))

        if (inds := self._float_dtypes).size:
            bounds = self._bounds[inds].T
            X[:, inds] = np.clip(X[:, inds], *bounds)

        return X[0] if is_single_row else X

    def inverse_transform_result(self, res: OptimizeResult) -> OptimizeResult:
        return OptimizeResult(
            res,
            x=self.inverse_transform(res.x),
            xv=self.inverse_transform(res.xv),
        )


class ObjectEncoder:
    def fit(self, values):
        index = list(values)
        assert index == list({k: True for k in index}), f'Non-unique object values: {values}'
        self.index = np.array(values, dtype=object)
        self.inv_index = dict(zip(index, range(len(index))))
        return self

    def transform(self, x):
        """Transform X From original to ours"""
        x = np.asarray(x)
        assert np.all(np.isin(np.atleast_1d(x), self.index)), (x, self.index)
        to_idx = self.inv_index.__getitem__
        return np.array([to_idx(i) for i in x])

    def inverse_transform(self, i):
        i = np.asanyarray(i)
        assert i.dtype.kind == 'i', f'Invalid dtype for index operation: {i.dtype}'
        assert np.all(i < len(self.index)), (i, len(self.index), self.index)
        return self.index[i]
