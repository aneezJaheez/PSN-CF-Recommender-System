# Author: Taylor G Smith

from __future__ import absolute_import, division

import numpy as np
from abc import ABCMeta, abstractmethod

from sklearn.externals import six
from sklearn.utils.validation import check_random_state
from sklearn.utils import validation as skval

import pandas as pd

from scipy import sparse

import numbers

__all__ = [
    'BootstrapCV',
    'check_cv',
    'train_test_split'
]

MAX_SEED = 1e6
ITYPE = np.int32
DTYPE = np.float64  # implicit asks for doubles, not float32s...


def check_consistent_length(u, i, r):
    skval.check_consistent_length(u, i, r)
    return np.asarray(u), np.asarray(i), np.asarray(r, dtype=DTYPE)


def _make_sparse_csr(data, rows, cols, dtype=DTYPE):
    # check lengths
    check_consistent_length(data, rows, cols)
    data, rows, cols = (np.asarray(x) for x in (data, rows, cols))

    shape = (np.unique(rows).shape[0], np.unique(cols).shape[0])
    return sparse.csr_matrix((data, (rows, cols)),
                             shape=shape, dtype=dtype)


def to_sparse_csr(u, i, r, axis=0, dtype=DTYPE):
    if axis not in (0, 1):
        raise ValueError("axis must be an int in (0, 1)")

    rows = u if axis == 0 else i
    cols = i if axis == 0 else u
    return _make_sparse_csr(data=r, rows=rows, cols=cols, dtype=dtype)


def _validate_train_size(train_size):
    assert isinstance(train_size, float) and (0. < train_size < 1.), \
        "train_size should be a float between 0 and 1"


def _get_stratified_tr_mask(u, i, train_size, random_state):
    _validate_train_size(train_size)  # validate it's a float
    random_state = check_random_state(random_state)
    n_events = u.shape[0]
    train_mask = random_state.rand(n_events) <= train_size  # type: np.ndarray

    for array in (u, i):
        present = array[train_mask]
        test_vals = array[~train_mask]
        missing = np.unique(test_vals[np.where(
            ~np.in1d(test_vals, present))[0]])

        if missing.shape[0] == 0:
            continue

        array_mask_missing = np.in1d(array, missing)
        where_missing = np.where(array_mask_missing)[0]
        
        added = set()
        for idx, val in zip(where_missing, array[where_missing]):
            if val in added:  # O(1) lookup
                continue
            train_mask[idx] = True
            added.add(val)

    return train_mask


def _make_sparse_tr_te(users, items, ratings, train_mask):
    r_train = to_sparse_csr(u=users[train_mask], i=items[train_mask],
                            r=ratings[train_mask], axis=0)
    r_test = to_sparse_csr(u=users, i=items, r=ratings, axis=0)
    return r_train, r_test


def train_test_split_cf(u, i, r, train_size=0.75, random_state=None):
    users, items, ratings = check_consistent_length(u, i, r)
    train_mask = _get_stratified_tr_mask(
        users, items, train_size=train_size,
        random_state=random_state)
    return _make_sparse_tr_te(users, items, ratings, train_mask=train_mask)