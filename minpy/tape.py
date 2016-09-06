#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tape for recording gradient calculation."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement
import contextlib
import collections

import numpy

from . import array

GradRecord = collections.namedtuple(
    'GradRecord', ['grad_func', 'result', 'primitive_type', 'extra_grads'])


class Tape(object):
    """Records gradient calculation."""
    global_tape = None

    def __init__(self):
        self._grads = {}
        self._array_grad_records = {}

    def add_partial_derivative(self, grad_func, owner, result, primitive_type,
                               extra_grads):
        """Add partial derivative.

        Parameters
        ----------
        x : type
            Description of `x`.
        """
        self._array_grad_records[owner] = GradRecord(
            grad_func=grad_func,
            result=result,
            primitive_type=primitive_type,
            extra_grads=extra_grads)

    def set_gradient_targets(self, targets):
        """Set gradient targets to ones."""
        for i in targets:
            self._grads[i] = array.Value.wrap(1.0 if isinstance(
                i, array.Number) else numpy.ones(i.shape))

    def get_gradient(self, origin):
        dfs_queue = [origin]
        while len(dfs_queue) != 0:
            pass


@contextlib.contextmanager
def tape():
    """Convenience context wrapper for creating temporary `Tape`."""
    Tape.global_tape = Tape()
    yield
    Tape.global_tape = None


def global_tape():
    """Returns current global `Tape`."""
    return Tape.global_tape
