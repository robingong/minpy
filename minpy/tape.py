#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=logging-format-interpolation
"""Tape for recording gradient calculation."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement
import contextlib
import collections

import numpy

from . import array
from . import array_variants
from .utils import log

# pylint: disable=invalid-name
_logger = log.get_logger(__name__)
# pylint: enable=invalid-name

GradRecord = collections.namedtuple(
    'GradRecord', ['grad_func', 'result', 'primitive_type', 'owner'])


class Tape(object):
    """Records gradient calculation."""
    global_tape = None

    def __init__(self):
        self._grads = {}
        self._array_grad_records = {}

    def add_partial_derivative(self, grad_func, owner, result, primitive_type):
        """Add partial derivative.

        Parameters
        ----------
        grad_func
            Function for calculating gradient.
        owner
            Owners of the gradient. Usually it is just one array. But a list is also allowed.
        result
            Result of previous forward execution.
        primitive_type
            Primitive type indicating location of execution.
        """

        def add_owner(single_owner):
            """Add owner.

            Deal with situation when there are multiple owners.
            """
            if isinstance(single_owner, array.Value):
                if single_owner not in self._array_grad_records:
                    self._array_grad_records[single_owner] = []
                self._array_grad_records[single_owner].append(
                    GradRecord(
                        grad_func=grad_func,
                        result=result,
                        primitive_type=primitive_type,
                        owner=owner))
            else:
                for sub_owner in single_owner:
                    add_owner(sub_owner)

        add_owner(owner)

    def set_gradient_target(self, target):
        """Set gradient targets to ones."""

        def set_single_gradient_target(target):
            """Set gradient target for one."""
            if isinstance(target, array.Value):
                self._grads[target] = array.Value.wrap(1.0 if isinstance(
                    target, array.Number) else numpy.ones(target.shape))
            else:
                for sub_target in target:
                    set_single_gradient_target(sub_target)

        set_single_gradient_target(target)

    def _is_gradable(self, current_array):
        """Check if gradient could now be calculated relative to the specified array.

        It means that all resulting arrays have gradients calculated.
        """

        def check_grad_record_empty(arr):
            """Check if gradient record is empty for the array."""
            if isinstance(arr, array.Value):
                if len(self._array_grad_records.get(arr, [])) != 0:
                    return False
            else:
                for sub_arr in arr:
                    if not check_grad_record_empty(sub_arr):
                        return False
            return True

        for res in self._array_grad_records.get(current_array, []):
            if not check_grad_record_empty(res.result):
                return False
        return True

    def _get_cached_gradient(self, arr):
        """Get cached gradient. Initialize if not exist."""
        if arr not in self._grads:
            self._grads[arr] = array.Value.wrap(0.0 if isinstance(
                arr, array.Number) else numpy.zeros(arr.shape))
        return self._grads[arr]

    def _cumulate_gradient(self, arr, grad):
        def add_gradient(arr, grad):
            """Add gradient.

            Recurse when handed multiple arrays.
            """
            if isinstance(arr, array.Value):
                current_gradient = self._get_cached_gradient(arr)
                current_gradient += array.Value.wrap(grad)
                self._grads[arr] = current_gradient
            else:
                if len(arr) != len(grad):
                    _logger.warning('Number of gradients does not match.')
                for sub_arr, sub_grad in zip(arr, grad):
                    add_gradient(sub_arr, sub_grad)

        add_gradient(arr, grad)

    def get_gradient(self, origin):
        """Get gradient of the specified array.

        Target gradients must be set prior to this function call.
        """
        dfs_queue = [origin]
        while len(dfs_queue) != 0:
            current_array = dfs_queue[-1]
            assert isinstance(current_array,
                              array.Value), 'Type is not `array.Value`.'
            if self._is_gradable(current_array):
                dfs_queue.pop()
                # Initialize if necessary.
                self._get_cached_gradient(current_array)
                grad_records = self._array_grad_records.get(current_array, [])

                def get_result_grad(result, primitive_type):
                    """Get gradient of result."""
                    if isinstance(result, array.Value):
                        return self._grads[result].get_data(primitive_type)
                    else:
                        return [get_result_grad(sub_result, primitive_type)
                                for sub_result in result]

                def get_context(result):
                    """Get context of result."""
                    if isinstance(result, array.Value):
                        return result.context
                    else:
                        return get_context(result[0])

                for grad_record in grad_records:
                    result_grad_value = get_result_grad(
                        grad_record.result, grad_record.primitive_type)
                    _logger.debug(
                        'Calling derivative func "{}" with type "{}".'.format(
                            grad_record.grad_func, grad_record.primitive_type))
                    if grad_record.primitive_type == array_variants.ArrayType.MXNET:
                        with get_context(grad_record.result).as_mxnet_context(
                        ):
                            grad = grad_record.grad_func(result_grad_value)
                    else:
                        grad = grad_record.grad_func(result_grad_value)
                    self._cumulate_gradient(grad_record.owner, grad)

                def remove_grad_record(owner, grad_record):
                    """Remove gradient record from owner."""
                    if isinstance(owner, array.Value):
                        self._array_grad_records[owner].remove(grad_record)
                    else:
                        for sub_owner in owner:
                            remove_grad_record(sub_owner, grad_record)

                for grad_record in grad_records.copy():
                    remove_grad_record(grad_record.owner, grad_record)
            else:

                def push_grad_record(arr):
                    """Push gradient record for a result array."""
                    if isinstance(arr, array.Value):
                        dfs_queue.append(arr)
                    else:
                        for sub_arr in arr:
                            push_grad_record(sub_arr)

                for rec in self._array_grad_records[current_array]:
                    push_grad_record(rec.result)
        return self._get_cached_gradient(origin)


@contextlib.contextmanager
def tape():
    """Convenience context wrapper for creating temporary `Tape`."""
    Tape.global_tape = Tape()
    yield Tape.global_tape
    Tape.global_tape = None


def global_tape():
    """Returns current global `Tape`."""
    return Tape.global_tape
