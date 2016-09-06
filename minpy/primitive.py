#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Primitive class definitions."""
from __future__ import absolute_import
from __future__ import print_function

import functools
import itertools
import collections
import operator

from .array import Value
from .array_variants import ArrayType
from . import context
from .utils import log
from . import tape

# pylint: disable= invalid-name
_logger = log.get_logger(__name__)
# pylint: enable= invalid-name

GradFunc = collections.namedtuple(
    'GradFunc', ['f', 'extra_grad_indices', 'extra_grad_kws'])


class Primitive(object):
    """Class for primitives. It includes both forward function and gradient definition."""
    __slots__ = [
        '_func',
        '_grad_func',
        '_grad_func_kw',
        '_type',
        '_mutate_args',
        '_mutate_kw',
    ]

    def __init__(self, func, ty, mutate_args=None, mutate_kw=None):
        """Initialize.

        Parameters
        ----------
        func
            A function that performs the forward computation.
        ty
            Type of primitive.
        mutate_args
            Whether the function mutates arguments.
        mutate_kw
            Whether the function mutates arguments.
        """
        self._func = func
        self._grad_func = {}
        self._grad_func_kw = {}
        self._type = ty
        self._mutate_args = [] if mutate_args is None else mutate_args
        self._mutate_kw = [] if mutate_kw is None else mutate_kw

    @property
    def type(self):
        """Return the type of the primitive (:attribute:`ArrayType.NUMPY` or :attribute:`ArrayType.MXNET`)."""
        return self._type

    @property
    def typestr(self):
        """Return the string representation of primitive type.

        Returns
        -------
        string
            String representation.
        """
        if self._type == ArrayType.NUMPY:
            return "NumPy"
        elif self._type == ArrayType.MXNET:
            return "MXNet"
        else:
            raise NotImplementedError()

    def __str__(self):
        return self._func.__name__

    def __call__(self, *args, **kwargs):
        """Call wrapped function.

        Parameters
        ----------
        args
            Arguments for the wrapped function.
        kwargs
            Arguments for the wrapped function.

        Returns
        -------
        array.Value
            An :class:`array.Value` representing the result.

        Raises
        ------
        IndexError
            No corresponding gradient function.
        KeyError
            No corresponding gradient function.
        """
        # pylint: disable= missing-docstring, invalid-name
        _logger.debug('Calling {} type {}.'.format(self._func, self.typestr))

        def get_val(x, mutate):
            try:
                xv = Value.wrap(x)
                if mutate:
                    return xv.get_data_mutable(self._type)
                else:
                    return xv.get_data(self._type)
            # If wrap failed, just return the original value.
            except TypeError:
                return x
        # Get underlying data.
        arg_values = tuple(
            get_val(args[i], i in self._mutate_args) for i in range(len(args)))
        kwargs_values = {
            k: get_val(kwargs[k], k in self._mutate_kw)
            for k in kwargs
        }
        # Call the real function with raw value.
        if self.type == ArrayType.MXNET:
            with context.current_context().as_mxnet_context():
                result_value = self._func(*arg_values, **kwargs_values)
        else:
            result_value = self._func(*arg_values, **kwargs_values)
        # If you want to do profiling, try to use `minprof(func)`.
        # result_value = minprof(self._func)(*arg_values, **kwargs_values)

        # Whether the result is on the bp path.
        def scan(accum, x):
            if isinstance(x, Value):
                return operator.or_(accum, x.marked_for_bp)
            else:
                return accum
        # Check whether the result value is on the path of bp phase.
        # If all the input arguments are not on the bp path, the result value
        # is not as well.
        need_bp = functools.reduce(scan, itertools.chain(
            args, kwargs.values()), False)
        # Wrap the result raw value with wrapper and node.
        result = Value.wrap(result_value, marked=need_bp)
        if need_bp:
            # Record partial derivative paths, only for `Value` type values.
            # If no gradient function is defined, also omit it
            for i, arg in enumerate(args):
                if isinstance(arg, Value) and arg.marked_for_bp:
                    if i not in self._grad_func:
                        _logger.warning('Partial derivative of func "{}" on \
                            argument {} is not defined.'
                                        .format(self._func.__name__, i))
                        continue
                    _logger.debug(
                        'Adding partial derivative to func "{}" on argument {}.'.
                        format(self._func, i))
                    grad_func = self._grad_func[i](result_value, *arg_values,
                                                   **kwargs_values)
                    tape.global_tape().add_partial_derivative(
                        grad_func.f, arg, result, self.type,
                        [args[i] for i in grad_func.extra_grad_indices] +
                        [kwargs[i] for i in grad_func.extra_grad_kws])
            for k, arg in kwargs.items():
                if isinstance(arg, Value) and arg.marked_for_bp:
                    if k not in self._grad_func_kw:
                        _logger.warning(
                            'Partial derivative of func "{}" on keyword argument "{}"\
                            is not defined.'.format(self._func.__name__, k))
                        continue
                    _logger.debug(
                        'Adding partial derivative to func "{}" on keyword argument "{}".'.
                        format(self._func, k))
                    grad_func = self._grad_func_kw[k](
                        result_value, *arg_values, **kwargs_values)
                    tape.global_tape().add_partial_derivative(
                        grad_func.f, arg, result, self.type,
                        [args[i] for i in grad_func.extra_grad_indices] +
                        [kwargs[i] for i in grad_func.extra_grad_kws])
        return result
        # pylint: enable= missing-docstring, invalid-name

    def def_grad(self, func, argnum=0):
        """Define gradient function.

        :param func: Gradient function.
        :param argnum: Index of the argument.
        :return: Self.
        """
        self._grad_func[argnum] = GradFunc(
            f=func, extra_grad_indices=(), extra_grad_kws=())
        return self

    def def_grad_kw(self, func, key):
        """Define gradient function.

        :param func: Gradient function.
        :param key: Key name of the argument.
        :return: Self.
        """
        self._grad_func_kw[key] = GradFunc(
            f=func, extra_grad_indices=(), extra_grad_kws=())
        return self

    def def_grad_zero(self, argnum=0):
        """Define zero gradient

        :param argnum: Index of the argument.
        :return: Self.
        """
        self.def_grad(lambda *args, **kwargs: lambda g: 0.0, argnum)
        return self

    def def_multiple_grad(self, func, argnums, kws):
        """Define multiple gradients with one function.

        :param func: Gradient function.
        :param argnums: Indexes of the arguments.
        :param kws: Key names of the arguments.
        """
        argnums = set(argnums)
        kws = set(kws)
        for argnum in argnums:
            extra_set = argnums.copy()
            extra_set.remove(argnum)
            self._grad_func[argnum] = GradFunc(
                f=func,
                extra_grad_indices=tuple(extra_set),
                extra_grad_kws=tuple(kws))
        for key in kws:
            extra_set = kws.copy()
            extra_set.remove(key)
            self._grad_func_kw[key] = GradFunc(
                f=func,
                extra_grad_indices=tuple(argnums),
                extra_grad_kws=tuple(extra_set))

    def gradable(self, bp_args, bp_kwargs):
        """Return whether the primitive has gradient function defined.

        :param tuple bp_args: Positional arguments that need back propagation.
        :param tuple bp_kwargs: Keyword arguments that need back propagation.
        :return: Whether all the arguments have gradients defined.
        """
        for i in bp_args:
            if i not in self._grad_func:
                return False
        for i in bp_kwargs:
            if i not in self._grad_func_kw:
                return False
        return True
