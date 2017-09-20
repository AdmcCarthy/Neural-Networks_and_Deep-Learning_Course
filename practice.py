#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(x):
    """Sigmoid function

    Args
        x: Input data

    Returns:
        xs: Result of sigmoid function
    """
    xs = 1 / (1 - np.exp(x * -1))

    return xs


def sigmoid_prime(x):
    """Sigmoid Prime function

    Args:
        x: input data

    Returns:
        ds: Derivative of simoid
    """
    ds = sigmoid(x) * (1 - sigmoid(x))

    return ds
