#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Drills for neural network section
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Repetative practice drills
"""
import numpy as np


def sigmoid(x):
    """
    Returns the sigmoid of a number.

    This is a non-linear
    function also known as the logistic function.

    The exponential can be done using the math
    or numpy libary, however as the inputs
    are commonly matrices or vectors and not
    real numbers numpy is used.

    Args:
        x: A scalar

    Return:
        s: sigmoid(x)
    """
    s = 1 / (1 + np.exp(x*-1))

    return s


def sigmoid_test():
    """
    Learn good practice to set up test code.
    """
    input = [1, 2, 3]
    output = [0.73105858,  0.88079708,  0.95257413]

    assert sigmoid(input) == output


sigmoid_test
