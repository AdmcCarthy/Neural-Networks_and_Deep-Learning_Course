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
        x: A scalar or numpy array.

    Return:
        s: sigmoid(x)
    """
    s = 1 / (1 + np.exp(x*-1))

    return s


def sigmoid_derivative(x):
    """
    Compute the gradient (otherwise known as the slope or derivative) 
    of the sigmoid function with respect to it´s input x.

    Args:
        x: A scalar or numpy array,

    Returns:
        ds: The computed gradient.
    """
    ds = sigmoid(x) * (1 - sigmoid(x))

    return ds


def sigmoid_test():
    """
    Learn good practice to set up test code.
    """
    input = np.array([1, 2, 3])
    output = np.array([0.73105858,  0.88079708,  0.95257413])

    print(sigmoid(input))
    print(output)

    assert sigmoid(input) == output

    print("Sigmoid OK")


def sigmoid_derivative_test():
    """
    Implement better docstring!!!
    """
    input = np.array([1, 2, 3])
    output = np.array([0.19661193, 0.10499359, 0.04517666])


sigmoid_test()
sigmoid_derivative_test()
