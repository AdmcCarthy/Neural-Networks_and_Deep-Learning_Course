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
    s = 1 / (1 + np.exp(x * -1))

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


def image2vec(x):
    """
    Convert an image to a vector.

    Reshape the image array into a one
    dimensional vector.

    Args:
        x: A numpy array of shape (length, height, depth).
    Returns:
        xv: A vector of shape (length*height*depth, 1).
    """

    xv = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], 1)

    return xv


def sigmoid_test():
    """
    Learn good practice to set up test code.
    """
    input_x = np.array([1, 2, 3])
    output = np.array([0.73105858,  0.88079708,  0.95257413])

    print(sigmoid(input_x))
    print(output)

    assert sigmoid(input) == output


def sigmoid_derivative_test():
    """
    Implement better docstring!!!
    """
    input_x = np.array([1, 2, 3])
    output = np.array([0.19661193, 0.10499359, 0.04517666])

    assert sigmoid_derivative(input_x) == output


def image2vec_test():
    """
    Test image2vec.
    """
    input_x = np.array([[[0.67826139,  0.29380381],
                        [0.90714982,  0.52835647],
                        [0.4215251,  0.45017551]],

                        [[0.92814219,  0.96677647],
                        [0.85304703,  0.52351845],
                        [0.19981397,  0.27417313]],

                        [[0.60659855,  0.00533165],
                        [0.10820313,  0.49978937],
                        [0.34144279,  0.94630077]]]
                        )

    output = np.array([[ 0.67826139] [ 0.29380381],
                      [ 0.90714982] [ 0.52835647], 
                      [ 0.4215251 ] [ 0.45017551], 
                      
                      [ 0.92814219] [ 0.96677647], 
                      [ 0.85304703] [ 0.52351845], 
                      [ 0.19981397] [ 0.27417313], 
                      
                      [ 0.60659855] [ 0.00533165], 
                      [ 0.10820313] [ 0.49978937], 
                      [ 0.34144279] [ 0.94630077]]
                      )


sigmoid_test()
sigmoid_derivative_test()
