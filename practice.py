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


def image2vec(x):
    """Convert an image to a vector.

    Reshape the image array into a one
    dimensional vector.

    Args:
        x: A numpy array of shape (length, height, depth).
    Returns:
        xv: A vector of shape (length*height*depth, 1).
    """

    xv = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], 1)

    return xv


def normalize_rows(x):
    """Normalize rows.

    Args:
        x: A numpy matrix of shape (n, m)

    Returns:
     xn: The normalized (by row) numpy matrix.
    """
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    xn = x / x_norm

    return xn


def softmax(x):
    """Use a softmax function on x.

    Args:
        x: A numpy matrix of shape (n,m)

    Returns:
        xs: A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    x_exp = np.exp(x)

    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    xs = x_exp / x_sum

    return xs
