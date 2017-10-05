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


def l_one_loss(y, yhat):
    """Get L1 loss score.

    Args:
        y: vector of size m (predicted labels)
        yhat: vector of size m (true labels)

    Returns:
        loss: L1 Loss
    """
    loss = np.sum(np.abs(y-yhat))

    return loss


def l_two_lost(y, yhat):
    """Get L2 loss score.

    Args:
        y: vector of size m (predicted labels)
        yhat: vector of size m (true labels)

    Returns:
        loss: L2 loss
    """
    loss = np.sum((y-yhat)**2)

    return loss


def softmax_test():
    input_x =  np.array([
                        [9, 2, 5, 0, 0],
                        [7, 5, 0, 0 ,0]])
    ouput = np.array([[9.80897665e-01, 8.94462891e-04, 1.79657674e-02, 1.21052389e-04, 1.21052389e-04], 
                      [ 8.78679856e-01, 1.18916387e-01, 8.01252314e-04, 8.01252314e-04, 8.01252314e-04]])
