import numpy as np
import random as rand
import time
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The mean squared error Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: a scalar denoting the mean square error
    """
    w = initial_w
    for n_iter in range(max_iters):
    
        gradient = compute_MSE_gradient(y,tx,w) #Compute gradient
        w = w - gamma*gradient # Update weights
 
    loss = compute_loss_MSE(y,tx,w)
    return w,loss
    
def compute_MSE_gradient(y,tx,w):
    """Computes the MSE gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx.dot(w)
    gradient = -1/y.size * tx.T.dot(e)
    return gradient
    
def compute_loss_MSE(y,tx,w):
    """Calculate the MSE loss

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    return 0.5/len(e)*e.T.dot(e)

def standardize(x):
    """Standardize the original data set by feature.
    Args:
        x: numpy array of shape(N,D) N is thee number of samples and D the number of features
    """
    
    mean_x = np.nanmean(x,axis=0)
    x = x - mean_x
    std_x = np.nanstd(x,axis=0)
    x = x / std_x
    return x, mean_x, std_x

def mean_squared_error_sgd(y, tx, initial_w,max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: a scalar denoting the mean square error
    """

    w = initial_w
    for n_iter in range(max_iters):
        random_index = rand.randint(0,len(y)-1)
        gradient = compute_MSE_gradient(y[random_index],tx[random_index,:],w)
        w = w - gamma*gradient
        
    loss = compute_loss_MSE(y,tx,w)
    return w,loss

# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: a scalar denoting the mean square error
    """
    w=np.linalg.solve(tx.T.dot(tx),tx.T.dot(y))
    error = y -tx.dot(w)
    loss = 0.5*error.T.dot(error)/len(error)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: a scalar denoting the mean square error
    """
    N=len(y)
    D = tx.shape[1]
    lambda_ = lambda_*2*N
    w_ridge = np.dot(np.dot(np.linalg.inv( np.dot(tx.T,tx) + lambda_*np.identity(D) ),tx.T),y)
    loss = compute_loss_MSE(y,tx,w_ridge)
    return w_ridge,loss
    
def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    
    return 1/(1+np.exp(-t))
def calculate_NLL_loss(y, tx, w):
    """compute the cost by negative log likelihood. with y equal to 0 or 1

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    N=len(y)
    L = 1/N * (np.sum(np.log(1+np.exp(tx.dot(w)))) - y.T.dot(tx.dot(w)))
    return L.squeeze()

def calculate_NLL_gradient(y, tx, w):
    """compute the gradient of negative log likelihood cost.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        grad: a vector of shape (D, 1)
    """
    N = len(y)
    grad = 1/N * tx.T.dot(sigmoid(tx.dot(w)) - y )
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Implements logistic regression using gradient descent.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w: shape=(D, 1)
        max_iters: scalar denoting the number of iterations
        gamma: scalar denoting the step size

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: a scalar denoting the mean square error
    """
    threshold = 1e-8
    losses = []
    w=initial_w
    if(max_iters==0):
        losses.append(calculate_NLL_loss(y,tx,w))
    for iter in range(max_iters):
        gradient = calculate_NLL_gradient(y,tx,w)
        w = w-gamma*gradient #Update the weights
        loss=calculate_NLL_loss(y,tx,w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w,losses[-1]

def reg_logistic_regression(y, tx,lambda_, initial_w, max_iters, gamma):
    """Implements logistic regression using gradient descent.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w: shape=(D, 1)
        max_iters: scalar denoting the number of iterations
        gamma: scalar denoting the step size

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: a scalar denoting the mean square error
    """
    threshold = 1e-8
    losses = []
    w=initial_w
    if(max_iters==0):
        losses.append(calculate_NLL_loss(y,tx,w))
    for iter in range(max_iters):
        gradient = calculate_NLL_gradient(y,tx,w)
        gradient += lambda_*2*w
        w = w-gamma*gradient #Update the weights
        loss=calculate_NLL_loss(y,tx,w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w,losses[-1]
    
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def accuracy(pred,y):
    """Computes the accuracy of the predictions
    
    Args:
        pred: numpy array of shape=(N,) probabilities
        y: numpy array of shape=(N,) real values
    Returns:
        A scalar containing the accuracy
    """
    N = len(y)
    pred[pred>=0.5]=1
    pred[pred<0.5]=0
    return 1/N*np.count_nonzero(pred==y)


def reg_logistic_regression_stoch(y, tx,lambda_, initial_w, max_iters, gamma,batch_size):
    """Implements logistic regression using gradient descent.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w: shape=(D, 1)
        max_iters: scalar denoting the number of iterations
        gamma: scalar denoting the step size

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: a scalar denoting the mean square error
    """
    w=initial_w
    N=len(y)
    ix=np.arange(N)
    for iter in range(max_iters):
        batch = rand.choices(ix,k=batch_size)
        gradient = calculate_NLL_gradient(y[batch],tx[batch],w)
        gradient += lambda_*2*w
        w = w-gamma*gradient #Update the weights
        
    loss  = calculate_NLL_loss(y,tx,w) #Compute the loss
    return w,loss

def F1(pred,y):
    """Computes the F1 score of the predictions
    
    Args:
        pred: numpy array of shape=(N,) probabilities
        y: numpy array of shape=(N,) real values (0 or 1)
    Returns:
        A scalar containing the accuracy
    """
    N = len(y)
    pred[pred>=0.5]=1
    pred[pred<0.5]=0
    TP = np.count_nonzero(np.logical_and(y==1,pred==1))
    FP = np.count_nonzero(np.logical_and(y==0,pred==1))
    FN = np.count_nonzero(np.logical_and(y==1,pred==0))
    TN = np.count_nonzero(np.logical_and(y==0,pred==0))
    
    P=0
    R=0
    if(TP!=0):
        P= TP/(TP+FP)
        R=TP/(TP+FN)
        
    if((P+R) == 0):
        return 0
    return 2*(P*R)/(P+R)