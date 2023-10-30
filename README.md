README for Machine Learning Algorithms

This repository contains a set of machine learning algorithms implemented in Python using NumPy. These algorithms cover various aspects of machine learning, including regression, classification, and gradient-based optimization. Below, you will find a brief description of each algorithm and its purpose.
Mean Squared Error (MSE) Gradient Descent (GD)
mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)

This function implements the Gradient Descent (GD) algorithm for solving a linear regression problem using Mean Squared Error (MSE) as the loss function. It iteratively updates the model parameters to minimize the MSE loss.

    y: Numpy array of shape (N,) representing the target values.
    tx: Numpy array of shape (N, D) representing the input data.
    initial_w: Numpy array of shape (D,) representing the initial guess for model parameters.
    max_iters: Scalar representing the maximum number of GD iterations.
    gamma: Scalar representing the step size (learning rate).

Returns:

    w: Numpy array of shape (D,) representing the optimal weights.
    loss: Scalar representing the final mean square error.

Mean Squared Error (MSE) Stochastic Gradient Descent (SGD)
mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)

This function implements Stochastic Gradient Descent (SGD) for solving a linear regression problem using Mean Squared Error (MSE) as the loss function. It updates the model parameters using a randomly selected data point (stochastic update).

    y: Numpy array of shape (N,) representing the target values.
    tx: Numpy array of shape (N, D) representing the input data.
    initial_w: Numpy array of shape (D,) representing the initial guess for model parameters.
    max_iters: Scalar representing the maximum number of SGD iterations.
    gamma: Scalar representing the step size (learning rate).

Returns:

    w: Numpy array of shape (D,) representing the optimal weights.
    loss: Scalar representing the final mean square error.

Least Squares
least_squares(y, tx)

This function calculates the least squares solution for a linear regression problem.

    y: Numpy array of shape (N,) representing the target values.
    tx: Numpy array of shape (N, D) representing the input data.

Returns:

    w: Numpy array of shape (D,) representing the optimal weights.
    loss: Scalar representing the mean square error.

Ridge Regression
ridge_regression(y, tx, lambda_)

This function implements Ridge Regression, a linear regression technique that adds a regularization term to the least squares cost function.

    y: Numpy array of shape (N,) representing the target values.
    tx: Numpy array of shape (N, D) representing the input data.
    lambda_: Scalar representing the regularization parameter.

Returns:

    w: Numpy array of shape (D,) representing the optimal weights.
    loss: Scalar representing the mean square error with ridge regularization.

Logistic Regression
logistic_regression(y, tx, initial_w, max_iters, gamma)

This function implements Logistic Regression using gradient descent. It is used for binary classification tasks.

    y: Numpy array of shape (N, 1) representing binary labels (0 or 1).
    tx: Numpy array of shape (N, D) representing the input data.
    initial_w: Numpy array of shape (D, 1) representing the initial guess for model parameters.
    max_iters: Scalar representing the maximum number of GD iterations.
    gamma: Scalar representing the step size (learning rate).

Returns:

    w: Numpy array of shape (D,) representing the optimal weights.
    loss: Scalar representing the final negative log-likelihood loss.

Regularized Logistic Regression
reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)

This function implements Regularized Logistic Regression using gradient descent. It includes L2 regularization to prevent overfitting.

    y: Numpy array of shape (N, 1) representing binary labels (0 or 1).
    tx: Numpy array of shape (N, D) representing the input data.
    lambda_: Scalar representing the regularization parameter.
    initial_w: Numpy array of shape (D, 1) representing the initial guess for model parameters.
    max_iters: Scalar representing the maximum number of GD iterations.
    gamma: Scalar representing the step size (learning rate).

Returns:

    w: Numpy array of shape (D,) representing the optimal weights.
    loss: Scalar representing the final negative log-likelihood loss with L2 regularization.

Utility Functions

The repository also includes several utility functions:

    compute_MSE_gradient(y, tx, w): Computes the gradient of Mean Squared Error (MSE) loss.
    compute_loss_MSE(y, tx, w): Calculates the MSE loss.
    sigmoid(t): Applies the sigmoid function to a scalar or array.
    calculate_NLL_loss(y, tx, w): Computes the cost by negative log-likelihood for logistic regression.
    calculate_NLL_gradient(y, tx, w): Computes the gradient of the negative log-likelihood cost.
    build_k_indices(y, k_fold, seed): Builds indices for k-fold cross-validation.
    accuracy(pred, y): Computes the accuracy of binary classification predictions.
    F1(pred, y): Computes the F1 score of binary classification predictions.
