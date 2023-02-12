import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    metrics = {'precision':'','recall':'','F1':'','accuracy':''}
    false_vals = y_pred[y_pred != y_true]
    true_vals = y_pred[y_pred == y_true]
    tp = sum(true_vals==1)
    tn = true_vals.shape[0] - tp
    fp = sum(false_vals==1)
    fn = false_vals.shape[0] - fp
    if tp == 0:
        metrics['precision'] = 0
        metrics['recall'] = 0
        metrics['F1'] = 0
    else:
        metrics['precision'] = tp / (tp + fp)
        metrics['recall'] = tp / (tp + fn)
        metrics['F1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    try:
        metrics['accuracy'] = true_vals.shape[0] / (true_vals.shape[0]+false_vals.shape[0])
    except ZeroDivisionError:
        metrics['accuracy'] = 0
    return metrics


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    return sum(y_pred == y_true)/y_true.shape[0]


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    return 1 - np.sum((y_true - y_pred)**2)/np.sum((y_true - np.mean(y_true))**2)


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    return np.sum((y_true - y_pred)**2)/y_true.shape[0]


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    return np.sum(np.abs(y_true - y_pred))/y_true.shape[0]
    