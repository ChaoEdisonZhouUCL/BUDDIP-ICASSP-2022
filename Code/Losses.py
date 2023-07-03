"""
Description:
    this is the module provides losses functions.

"""
import sys

import numpy as np

sys.path.append("..")


# {code}


def RMSE_metric(y_true, y_pred):
    """

    :param y_true: (No_Pixels, No_Endm)
    :param y_pred: (No_Pixels, No_Endm)
    :return:
    """
    MSE = np.mean(np.square(y_true - y_pred), axis=-1)
    rmse = np.sqrt(MSE)
    # averaged over pixels
    return np.mean(rmse)


def MAE_metric(y_true, y_pred):
    MAE = np.abs(y_true - y_pred) * 100
    return np.mean(MAE)


def angle_distance_metric(y_true, y_pred, verbose=False):
    """

    :param y_true: (No_Endm, No_Bands)
    :param y_pred: (No_Endm, No_Bands)
    :return:
    """

    dot_product = np.sum(y_true * y_pred, axis=-1)
    l2_norms = np.linalg.norm(y_true, axis=-1) * np.linalg.norm(y_pred, axis=-1) + 1e-8
    cosine_similarity = dot_product / l2_norms

    # Clamp the cosine similarity to the range [-1, 1] to avoid NaN values
    eps = 1e-7
    cosine_similarity = np.clip(cosine_similarity, a_min=-1.0 + eps, a_max=1.0 - eps)
    AAD = np.arccos(cosine_similarity) * 180.0 / np.pi

    if verbose:
        print(f"angle distance is: {AAD}")

    # returned value is averaged over different number of endmembers
    if verbose:
        return AAD, np.mean(AAD)
    else:
        return np.mean(AAD)
