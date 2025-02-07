import numpy as np
import ast
import csv
import calculate_calibration_metrics
import torch
import math
from copy import deepcopy
from torch import nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression as LR
from netcal.scaling import LogisticCalibration as nPS
from netcal.scaling import TemperatureScaling



# uses logits and calculates them from confidences
def platt_scaling(predictions_calibration, labels_calibration, predictions_test, lables_test):
    """ Netcal Impl of Platt Scaling/Logistic Regression
    """
    nps = nPS()
    nps.fit(predictions_calibration, labels_calibration)
    p_calibrated_test = nps.transform(predictions_test)
    p_calibrated_cal = nps.transform(predictions_calibration)
    return np.asarray(p_calibrated_test), np.asarray(p_calibrated_cal)

# uses logits
def temperature_scaling(predictions_calibration, labels_calibration, predictions_test, lables_test):
    """ Netcal impl of TemperatureScaling
    """
    temperature = TemperatureScaling()
    temperature.fit(predictions_calibration, labels_calibration)
    p_calibrated_test = temperature.transform(predictions_test)
    p_calibrated_cal = temperature.transform(predictions_calibration)
    return np.asarray(p_calibrated_test), np.asarray(p_calibrated_cal)


def invert_sigmoid_scores(predictions):
    # https://stackoverflow.com/questions/66116840/inverse-sigmoid-function-in-python-for-neural-networks
    inverted_np = np.log(predictions) - np.log(1 - predictions)
    return inverted_np




