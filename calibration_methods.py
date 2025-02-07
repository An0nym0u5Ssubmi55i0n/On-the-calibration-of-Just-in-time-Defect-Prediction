import numpy as np
import ast
import csv
import calculate_calibration_metrics
from calculate_calibration_metrics import calculate_brier_score, calculate_ece, calculate_mce
from conformal_prediction import apply_calibration, apply_conformal_prediction
import torch
import math
from copy import deepcopy

from torch import nn

from pycalib.models.calibrators import BinningCalibration
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.isotonic import IsotonicRegression as IR
from sklearn.metrics import roc_auc_score, log_loss

from netcal.regression import IsotonicRegression as nIR
from netcal.scaling import LogisticCalibration as nPS
from netcal.binning import HistogramBinning as nHB
from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE



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




