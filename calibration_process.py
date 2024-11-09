import numpy as np
import ast
import csv
import calibration_metrics as calculate_metrics
from calculate_calibration_metrics import calculate_brier_score, calculate_ece, calculate_mce
from calibration_methods import custom_platt_scaling, platt_scaling, temperature_scaling, invert_sigmoid_scores
import torch
import math
from copy import deepcopy

from torch import nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score, log_loss
from netcal.scaling import LogisticCalibration as nPS
from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE

DEFECT_LABEL = 1.0
HEALTHY_LABEL = 0.0
DEFECT_CLASSIFICATION_VALUE = 0.5

# TODO change, so that QT and OP
def read_input(filename):
    predictions = []
    labels = []
    confusion_matrix_category = []
    classification_value = 0
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index == 0: continue
            if index == 1: classification_value = float(row[3])
            predictions.append(float(row[1]))
            labels.append(float(row[2]))
            confusion_matrix_category.append(row[5])
    if classification_value != DEFECT_CLASSIFICATION_VALUE: raise RuntimeError("Classifcation Values don't match! ")
    return predictions, labels, confusion_matrix_category


def write_calibrated_metrics(rows):
    """ Method used to write all metrics for all calibrated predictions to CSV
    """
    write_filename = f"Calibrated_Predictions_Metrics_evaluation_op.csv"
    # row_structure = metrics, metrics_names, custom_eces, custom_eces_names, custom_mces, custom_mces_names

    with open(write_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        metrics, metrics_names, custom_eces, custom_eces_names, custom_mces, custom_mces_names = rows[0]
        header = ["index", *metrics_names, *custom_eces_names, *custom_mces_names]
        writer.writerow(header)
        for index, raw_row in enumerate(rows):
            metrics, metrics_names, custom_eces, custom_eces_names, custom_mces, custom_mces_names = raw_row
            row = [index, *metrics, *custom_eces, *custom_mces]
            writer.writerow(row)


def write_calibrated_predictions(preds_test, preds_test_inverted, labels_test, preds_cal, preds_cal_inverted, labels_cal, p_calibrated, iteration):
    """ Method used to write all calibrated predictions for both test and cal set to CSV
    """
    write_filename_cal = f"Calibrated_Predictions_eval_calibration_True_iteration_{iteration}_evaluation_op.csv"
    write_filename_test = f"Calibrated_Predictions_eval_calibration_False_iteration_{iteration}_evaluation_op.csv"
    
    p_calibrated_inv_ps,p_calibrated_inv_ps_cal, p_calibrated_nps, p_calibrated_nps_cal, p_calibrated_nts, p_calibrated_nts_cal, p_calibrated_ir, p_calibrated_ir_cal, p_calibrated_nir, p_calibrated_nir_cal, p_calibrated_hb_15_static, p_calibrated_hb_50_static, p_calibrated_hb_15_interactive, p_calibrated_hb_50_interactive, p_calibrated_hb_cal_15_static, p_calibrated_hb_cal_50_static, p_calibrated_hb_cal_15_interactive, p_calibrated_hb_cal_50_interactive, p_calibrated_nhb_15_static, p_calibrated_nhb_50_static, p_calibrated_nhb_15_interactive, p_calibrated_nhb_50_interactive, p_calibrated_nhb_cal_15_static, p_calibrated_nhb_cal_50_static, p_calibrated_nhb_cal_15_interactive, p_calibrated_nhb_cal_50_interactive = p_calibrated

    # Write CAL Dataset
    with open(write_filename_cal, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["index", "p", "p_inverted", "label", "p_calibrated_inv_ps_cal", "p_calibrated_nps_cal", "p_calibrated_nts_cal", "p_calibrated_ir_cal", "p_calibrated_nir_cal","p_calibrated_hb_15_static_cal", "p_calibrated_hb_50_static_cal","p_calibrated_hb_15_interactive_cal", "p_calibrated_hb_50_interactive_cal", "p_calibrated_nhb_15_static_cal", "p_calibrated_nhb_50_static_cal", "p_calibrated_nhb_15_interactive_cal", "p_calibrated_nhb_50_interactive_cal"]
        writer.writerow(header)
        for index, pred in enumerate(p_calibrated_inv_ps_cal):
            row = [index, preds_cal[index], preds_cal_inverted[index], labels_cal[index], p_calibrated_inv_ps_cal[index], p_calibrated_nps_cal[index], p_calibrated_nts_cal[index], p_calibrated_ir_cal[index], p_calibrated_nir_cal[index], p_calibrated_hb_cal_15_static[index], p_calibrated_hb_cal_50_static[index],p_calibrated_hb_cal_15_interactive[index], p_calibrated_hb_cal_50_interactive[index], p_calibrated_nhb_cal_15_static[index], p_calibrated_nhb_cal_50_static[index], p_calibrated_nhb_cal_15_interactive[index], p_calibrated_nhb_cal_50_interactive[index]]
            writer.writerow(row)

    # Write TEST Dataset
    with open(write_filename_test, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["index", "p", "p_inverted", "label", "p_calibrated_inv_ps", "p_calibrated_nps", "p_calibrated_nts", "p_calibrated_ir", "p_calibrated_nir","p_calibrated_hb_15_static", "p_calibrated_hb_50_static","p_calibrated_hb_15_interactive", "p_calibrated_hb_50_interactive", "p_calibrated_nhb_15_static", "p_calibrated_nhb_50_static", "p_calibrated_nhb_15_interactive", "p_calibrated_nhb_50_interactive"]
        writer.writerow(header)
        for index, pred in enumerate(p_calibrated_inv_ps):
            row = [index, preds_test[index], preds_test_inverted[index], labels_test[index], p_calibrated_inv_ps[index], p_calibrated_nps[index], p_calibrated_nts[index], p_calibrated_ir[index], p_calibrated_nir[index],p_calibrated_hb_15_static[index], p_calibrated_hb_50_static[index],p_calibrated_hb_15_interactive[index], p_calibrated_hb_50_interactive[index], p_calibrated_nhb_15_static[index], p_calibrated_nhb_50_static[index], p_calibrated_nhb_15_interactive[index], p_calibrated_nhb_50_interactive[index]]
            writer.writerow(row)





def calculate_calibrated_metrics(preds_test, labels_test, preds_cal, labels_cal, p_calibrated, rel_dias):
    # Extract p_calibrated and rel_dias
    p_calibrated_inv_ps,p_calibrated_inv_ps_cal, p_calibrated_nps, p_calibrated_nps_cal, p_calibrated_nts, p_calibrated_nts_cal, p_calibrated_ir, p_calibrated_ir_cal, p_calibrated_nir, p_calibrated_nir_cal, p_calibrated_hb_15_static, p_calibrated_hb_50_static, p_calibrated_hb_15_interactive, p_calibrated_hb_50_interactive, p_calibrated_hb_cal_15_static, p_calibrated_hb_cal_50_static, p_calibrated_hb_cal_15_interactive, p_calibrated_hb_cal_50_interactive, p_calibrated_nhb_15_static, p_calibrated_nhb_50_static, p_calibrated_nhb_15_interactive, p_calibrated_nhb_50_interactive, p_calibrated_nhb_cal_15_static, p_calibrated_nhb_cal_50_static, p_calibrated_nhb_cal_15_interactive, p_calibrated_nhb_cal_50_interactive = p_calibrated

    rel_diagram_bin_15_test_uncalibrated_test_static, rel_diagram_bin_50_test_uncalibrated_test_static, rel_diagram_bin_15_test_uncalibrated_test_interactive, rel_diagram_bin_50_test_uncalibrated_test_interactive, rel_diagram_bin_15_test_uncalibrated_cal_static, rel_diagram_bin_50_test_uncalibrated_cal_static, rel_diagram_bin_15_test_uncalibrated_cal_interactive, rel_diagram_bin_50_test_uncalibrated_cal_interactive, rel_diagram_bin_15_test_ps_static_test, rel_diagram_bin_50_test_ps_static_test, rel_diagram_bin_15_test_ps_interactive_test, rel_diagram_bin_50_test_ps_interactive_test, rel_diagram_bin_15_test_ps_static_cal, rel_diagram_bin_50_test_ps_static_cal, rel_diagram_bin_15_test_ps_interactive_cal, rel_diagram_bin_50_test_ps_interactive_cal, rel_diagram_bin_15_test_nps_static_test, rel_diagram_bin_50_test_nps_static_test, rel_diagram_bin_15_test_nps_interactive_test, rel_diagram_bin_50_test_nps_interactive_test, rel_diagram_bin_15_test_nps_static_cal, rel_diagram_bin_50_test_nps_static_cal, rel_diagram_bin_15_test_nps_interactive_cal, rel_diagram_bin_50_test_nps_interactive_cal, rel_diagram_bin_15_test_nts_static_test, rel_diagram_bin_50_test_nts_static_test, rel_diagram_bin_15_test_nts_interactive_test, rel_diagram_bin_50_test_nts_interactive_test, rel_diagram_bin_15_test_nts_static_cal, rel_diagram_bin_50_test_nts_static_cal, rel_diagram_bin_15_test_nts_interactive_cal, rel_diagram_bin_50_test_nts_interactive_cal, rel_diagram_bin_15_test_ir_static_test, rel_diagram_bin_50_test_ir_static_test, rel_diagram_bin_15_test_ir_interactive_test, rel_diagram_bin_50_test_ir_interactive_test, rel_diagram_bin_15_test_ir_static_cal, rel_diagram_bin_50_test_ir_static_cal, rel_diagram_bin_15_test_ir_interactive_cal, rel_diagram_bin_50_test_ir_interactive_cal, rel_diagram_bin_15_test_nir_static_test, rel_diagram_bin_50_test_nir_static_test, rel_diagram_bin_15_test_nir_interactive_test, rel_diagram_bin_50_test_nir_interactive_test, rel_diagram_bin_15_test_nir_static_cal, rel_diagram_bin_50_test_nir_static_cal, rel_diagram_bin_15_test_nir_interactive_cal, rel_diagram_bin_50_test_nir_interactive_cal, rel_diagram_bin_15_test_hb_static_test, rel_diagram_bin_15_test_hb_interactive_test, rel_diagram_bin_15_test_hb_static_cal, rel_diagram_bin_15_test_hb_interactive_cal, rel_diagram_bin_15_test_nhb_static_test, rel_diagram_bin_15_test_nhb_interactive_test, rel_diagram_bin_15_test_nhb_static_cal, rel_diagram_bin_15_test_nhb_interactive_cal, rel_diagram_bin_50_test_hb_static_test, rel_diagram_bin_50_test_hb_interactive_test, rel_diagram_bin_50_test_hb_static_cal, rel_diagram_bin_50_test_hb_interactive_cal, rel_diagram_bin_50_test_nhb_static_test, rel_diagram_bin_50_test_nhb_interactive_test, rel_diagram_bin_50_test_nhb_static_cal, rel_diagram_bin_50_test_nhb_interactive_cal, rel_diagram_bin_15_test_hb_static_test_interactive_calibration, rel_diagram_bin_15_test_hb_interactive_test_interactive_calibration, rel_diagram_bin_15_test_hb_static_cal_interactive_calibration, rel_diagram_bin_15_test_hb_interactive_cal_interactive_calibration, rel_diagram_bin_15_test_nhb_static_test_interactive_calibration, rel_diagram_bin_15_test_nhb_interactive_test_interactive_calibration, rel_diagram_bin_15_test_nhb_static_cal_interactive_calibration, rel_diagram_bin_15_test_nhb_interactive_cal_interactive_calibration, rel_diagram_bin_50_test_hb_static_test_interactive_calibration, rel_diagram_bin_50_test_hb_interactive_test_interactive_calibration, rel_diagram_bin_50_test_hb_static_cal_interactive_calibration, rel_diagram_bin_50_test_hb_interactive_cal_interactive_calibration, rel_diagram_bin_50_test_nhb_static_test_interactive_calibration, rel_diagram_bin_50_test_nhb_interactive_test_interactive_calibration, rel_diagram_bin_50_test_nhb_static_cal_interactive_calibration, rel_diagram_bin_50_test_nhb_interactive_cal_interactive_calibration = rel_dias


    rel_dias_names = ("rel_diagram_bin_15_test_uncalibrated_test_static", "rel_diagram_bin_50_test_uncalibrated_test_static", "rel_diagram_bin_15_test_uncalibrated_test_interactive", "rel_diagram_bin_50_test_uncalibrated_test_interactive", "rel_diagram_bin_15_test_uncalibrated_cal_static", "rel_diagram_bin_50_test_uncalibrated_cal_static", "rel_diagram_bin_15_test_uncalibrated_cal_interactive", "rel_diagram_bin_50_test_uncalibrated_cal_interactive", "rel_diagram_bin_15_test_ps_static_test", "rel_diagram_bin_50_test_ps_static_test", "rel_diagram_bin_15_test_ps_interactive_test", "rel_diagram_bin_50_test_ps_interactive_test", "rel_diagram_bin_15_test_ps_static_cal", "rel_diagram_bin_50_test_ps_static_cal", "rel_diagram_bin_15_test_ps_interactive_cal", "rel_diagram_bin_50_test_ps_interactive_cal", "rel_diagram_bin_15_test_nps_static_test", "rel_diagram_bin_50_test_nps_static_test", "rel_diagram_bin_15_test_nps_interactive_test", "rel_diagram_bin_50_test_nps_interactive_test", "rel_diagram_bin_15_test_nps_static_cal", "rel_diagram_bin_50_test_nps_static_cal", "rel_diagram_bin_15_test_nps_interactive_cal", "rel_diagram_bin_50_test_nps_interactive_cal", "rel_diagram_bin_15_test_nts_static_test", "rel_diagram_bin_50_test_nts_static_test", "rel_diagram_bin_15_test_nts_interactive_test", "rel_diagram_bin_50_test_nts_interactive_test", "rel_diagram_bin_15_test_nts_static_cal", "rel_diagram_bin_50_test_nts_static_cal", "rel_diagram_bin_15_test_nts_interactive_cal", "rel_diagram_bin_50_test_nts_interactive_cal", "rel_diagram_bin_15_test_ir_static_test", "rel_diagram_bin_50_test_ir_static_test", "rel_diagram_bin_15_test_ir_interactive_test", "rel_diagram_bin_50_test_ir_interactive_test", "rel_diagram_bin_15_test_ir_static_cal", "rel_diagram_bin_50_test_ir_static_cal", "rel_diagram_bin_15_test_ir_interactive_cal", "rel_diagram_bin_50_test_ir_interactive_cal", "rel_diagram_bin_15_test_nir_static_test", "rel_diagram_bin_50_test_nir_static_test", "rel_diagram_bin_15_test_nir_interactive_test", "rel_diagram_bin_50_test_nir_interactive_test", "rel_diagram_bin_15_test_nir_static_cal", "rel_diagram_bin_50_test_nir_static_cal", "rel_diagram_bin_15_test_nir_interactive_cal", "rel_diagram_bin_50_test_nir_interactive_cal", "rel_diagram_bin_15_test_hb_static_test", "rel_diagram_bin_15_test_hb_interactive_test", "rel_diagram_bin_15_test_hb_static_cal", "rel_diagram_bin_15_test_hb_interactive_cal", "rel_diagram_bin_15_test_nhb_static_test", "rel_diagram_bin_15_test_nhb_interactive_test", "rel_diagram_bin_15_test_nhb_static_cal", "rel_diagram_bin_15_test_nhb_interactive_cal", "rel_diagram_bin_50_test_hb_static_test", "rel_diagram_bin_50_test_hb_interactive_test", "rel_diagram_bin_50_test_hb_static_cal", "rel_diagram_bin_50_test_hb_interactive_cal", "rel_diagram_bin_50_test_nhb_static_test", "rel_diagram_bin_50_test_nhb_interactive_test", "rel_diagram_bin_50_test_nhb_static_cal", "rel_diagram_bin_50_test_nhb_interactive_cal", "rel_diagram_bin_15_test_hb_static_test_interactive_calibration", "rel_diagram_bin_15_test_hb_interactive_test_interactive_calibration", "rel_diagram_bin_15_test_hb_static_cal_interactive_calibration", "rel_diagram_bin_15_test_hb_interactive_cal_interactive_calibration", "rel_diagram_bin_15_test_nhb_static_test_interactive_calibration", "rel_diagram_bin_15_test_nhb_interactive_test_interactive_calibration", "rel_diagram_bin_15_test_nhb_static_cal_interactive_calibration", "rel_diagram_bin_15_test_nhb_interactive_cal_interactive_calibration", "rel_diagram_bin_50_test_hb_static_test_interactive_calibration", "rel_diagram_bin_50_test_hb_interactive_test_interactive_calibration", "rel_diagram_bin_50_test_hb_static_cal_interactive_calibration", "rel_diagram_bin_50_test_hb_interactive_cal_interactive_calibration", "rel_diagram_bin_50_test_nhb_static_test_interactive_calibration", "rel_diagram_bin_50_test_nhb_interactive_test_interactive_calibration", "rel_diagram_bin_50_test_nhb_static_cal_interactive_calibration", "rel_diagram_bin_50_test_nhb_interactive_cal_interactive_calibration")

    # netcal ECE
    ece = ECE(15)
    ece_score_uncalibrated = ece.measure(preds_test, labels_test)
    ece_score_uncalibrated_cal = ece.measure(preds_cal, labels_cal)

    ece_score_calibrated_nps = ece.measure(p_calibrated_nps, labels_test)
    ece_score_calibrated_nps_cal = ece.measure(p_calibrated_nps_cal, labels_cal)
    ece_score_calibrated_inv_ps = ece.measure(p_calibrated_inv_ps, labels_test)
    ece_score_calibrated_inv_ps_cal = ece.measure(p_calibrated_inv_ps_cal, labels_cal)

    ece_score_calibrated_nts = ece.measure(p_calibrated_nts, labels_test)
    ece_score_calibrated_nts_cal = ece.measure(p_calibrated_nts_cal, labels_cal)

    ece_score_calibrated_ir = ece.measure(p_calibrated_ir, labels_test)
    ece_score_calibrated_ir_cal = ece.measure(p_calibrated_ir_cal, labels_cal)
    ece_score_calibrated_nir = ece.measure(p_calibrated_nir, labels_test)
    ece_score_calibrated_nir_cal = ece.measure(p_calibrated_nir_cal, labels_cal)

    ece_score_calibrated_hb_15_static = ece.measure(p_calibrated_hb_15_static, labels_test)
    ece_score_calibrated_hb_50_static = ece.measure(p_calibrated_hb_50_static, labels_test)
    ece_score_calibrated_hb_15_interactive = ece.measure(p_calibrated_hb_15_interactive, labels_test)
    ece_score_calibrated_hb_50_interactive = ece.measure(p_calibrated_hb_50_interactive, labels_test)
    ece_score_calibrated_hb_cal_15_static = ece.measure(p_calibrated_hb_cal_15_static, labels_cal)
    ece_score_calibrated_hb_cal_50_static = ece.measure(p_calibrated_hb_cal_50_static, labels_cal)
    ece_score_calibrated_hb_cal_15_interactive = ece.measure(p_calibrated_hb_cal_15_interactive, labels_cal)
    ece_score_calibrated_hb_cal_50_interactive = ece.measure(p_calibrated_hb_cal_50_interactive, labels_cal)
    ece_score_calibrated_nhb_15_static = ece.measure(p_calibrated_nhb_15_static, labels_test)
    ece_score_calibrated_nhb_50_static = ece.measure(p_calibrated_nhb_50_static, labels_test)
    ece_score_calibrated_nhb_15_static = ece.measure(p_calibrated_nhb_15_interactive, labels_test)
    ece_score_calibrated_nhb_50_static = ece.measure(p_calibrated_nhb_50_interactive, labels_test)
    ece_score_calibrated_nhb_cal_15_static = ece.measure(p_calibrated_nhb_cal_15_static, labels_cal)
    ece_score_calibrated_nhb_cal_50_static = ece.measure(p_calibrated_nhb_cal_50_static, labels_cal)
    ece_score_calibrated_nhb_cal_15_interactive = ece.measure(p_calibrated_nhb_cal_15_interactive, labels_cal)
    ece_score_calibrated_nhb_cal_50_interactive = ece.measure(p_calibrated_nhb_cal_50_interactive, labels_cal)


    ece_50 = ECE(50)
    ece_50_score_uncalibrated = ece_50.measure(preds_test, labels_test)
    ece_50_score_uncalibrated_cal = ece_50.measure(preds_cal, labels_cal)

    ece_50_score_calibrated_nps = ece_50.measure(p_calibrated_nps, labels_test)
    ece_50_score_calibrated_nps_cal = ece_50.measure(p_calibrated_nps_cal, labels_cal)
    ece_50_score_calibrated_inv_ps = ece_50.measure(p_calibrated_inv_ps, labels_test)
    ece_50_score_calibrated_inv_ps_cal = ece_50.measure(p_calibrated_inv_ps_cal, labels_cal)

    ece_50_score_calibrated_nts = ece_50.measure(p_calibrated_nts, labels_test)
    ece_50_score_calibrated_nts_cal = ece_50.measure(p_calibrated_nts_cal, labels_cal)

    ece_50_score_calibrated_ir = ece_50.measure(p_calibrated_ir, labels_test)
    ece_50_score_calibrated_ir_cal = ece_50.measure(p_calibrated_ir_cal, labels_cal)
    ece_50_score_calibrated_nir = ece_50.measure(p_calibrated_nir, labels_test)
    ece_50_score_calibrated_nir_cal = ece_50.measure(p_calibrated_nir_cal, labels_cal)

    ece_50_score_calibrated_hb_15_static = ece_50.measure(p_calibrated_hb_15_static, labels_test)
    ece_50_score_calibrated_hb_50_static = ece_50.measure(p_calibrated_hb_50_static, labels_test)
    ece_50_score_calibrated_hb_15_interactive = ece_50.measure(p_calibrated_hb_15_interactive, labels_test)
    ece_50_score_calibrated_hb_50_interactive = ece_50.measure(p_calibrated_hb_50_interactive, labels_test)
    ece_50_score_calibrated_hb_cal_15_static = ece_50.measure(p_calibrated_hb_cal_15_static, labels_cal)
    ece_50_score_calibrated_hb_cal_50_static = ece_50.measure(p_calibrated_hb_cal_50_static, labels_cal)
    ece_50_score_calibrated_hb_cal_15_interactive = ece_50.measure(p_calibrated_hb_cal_15_interactive, labels_cal)
    ece_50_score_calibrated_hb_cal_50_interactive = ece_50.measure(p_calibrated_hb_cal_50_interactive, labels_cal)
    ece_50_score_calibrated_nhb_15_static = ece_50.measure(p_calibrated_nhb_15_static, labels_test)
    ece_50_score_calibrated_nhb_50_static = ece_50.measure(p_calibrated_nhb_50_static, labels_test)
    ece_50_score_calibrated_nhb_15_static = ece_50.measure(p_calibrated_nhb_15_interactive, labels_test)
    ece_50_score_calibrated_nhb_50_static = ece_50.measure(p_calibrated_nhb_50_interactive, labels_test)
    ece_50_score_calibrated_nhb_cal_15_static = ece_50.measure(p_calibrated_nhb_cal_15_static, labels_cal)
    ece_50_score_calibrated_nhb_cal_50_static = ece_50.measure(p_calibrated_nhb_cal_50_static, labels_cal)
    ece_50_score_calibrated_nhb_cal_15_interactive = ece_50.measure(p_calibrated_nhb_cal_15_interactive, labels_cal)
    ece_50_score_calibrated_nhb_cal_50_interactive = ece_50.measure(p_calibrated_nhb_cal_50_interactive, labels_cal)

    # AUC score
    auc_score_uncalibrated = roc_auc_score(labels_test, preds_test)
    auc_score_uncalibrated_cal = roc_auc_score(labels_cal, preds_cal)

    auc_score_calibrated_nps = roc_auc_score(labels_test, p_calibrated_nps)
    auc_score_calibrated_nps_cal = roc_auc_score(labels_cal, p_calibrated_nps_cal)
    auc_score_calibrated_inv_ps = roc_auc_score(labels_test, p_calibrated_inv_ps)
    auc_score_calibrated_inv_ps_cal = roc_auc_score(labels_cal, p_calibrated_inv_ps_cal)

    auc_score_calibrated_nts = roc_auc_score(labels_test, p_calibrated_nts)
    auc_score_calibrated_nts_cal = roc_auc_score(labels_cal, p_calibrated_nts_cal)

    auc_score_calibrated_ir = roc_auc_score(labels_test, p_calibrated_ir)
    auc_score_calibrated_ir_cal = roc_auc_score(labels_cal, p_calibrated_ir_cal)
    auc_score_calibrated_nir = roc_auc_score(labels_test, p_calibrated_nir)
    auc_score_calibrated_nir_cal = roc_auc_score(labels_cal, p_calibrated_nir_cal)

    auc_score_calibrated_hb_15_static = roc_auc_score(labels_test, p_calibrated_hb_15_static)
    auc_score_calibrated_hb_50_static = roc_auc_score(labels_test, p_calibrated_hb_50_static)
    auc_score_calibrated_hb_15_interactive = roc_auc_score(labels_test, p_calibrated_hb_15_interactive)
    auc_score_calibrated_hb_50_interactive = roc_auc_score(labels_test, p_calibrated_hb_50_interactive)
    auc_score_calibrated_hb_cal_15_static = roc_auc_score(labels_cal, p_calibrated_hb_cal_15_static)
    auc_score_calibrated_hb_cal_50_static = roc_auc_score(labels_cal, p_calibrated_hb_cal_50_static)
    auc_score_calibrated_hb_cal_15_interactive = roc_auc_score(labels_cal, p_calibrated_hb_cal_15_interactive)
    auc_score_calibrated_hb_cal_50_interactive = roc_auc_score(labels_cal, p_calibrated_hb_cal_50_interactive)
    auc_score_calibrated_nhb_15_static = roc_auc_score(labels_test, p_calibrated_nhb_15_static)
    auc_score_calibrated_nhb_50_static = roc_auc_score(labels_test, p_calibrated_nhb_50_static)
    auc_score_calibrated_nhb_15_static = roc_auc_score(labels_test, p_calibrated_nhb_15_interactive)
    auc_score_calibrated_nhb_50_static = roc_auc_score(labels_test, p_calibrated_nhb_50_interactive)
    auc_score_calibrated_nhb_cal_15_static = roc_auc_score(labels_cal, p_calibrated_nhb_cal_15_static)
    auc_score_calibrated_nhb_cal_50_static = roc_auc_score(labels_cal, p_calibrated_nhb_cal_50_static)
    auc_score_calibrated_nhb_cal_15_interactive = roc_auc_score(labels_cal, p_calibrated_nhb_cal_15_interactive)
    auc_score_calibrated_nhb_cal_50_interactive = roc_auc_score(labels_cal, p_calibrated_nhb_cal_50_interactive)

    # Log Loss
    log_loss_uncalibrated = log_loss(labels_test, preds_test)
    log_loss_uncalibrated_cal = log_loss(labels_cal, preds_cal)
    log_loss_calibrated_inv_ps = log_loss(labels_test, p_calibrated_inv_ps)
    log_loss_calibrated_inv_ps_cal = log_loss(labels_cal, p_calibrated_inv_ps_cal)
    log_loss_calibrated_nps = log_loss(labels_test, p_calibrated_nps)
    log_loss_calibrated_nps_cal = log_loss(labels_cal, p_calibrated_nps_cal)
    
    log_loss_calibrated_nts = log_loss(labels_test, p_calibrated_nts)
    log_loss_calibrated_nts_cal = log_loss(labels_cal, p_calibrated_nts_cal)

    log_loss_calibrated_ir = log_loss(labels_test, p_calibrated_ir)
    log_loss_calibrated_ir_cal = log_loss(labels_cal, p_calibrated_ir_cal)
    log_loss_calibrated_nir = log_loss(labels_test, p_calibrated_nir)
    log_loss_calibrated_nir_cal = log_loss(labels_cal, p_calibrated_nir_cal)

    log_loss_calibrated_hb_15_static = log_loss(labels_test, p_calibrated_hb_15_static)
    log_loss_calibrated_hb_50_static = log_loss(labels_test, p_calibrated_hb_50_static)
    log_loss_calibrated_hb_15_interactive = log_loss(labels_test, p_calibrated_hb_15_interactive)
    log_loss_calibrated_hb_50_interactive = log_loss(labels_test, p_calibrated_hb_50_interactive)
    log_loss_calibrated_hb_cal_15_static = log_loss(labels_cal, p_calibrated_hb_cal_15_static)
    log_loss_calibrated_hb_cal_50_static = log_loss(labels_cal, p_calibrated_hb_cal_50_static)
    log_loss_calibrated_hb_cal_15_interactive = log_loss(labels_cal, p_calibrated_hb_cal_15_interactive)
    log_loss_calibrated_hb_cal_50_interactive = log_loss(labels_cal, p_calibrated_hb_cal_50_interactive)
    log_loss_calibrated_nhb_15_static = log_loss(labels_test, p_calibrated_nhb_15_static)
    log_loss_calibrated_nhb_50_static = log_loss(labels_test, p_calibrated_nhb_50_static)
    log_loss_calibrated_nhb_15_static = log_loss(labels_test, p_calibrated_nhb_15_interactive)
    log_loss_calibrated_nhb_50_static = log_loss(labels_test, p_calibrated_nhb_50_interactive)
    log_loss_calibrated_nhb_cal_15_static = log_loss(labels_cal, p_calibrated_nhb_cal_15_static)
    log_loss_calibrated_nhb_cal_50_static = log_loss(labels_cal, p_calibrated_nhb_cal_50_static)
    log_loss_calibrated_nhb_cal_15_interactive = log_loss(labels_cal, p_calibrated_nhb_cal_15_interactive)
    log_loss_calibrated_nhb_cal_50_interactive = log_loss(labels_cal, p_calibrated_nhb_cal_50_interactive)

    brier_score_uncalibrated = calculate_brier_score(labels_test, preds_test)
    brier_score_uncalibrated_cal = calculate_brier_score(labels_cal, preds_cal)
    brier_score_calibrated_inv_ps = calculate_brier_score(labels_test, p_calibrated_inv_ps)
    brier_score_calibrated_inv_ps_cal = calculate_brier_score(labels_cal, p_calibrated_inv_ps_cal)
    brier_score_calibrated_nps = calculate_brier_score(labels_test, p_calibrated_nps)
    brier_score_calibrated_nps_cal = calculate_brier_score(labels_cal, p_calibrated_nps_cal)
    
    brier_score_calibrated_nts = calculate_brier_score(labels_test, p_calibrated_nts)
    brier_score_calibrated_nts_cal = calculate_brier_score(labels_cal, p_calibrated_nts_cal)

    brier_score_calibrated_ir = calculate_brier_score(labels_test, p_calibrated_ir)
    brier_score_calibrated_ir_cal = calculate_brier_score(labels_cal, p_calibrated_ir_cal)
    brier_score_calibrated_nir = calculate_brier_score(labels_test, p_calibrated_nir)
    brier_score_calibrated_nir_cal = calculate_brier_score(labels_cal, p_calibrated_nir_cal)

    brier_score_calibrated_hb_15_static = calculate_brier_score(labels_test, p_calibrated_hb_15_static)
    brier_score_calibrated_hb_50_static = calculate_brier_score(labels_test, p_calibrated_hb_50_static)
    brier_score_calibrated_hb_15_interactive = calculate_brier_score(labels_test, p_calibrated_hb_15_interactive)
    brier_score_calibrated_hb_50_interactive = calculate_brier_score(labels_test, p_calibrated_hb_50_interactive)
    brier_score_calibrated_hb_cal_15_static = calculate_brier_score(labels_cal, p_calibrated_hb_cal_15_static)
    brier_score_calibrated_hb_cal_50_static = calculate_brier_score(labels_cal, p_calibrated_hb_cal_50_static)
    brier_score_calibrated_hb_cal_15_interactive = calculate_brier_score(labels_cal, p_calibrated_hb_cal_15_interactive)
    brier_score_calibrated_hb_cal_50_interactive = calculate_brier_score(labels_cal, p_calibrated_hb_cal_50_interactive)
    brier_score_calibrated_nhb_15_static = calculate_brier_score(labels_test, p_calibrated_nhb_15_static)
    brier_score_calibrated_nhb_50_static = calculate_brier_score(labels_test, p_calibrated_nhb_50_static)
    brier_score_calibrated_nhb_15_static = calculate_brier_score(labels_test, p_calibrated_nhb_15_interactive)
    brier_score_calibrated_nhb_50_static = calculate_brier_score(labels_test, p_calibrated_nhb_50_interactive)
    brier_score_calibrated_nhb_cal_15_static = calculate_brier_score(labels_cal, p_calibrated_nhb_cal_15_static)
    brier_score_calibrated_nhb_cal_50_static = calculate_brier_score(labels_cal, p_calibrated_nhb_cal_50_static)
    brier_score_calibrated_nhb_cal_15_interactive = calculate_brier_score(labels_cal, p_calibrated_nhb_cal_15_interactive)
    brier_score_calibrated_nhb_cal_50_interactive = calculate_brier_score(labels_cal, p_calibrated_nhb_cal_50_interactive)

    # zipping
    metrics = (ece_score_uncalibrated, ece_score_uncalibrated_cal, ece_score_calibrated_nps, ece_score_calibrated_nps_cal, ece_score_calibrated_inv_ps, ece_score_calibrated_inv_ps_cal, ece_score_calibrated_nts, ece_score_calibrated_nts_cal, ece_score_calibrated_ir, ece_score_calibrated_ir_cal, ece_score_calibrated_nir, ece_score_calibrated_nir_cal, ece_score_calibrated_hb_15_static, ece_score_calibrated_hb_50_static, ece_score_calibrated_hb_15_interactive, ece_score_calibrated_hb_50_interactive, ece_score_calibrated_hb_cal_15_static, ece_score_calibrated_hb_cal_50_static, ece_score_calibrated_hb_cal_15_interactive, ece_score_calibrated_hb_cal_50_interactive, ece_score_calibrated_nhb_15_static, ece_score_calibrated_nhb_50_static, ece_score_calibrated_nhb_15_static, ece_score_calibrated_nhb_50_static, ece_score_calibrated_nhb_cal_15_static, ece_score_calibrated_nhb_cal_50_static, ece_score_calibrated_nhb_cal_15_interactive, ece_score_calibrated_nhb_cal_50_interactive, ece_50_score_uncalibrated, ece_50_score_uncalibrated_cal, ece_50_score_calibrated_nps, ece_50_score_calibrated_nps_cal, ece_50_score_calibrated_inv_ps, ece_50_score_calibrated_inv_ps_cal, ece_50_score_calibrated_nts, ece_50_score_calibrated_nts_cal, ece_50_score_calibrated_ir, ece_50_score_calibrated_ir_cal, ece_50_score_calibrated_nir, ece_50_score_calibrated_nir_cal, ece_50_score_calibrated_hb_15_static, ece_50_score_calibrated_hb_50_static, ece_50_score_calibrated_hb_15_interactive, ece_50_score_calibrated_hb_50_interactive, ece_50_score_calibrated_hb_cal_15_static, ece_50_score_calibrated_hb_cal_50_static, ece_50_score_calibrated_hb_cal_15_interactive, ece_50_score_calibrated_hb_cal_50_interactive, ece_50_score_calibrated_nhb_15_static, ece_50_score_calibrated_nhb_50_static, ece_50_score_calibrated_nhb_15_static, ece_50_score_calibrated_nhb_50_static, ece_50_score_calibrated_nhb_cal_15_static, ece_50_score_calibrated_nhb_cal_50_static, ece_50_score_calibrated_nhb_cal_15_interactive, ece_50_score_calibrated_nhb_cal_50_interactive, auc_score_uncalibrated, auc_score_uncalibrated_cal, auc_score_calibrated_nps, auc_score_calibrated_nps_cal, auc_score_calibrated_inv_ps, auc_score_calibrated_inv_ps_cal, auc_score_calibrated_nts, auc_score_calibrated_nts_cal, auc_score_calibrated_ir, auc_score_calibrated_ir_cal, auc_score_calibrated_nir, auc_score_calibrated_nir_cal, auc_score_calibrated_hb_15_static, auc_score_calibrated_hb_50_static, auc_score_calibrated_hb_15_interactive, auc_score_calibrated_hb_50_interactive, auc_score_calibrated_hb_cal_15_static, auc_score_calibrated_hb_cal_50_static, auc_score_calibrated_hb_cal_15_interactive, auc_score_calibrated_hb_cal_50_interactive, auc_score_calibrated_nhb_15_static, auc_score_calibrated_nhb_50_static, auc_score_calibrated_nhb_15_static, auc_score_calibrated_nhb_50_static, auc_score_calibrated_nhb_cal_15_static, auc_score_calibrated_nhb_cal_50_static, auc_score_calibrated_nhb_cal_15_interactive, auc_score_calibrated_nhb_cal_50_interactive, log_loss_uncalibrated, log_loss_uncalibrated_cal, log_loss_calibrated_inv_ps, log_loss_calibrated_inv_ps_cal, log_loss_calibrated_nps, log_loss_calibrated_nps_cal, log_loss_calibrated_nts, log_loss_calibrated_nts_cal, log_loss_calibrated_ir, log_loss_calibrated_ir_cal, log_loss_calibrated_nir, log_loss_calibrated_nir_cal, log_loss_calibrated_hb_15_static, log_loss_calibrated_hb_50_static, log_loss_calibrated_hb_15_interactive, log_loss_calibrated_hb_50_interactive, log_loss_calibrated_hb_cal_15_static, log_loss_calibrated_hb_cal_50_static, log_loss_calibrated_hb_cal_15_interactive, log_loss_calibrated_hb_cal_50_interactive, log_loss_calibrated_nhb_15_static, log_loss_calibrated_nhb_50_static, log_loss_calibrated_nhb_15_static, log_loss_calibrated_nhb_50_static, log_loss_calibrated_nhb_cal_15_static, log_loss_calibrated_nhb_cal_50_static, log_loss_calibrated_nhb_cal_15_interactive, log_loss_calibrated_nhb_cal_50_interactive, brier_score_uncalibrated, brier_score_uncalibrated_cal, brier_score_calibrated_inv_ps, brier_score_calibrated_inv_ps_cal, brier_score_calibrated_nps, brier_score_calibrated_nps_cal, brier_score_calibrated_nts, brier_score_calibrated_nts_cal, brier_score_calibrated_ir, brier_score_calibrated_ir_cal, brier_score_calibrated_nir, brier_score_calibrated_nir_cal, brier_score_calibrated_hb_15_static, brier_score_calibrated_hb_50_static, brier_score_calibrated_hb_15_interactive, brier_score_calibrated_hb_50_interactive, brier_score_calibrated_hb_cal_15_static, brier_score_calibrated_hb_cal_50_static, brier_score_calibrated_hb_cal_15_interactive, brier_score_calibrated_hb_cal_50_interactive, brier_score_calibrated_nhb_15_static, brier_score_calibrated_nhb_50_static, brier_score_calibrated_nhb_15_static, brier_score_calibrated_nhb_50_static, brier_score_calibrated_nhb_cal_15_static, brier_score_calibrated_nhb_cal_50_static, brier_score_calibrated_nhb_cal_15_interactive, brier_score_calibrated_nhb_cal_50_interactive)

    metrics_names = ["ece_score_uncalibrated", "ece_score_uncalibrated_cal", "ece_score_calibrated_nps", "ece_score_calibrated_nps_cal", "ece_score_calibrated_inv_ps", "ece_score_calibrated_inv_ps_cal", "ece_score_calibrated_nts", "ece_score_calibrated_nts_cal", "ece_score_calibrated_ir", "ece_score_calibrated_ir_cal", "ece_score_calibrated_nir", "ece_score_calibrated_nir_cal", "ece_score_calibrated_hb_15_static", "ece_score_calibrated_hb_50_static", "ece_score_calibrated_hb_15_interactive", "ece_score_calibrated_hb_50_interactive", "ece_score_calibrated_hb_cal_15_static", "ece_score_calibrated_hb_cal_50_static", "ece_score_calibrated_hb_cal_15_interactive", "ece_score_calibrated_hb_cal_50_interactive", "ece_score_calibrated_nhb_15_static", "ece_score_calibrated_nhb_50_static", "ece_score_calibrated_nhb_15_static", "ece_score_calibrated_nhb_50_static", "ece_score_calibrated_nhb_cal_15_static", "ece_score_calibrated_nhb_cal_50_static", "ece_score_calibrated_nhb_cal_15_interactive", "ece_score_calibrated_nhb_cal_50_interactive", "ece_50_score_uncalibrated", "ece_50_score_uncalibrated_cal", "ece_50_score_calibrated_nps", "ece_50_score_calibrated_nps_cal", "ece_50_score_calibrated_inv_ps", "ece_50_score_calibrated_inv_ps_cal", "ece_50_score_calibrated_nts", "ece_50_score_calibrated_nts_cal", "ece_50_score_calibrated_ir", "ece_50_score_calibrated_ir_cal", "ece_50_score_calibrated_nir", "ece_50_score_calibrated_nir_cal", "ece_50_score_calibrated_hb_15_static", "ece_50_score_calibrated_hb_50_static", "ece_50_score_calibrated_hb_15_interactive", "ece_50_score_calibrated_hb_50_interactive", "ece_50_score_calibrated_hb_cal_15_static", "ece_50_score_calibrated_hb_cal_50_static", "ece_50_score_calibrated_hb_cal_15_interactive", "ece_50_score_calibrated_hb_cal_50_interactive", "ece_50_score_calibrated_nhb_15_static", "ece_50_score_calibrated_nhb_50_static", "ece_50_score_calibrated_nhb_15_static", "ece_50_score_calibrated_nhb_50_static", "ece_50_score_calibrated_nhb_cal_15_static", "ece_50_score_calibrated_nhb_cal_50_static", "ece_50_score_calibrated_nhb_cal_15_interactive", "ece_50_score_calibrated_nhb_cal_50_interactive", "auc_score_uncalibrated", "auc_score_uncalibrated_cal", "auc_score_calibrated_nps", "auc_score_calibrated_nps_cal", "auc_score_calibrated_inv_ps", "auc_score_calibrated_inv_ps_cal", "auc_score_calibrated_nts", "auc_score_calibrated_nts_cal", "auc_score_calibrated_ir", "auc_score_calibrated_ir_cal", "auc_score_calibrated_nir", "auc_score_calibrated_nir_cal", "auc_score_calibrated_hb_15_static", "auc_score_calibrated_hb_50_static", "auc_score_calibrated_hb_15_interactive", "auc_score_calibrated_hb_50_interactive", "auc_score_calibrated_hb_cal_15_static", "auc_score_calibrated_hb_cal_50_static", "auc_score_calibrated_hb_cal_15_interactive", "auc_score_calibrated_hb_cal_50_interactive", "auc_score_calibrated_nhb_15_static", "auc_score_calibrated_nhb_50_static", "auc_score_calibrated_nhb_15_static", "auc_score_calibrated_nhb_50_static", "auc_score_calibrated_nhb_cal_15_static", "auc_score_calibrated_nhb_cal_50_static", "auc_score_calibrated_nhb_cal_15_interactive", "auc_score_calibrated_nhb_cal_50_interactive", "log_loss_uncalibrated", "log_loss_uncalibrated_cal", "log_loss_calibrated_inv_ps", "log_loss_calibrated_inv_ps_cal", "log_loss_calibrated_nps", "log_loss_calibrated_nps_cal", "log_loss_calibrated_nts", "log_loss_calibrated_nts_cal", "log_loss_calibrated_ir", "log_loss_calibrated_ir_cal", "log_loss_calibrated_nir", "log_loss_calibrated_nir_cal", "log_loss_calibrated_hb_15_static", "log_loss_calibrated_hb_50_static", "log_loss_calibrated_hb_15_interactive", "log_loss_calibrated_hb_50_interactive", "log_loss_calibrated_hb_cal_15_static", "log_loss_calibrated_hb_cal_50_static", "log_loss_calibrated_hb_cal_15_interactive", "log_loss_calibrated_hb_cal_50_interactive", "log_loss_calibrated_nhb_15_static", "log_loss_calibrated_nhb_50_static", "log_loss_calibrated_nhb_15_static", "log_loss_calibrated_nhb_50_static", "log_loss_calibrated_nhb_cal_15_static", "log_loss_calibrated_nhb_cal_50_static", "log_loss_calibrated_nhb_cal_15_interactive", "log_loss_calibrated_nhb_cal_50_interactive", "brier_score_uncalibrated", "brier_score_uncalibrated_cal", "brier_score_calibrated_inv_ps", "brier_score_calibrated_inv_ps_cal", "brier_score_calibrated_nps", "brier_score_calibrated_nps_cal", "brier_score_calibrated_nts", "brier_score_calibrated_nts_cal", "brier_score_calibrated_ir", "brier_score_calibrated_ir_cal", "brier_score_calibrated_nir", "brier_score_calibrated_nir_cal", "brier_score_calibrated_hb_15_static", "brier_score_calibrated_hb_50_static", "brier_score_calibrated_hb_15_interactive", "brier_score_calibrated_hb_50_interactive", "brier_score_calibrated_hb_cal_15_static", "brier_score_calibrated_hb_cal_50_static", "brier_score_calibrated_hb_cal_15_interactive", "brier_score_calibrated_hb_cal_50_interactive", "brier_score_calibrated_nhb_15_static", "brier_score_calibrated_nhb_50_static", "brier_score_calibrated_nhb_15_static", "brier_score_calibrated_nhb_50_static", "brier_score_calibrated_nhb_cal_15_static", "brier_score_calibrated_nhb_cal_50_static", "brier_score_calibrated_nhb_cal_15_interactive", "brier_score_calibrated_nhb_cal_50_interactive"]

    custom_eces = []
    custom_eces_names = []
    custom_mces = []
    custom_mces_names = []
    for index, rel_dia in enumerate(rel_dias):
        ece_name = f"Custom_ECE_{rel_dias_names[index]}"
        custom_ece = calculate_ece(rel_dia[0], rel_dia[1])
        custom_eces.append(custom_ece)
        custom_eces_names.append(ece_name)
        mce_name = f"Custom_MCE_{rel_dias_names[index]}"
        custom_mce = calculate_mce(rel_dia[0], rel_dia[1])
        custom_mces.append(custom_mce)
        custom_mces_names.append(mce_name)

    return metrics, metrics_names, custom_eces, custom_eces_names, custom_mces, custom_mces_names



if __name__ == '__main__':
    iteration_reps = 100
    calibration_metrics = []
    for iteration in range(iteration_reps):
        print(f"Iteration {iteration}")
        preds_cal = []
        labels_cal = []
        preds_test = []
        labels_test = []
        for mode in [False, True]:
            # TODO: here after evaluation comes _qt for QT but nothing for OP
            filename = f"eval_calibration_{mode}_iteration_{iteration}_evaluation.csv"
            print(f"[Calibrate Predictions] for file {filename}")
            preds, labels, confusion_matrix_category = read_input(filename)
            if mode: 
                preds_cal = np.asarray(preds)
                labels_cal = np.asarray(labels)
                continue
            preds_test = np.asarray(preds)
            labels_test = np.asarray(labels)
        
        preds_test_inverted = np.asarray(invert_sigmoid_scores(preds_test))
        preds_cal_inverted = np.asarray(invert_sigmoid_scores(preds_cal))

        # Calculate calibrated predictions
        # Platt scaling
        p_calibrated_inv_ps, p_calibrated_inv_ps_cal = custom_platt_scaling(preds_cal_inverted, labels_cal, preds_test_inverted, labels_test)
        p_calibrated_nps, p_calibrated_nps_cal = platt_scaling(preds_cal, labels_cal, preds_test, labels_test)

        # TemperatureScaling
        p_calibrated_nts, p_calibrated_nts_cal = temperature_scaling(preds_cal, labels_cal, preds_test, labels_test)


        p_calibrated = (p_calibrated_inv_ps,p_calibrated_inv_ps_cal, p_calibrated_nps, p_calibrated_nps_cal, p_calibrated_nts, p_calibrated_nts_cal)
        
        write_calibrated_predictions(preds_test, preds_test_inverted, labels_test, preds_cal, preds_cal_inverted, labels_cal, p_calibrated, iteration=iteration)

        # rel dias
        # uncalibrated
        rel_diagram_bin_15_test_uncalibrated_test_static = calculate_metrics.evaluate_reliability_diagram(labels_test, preds_test, None, 15, iteration, 'Op_1_uncal_test_15_static', False)
        rel_diagram_bin_50_test_uncalibrated_test_static = calculate_metrics.evaluate_reliability_diagram(labels_test, preds_test, None, 50, iteration, 'Op_1_uncal_test_50_static', False)
        rel_diagram_bin_15_test_uncalibrated_test_interactive = calculate_metrics.evaluate_reliability_diagram(labels_test, preds_test, None, 15, iteration, 'Op_1_uncal_test_15_interactive', True)
        rel_diagram_bin_50_test_uncalibrated_test_interactive = calculate_metrics.evaluate_reliability_diagram(labels_test, preds_test, None, 50, iteration, 'Op_1_uncal_test_50_interactive', True)

        rel_diagram_bin_15_test_uncalibrated_cal_static = calculate_metrics.evaluate_reliability_diagram(labels_cal, preds_cal, None, 15, iteration, 'Op_1_uncal_cal_15_static', False)
        rel_diagram_bin_50_test_uncalibrated_cal_static = calculate_metrics.evaluate_reliability_diagram(labels_cal, preds_cal, None, 50, iteration, 'Op_1_uncal_cal_50_static', False)
        rel_diagram_bin_15_test_uncalibrated_cal_interactive = calculate_metrics.evaluate_reliability_diagram(labels_cal, preds_cal, None, 15, iteration, 'Op_1_uncal_cal_15_interactive', True)
        rel_diagram_bin_50_test_uncalibrated_cal_interactive = calculate_metrics.evaluate_reliability_diagram(labels_cal, preds_cal, None, 50, iteration, 'Op_1_uncal_cal_50_interactive', True)

        # Platt Scaling
        # custom platt
        rel_diagram_bin_15_test_ps_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_inv_ps, None, 15, iteration, 'Op_2_ps_test_15_static', False)
        rel_diagram_bin_50_test_ps_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_inv_ps, None, 50, iteration, 'Op_2_ps_test_50_static', False)
        rel_diagram_bin_15_test_ps_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_inv_ps, None, 15, iteration, 'Op_2_ps_test_15_interactive', True)
        rel_diagram_bin_50_test_ps_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_inv_ps, None, 50, iteration, 'Op_2_ps_test_50_interactive', True)
        rel_diagram_bin_15_test_ps_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_inv_ps_cal, None, 15, iteration, 'Op_2_ps_cal_15_static', False)
        rel_diagram_bin_50_test_ps_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_inv_ps_cal, None, 50, iteration, 'Op_2_ps_cal_50_static', False)
        rel_diagram_bin_15_test_ps_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_inv_ps_cal, None, 15, iteration, 'Op_2_ps_cal_15_interactive', True)
        rel_diagram_bin_50_test_ps_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_inv_ps_cal, None, 50, iteration, 'Op_2_ps_cal_50_interactive', True)

        # netcal platt
        rel_diagram_bin_15_test_nps_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nps, None, 15, iteration, 'Op_3_nps_test_15_static', False)
        rel_diagram_bin_50_test_nps_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nps, None, 50, iteration, 'Op_3_nps_test_50_static', False)
        rel_diagram_bin_15_test_nps_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nps, None, 15, iteration, 'Op_3_nps_test_15_interactive', True)
        rel_diagram_bin_50_test_nps_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nps, None, 50, iteration, 'Op_3_nps_test_50_interactive', True)
        rel_diagram_bin_15_test_nps_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nps_cal, None, 15, iteration, 'Op_3_nps_cal_15_static', False)
        rel_diagram_bin_50_test_nps_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nps_cal, None, 50, iteration, 'Op_3_nps_cal_50_static', False)
        rel_diagram_bin_15_test_nps_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nps_cal, None, 15, iteration, 'Op_3_nps_cal_15_interactive', True)
        rel_diagram_bin_50_test_nps_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nps_cal, None, 50, iteration, 'Op_3_nps_cal_50_interactive', True)

        # Temperature Scaling
        rel_diagram_bin_15_test_nts_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nts, None, 15, iteration, 'Op_6_nhb_test_15_static', False)
        rel_diagram_bin_50_test_nts_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nts, None, 50, iteration, 'Op_6_nhb_test_50_static', False)
        rel_diagram_bin_15_test_nts_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nts, None, 15, iteration, 'Op_6_nhb_test_15_interactive', True)
        rel_diagram_bin_50_test_nts_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nts, None, 50, iteration, 'Op_6_nhb_test_50_interactive', True)
        rel_diagram_bin_15_test_nts_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nts_cal, None, 15, iteration, 'Op_6_nhb_cal_15_static', False)
        rel_diagram_bin_50_test_nts_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nts_cal, None, 50, iteration, 'Op_6_nhb_cal_50_static', False)
        rel_diagram_bin_15_test_nts_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nts_cal, None, 15, iteration, 'Op_6_nhb_cal_15_interactive', True)
        rel_diagram_bin_50_test_nts_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nts_cal, None, 50, iteration, 'Op_6_nhb_cal_50_interactive', True)

        # Isotonic Regression
        # custom IR
        rel_diagram_bin_15_test_ir_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_ir, None, 15, iteration, 'Op_3_ir_test_15_static', False)
        rel_diagram_bin_50_test_ir_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_ir, None, 50, iteration, 'Op_3_ir_test_50_static', False)
        rel_diagram_bin_15_test_ir_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_ir, None, 15, iteration, 'Op_3_ir_test_15_interactive', True)
        rel_diagram_bin_50_test_ir_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_ir, None, 50, iteration, 'Op_3_ir_test_50_interactive', True)
        rel_diagram_bin_15_test_ir_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_ir_cal, None, 15, iteration, 'Op_3_ir_cal_15_static', False)
        rel_diagram_bin_50_test_ir_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_ir_cal, None, 50, iteration, 'Op_3_ir_cal_50_static', False)
        rel_diagram_bin_15_test_ir_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_ir_cal, None, 15, iteration, 'Op_3_ir_cal_15_interactive', True)
        rel_diagram_bin_50_test_ir_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_ir_cal, None, 50, iteration, 'Op_3_ir_cal_50_interactive', True)

        # netcal IR
        rel_diagram_bin_15_test_nir_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nir, None, 15, iteration, 'Op_4_nir_test_15_static', False)
        rel_diagram_bin_50_test_nir_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nir, None, 50, iteration, 'Op_4_nir_test_50_static', False)
        rel_diagram_bin_15_test_nir_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nir, None, 15, iteration, 'Op_4_nir_test_15_interactive', True)
        rel_diagram_bin_50_test_nir_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nir, None, 50, iteration, 'Op_4_nir_test_50_interactive', True)
        rel_diagram_bin_15_test_nir_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nir_cal, None, 15, iteration, 'Op_4_nir_cal_15_static', False)
        rel_diagram_bin_50_test_nir_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nir_cal, None, 50, iteration, 'Op_4_nir_cal_50_static', False)
        rel_diagram_bin_15_test_nir_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nir_cal, None, 15, iteration, 'Op_4_nir_cal_15_interactive', True)
        rel_diagram_bin_50_test_nir_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nir_cal, None, 50, iteration, 'Op_4_nir_cal_50_interactive', True)
        
        # Histogram Binning
        # calibration was static
        rel_diagram_bin_15_test_hb_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_hb_15_static, None, 15, iteration, 'Op_5_hb_15_static_calibration_test', False)
        rel_diagram_bin_15_test_hb_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_hb_15_static, None, 15, iteration, 'Op_5_hb_15_static_calibration_test', True)
        rel_diagram_bin_15_test_hb_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_hb_cal_15_static, None, 15, iteration, 'Op_5_hb_15_static_calibration_cal', False)
        rel_diagram_bin_15_test_hb_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_hb_cal_15_static, None, 15, iteration, 'Op_5_hb_15_static_calibration_cal', True)
        rel_diagram_bin_15_test_nhb_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nhb_15_static, None, 15, iteration, 'Op_6_nhb_15_static_calibration_test', False)
        rel_diagram_bin_15_test_nhb_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nhb_15_static, None, 15, iteration, 'Op_6_nhb_15_static_calibration_test', True)
        rel_diagram_bin_15_test_nhb_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nhb_cal_15_static, None, 15, iteration, 'Op_6_nhb_15_static_calibration_cal', False)
        rel_diagram_bin_15_test_nhb_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nhb_cal_15_static, None, 15, iteration, 'Op_6_nhb_15_static_calibration_cal', True)
        
        rel_diagram_bin_50_test_hb_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_hb_50_static, None, 50, iteration, 'Op_5_hb_50_static_calibration_test', False)
        rel_diagram_bin_50_test_hb_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_hb_50_static, None, 50, iteration, 'Op_5_hb_50_static_calibration_test', True)
        rel_diagram_bin_50_test_hb_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_hb_cal_50_static, None, 50, iteration, 'Op_5_hb_50_static_calibration_cal', False)
        rel_diagram_bin_50_test_hb_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_hb_cal_50_static, None, 50, iteration, 'Op_5_hb_50_static_calibration_cal', True)
        rel_diagram_bin_50_test_nhb_static_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nhb_50_static, None, 50, iteration, 'Op_6_nhb_50_static_calibration_test', False)
        rel_diagram_bin_50_test_nhb_interactive_test = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nhb_50_static, None, 50, iteration, 'Op_6_nhb_50_static_calibration_test', True)
        rel_diagram_bin_50_test_nhb_static_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nhb_cal_50_static, None, 50, iteration, 'Op_6_nhb_50_static_calibration_cal', False)
        rel_diagram_bin_50_test_nhb_interactive_cal = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nhb_cal_50_static, None, 50, iteration, 'Op_6_nhb_50_static_calibration_cal', True)
        
        # calibration was interactive
        rel_diagram_bin_15_test_hb_static_test_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_hb_15_interactive, None, 15, iteration, 'Op_5_hb_15_interactive_calibration_test', False)
        rel_diagram_bin_15_test_hb_interactive_test_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_hb_15_interactive, None, 15, iteration, 'Op_5_hb_15_interactive_calibration_test', True)
        rel_diagram_bin_15_test_hb_static_cal_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_hb_cal_15_interactive, None, 15, iteration, 'Op_5_hb_15_interactive_calibration_cal', False)
        rel_diagram_bin_15_test_hb_interactive_cal_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_hb_cal_15_interactive, None, 15, iteration, 'Op_5_hb_15_interactive_calibration_cal', True)
        rel_diagram_bin_15_test_nhb_static_test_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nhb_15_interactive, None, 15, iteration, 'Op_6_nhb_15_interactive_calibration_test', False)
        rel_diagram_bin_15_test_nhb_interactive_test_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nhb_15_interactive, None, 15, iteration, 'Op_6_nhb_15_interactive_calibration_test', True)
        rel_diagram_bin_15_test_nhb_static_cal_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nhb_cal_15_interactive, None, 15, iteration, 'Op_6_nhb_15_interactive_calibration_cal', False)
        rel_diagram_bin_15_test_nhb_interactive_cal_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nhb_cal_15_interactive, None, 15, iteration, 'Op_6_nhb_15_interactive_calibration_cal', True)
        
        rel_diagram_bin_50_test_hb_static_test_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_hb_50_interactive, None, 50, iteration, 'Op_5_hb_50_interactive_calibration_test', False)
        rel_diagram_bin_50_test_hb_interactive_test_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_hb_50_interactive, None, 50, iteration, 'Op_5_hb_50_interactive_calibration_test', True)
        rel_diagram_bin_50_test_hb_static_cal_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_hb_cal_50_interactive, None, 50, iteration, 'Op_5_hb_50_interactive_calibration_cal', False)
        rel_diagram_bin_50_test_hb_interactive_cal_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_hb_cal_50_interactive, None, 50, iteration, 'Op_5_hb_50_interactive_calibration_cal', True)
        rel_diagram_bin_50_test_nhb_static_test_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nhb_50_interactive, None, 50, iteration, 'Op_6_nhb_50_interactive_calibration_test', False)
        rel_diagram_bin_50_test_nhb_interactive_test_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_test, p_calibrated_nhb_50_interactive, None, 50, iteration, 'Op_6_nhb_50_interactive_calibration_test', True)
        rel_diagram_bin_50_test_nhb_static_cal_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nhb_cal_50_interactive, None, 50, iteration, 'Op_6_nhb_50_interactive_calibration_cal', False)
        rel_diagram_bin_50_test_nhb_interactive_cal_interactive_calibration = calculate_metrics.evaluate_reliability_diagram(labels_cal, p_calibrated_nhb_cal_50_interactive, None, 50, iteration, 'Op_6_nhb_50_interactive_calibration_cal', True)
        

        rel_dias = rel_diagram_bin_15_test_uncalibrated_test_static, rel_diagram_bin_50_test_uncalibrated_test_static, rel_diagram_bin_15_test_uncalibrated_test_interactive, rel_diagram_bin_50_test_uncalibrated_test_interactive, rel_diagram_bin_15_test_uncalibrated_cal_static, rel_diagram_bin_50_test_uncalibrated_cal_static, rel_diagram_bin_15_test_uncalibrated_cal_interactive, rel_diagram_bin_50_test_uncalibrated_cal_interactive, rel_diagram_bin_15_test_ps_static_test, rel_diagram_bin_50_test_ps_static_test, rel_diagram_bin_15_test_ps_interactive_test, rel_diagram_bin_50_test_ps_interactive_test, rel_diagram_bin_15_test_ps_static_cal, rel_diagram_bin_50_test_ps_static_cal, rel_diagram_bin_15_test_ps_interactive_cal, rel_diagram_bin_50_test_ps_interactive_cal, rel_diagram_bin_15_test_nps_static_test, rel_diagram_bin_50_test_nps_static_test, rel_diagram_bin_15_test_nps_interactive_test, rel_diagram_bin_50_test_nps_interactive_test, rel_diagram_bin_15_test_nps_static_cal, rel_diagram_bin_50_test_nps_static_cal, rel_diagram_bin_15_test_nps_interactive_cal, rel_diagram_bin_50_test_nps_interactive_cal, rel_diagram_bin_15_test_nts_static_test, rel_diagram_bin_50_test_nts_static_test, rel_diagram_bin_15_test_nts_interactive_test, rel_diagram_bin_50_test_nts_interactive_test, rel_diagram_bin_15_test_nts_static_cal, rel_diagram_bin_50_test_nts_static_cal, rel_diagram_bin_15_test_nts_interactive_cal, rel_diagram_bin_50_test_nts_interactive_cal, rel_diagram_bin_15_test_ir_static_test, rel_diagram_bin_50_test_ir_static_test, rel_diagram_bin_15_test_ir_interactive_test, rel_diagram_bin_50_test_ir_interactive_test, rel_diagram_bin_15_test_ir_static_cal, rel_diagram_bin_50_test_ir_static_cal, rel_diagram_bin_15_test_ir_interactive_cal, rel_diagram_bin_50_test_ir_interactive_cal, rel_diagram_bin_15_test_nir_static_test, rel_diagram_bin_50_test_nir_static_test, rel_diagram_bin_15_test_nir_interactive_test, rel_diagram_bin_50_test_nir_interactive_test, rel_diagram_bin_15_test_nir_static_cal, rel_diagram_bin_50_test_nir_static_cal, rel_diagram_bin_15_test_nir_interactive_cal, rel_diagram_bin_50_test_nir_interactive_cal, rel_diagram_bin_15_test_hb_static_test, rel_diagram_bin_15_test_hb_interactive_test, rel_diagram_bin_15_test_hb_static_cal, rel_diagram_bin_15_test_hb_interactive_cal, rel_diagram_bin_15_test_nhb_static_test, rel_diagram_bin_15_test_nhb_interactive_test, rel_diagram_bin_15_test_nhb_static_cal, rel_diagram_bin_15_test_nhb_interactive_cal, rel_diagram_bin_50_test_hb_static_test, rel_diagram_bin_50_test_hb_interactive_test, rel_diagram_bin_50_test_hb_static_cal, rel_diagram_bin_50_test_hb_interactive_cal, rel_diagram_bin_50_test_nhb_static_test, rel_diagram_bin_50_test_nhb_interactive_test, rel_diagram_bin_50_test_nhb_static_cal, rel_diagram_bin_50_test_nhb_interactive_cal, rel_diagram_bin_15_test_hb_static_test_interactive_calibration, rel_diagram_bin_15_test_hb_interactive_test_interactive_calibration, rel_diagram_bin_15_test_hb_static_cal_interactive_calibration, rel_diagram_bin_15_test_hb_interactive_cal_interactive_calibration, rel_diagram_bin_15_test_nhb_static_test_interactive_calibration, rel_diagram_bin_15_test_nhb_interactive_test_interactive_calibration, rel_diagram_bin_15_test_nhb_static_cal_interactive_calibration, rel_diagram_bin_15_test_nhb_interactive_cal_interactive_calibration, rel_diagram_bin_50_test_hb_static_test_interactive_calibration, rel_diagram_bin_50_test_hb_interactive_test_interactive_calibration, rel_diagram_bin_50_test_hb_static_cal_interactive_calibration, rel_diagram_bin_50_test_hb_interactive_cal_interactive_calibration, rel_diagram_bin_50_test_nhb_static_test_interactive_calibration, rel_diagram_bin_50_test_nhb_interactive_test_interactive_calibration, rel_diagram_bin_50_test_nhb_static_cal_interactive_calibration, rel_diagram_bin_50_test_nhb_interactive_cal_interactive_calibration



        metrics_row = calculate_calibrated_metrics(preds_test, labels_test, preds_cal, labels_cal, p_calibrated, rel_dias)
        calibration_metrics.append(metrics_row)
    write_calibrated_metrics(calibration_metrics)





