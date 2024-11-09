import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import brier_score_loss


def calibration_score(all_label, all_predict):
    # Weight distribution with 15 bins of equal width
    bin_edges_15 = np.linspace(0.0, 1.0, 16)
    bin_ids_15 = np.searchsorted(bin_edges_15[1:-1], all_predict)
    weight_15 = np.bincount(bin_ids_15)
    weight_15 = weight_15[weight_15 != 0]
    prob_true, prob_pred = calibration_curve(all_label, all_predict, n_bins=15, strategy='uniform')

    # Calculation of ECE and MCE with 15 bins of equal width
    ece_15 = sum(weight_15 * abs(prob_true - prob_pred)) / len(all_label)
    mce_15 = max(abs(prob_true - prob_pred))
    print("Expected Calibration Error with 15 bins:", ece_15)
    print("Maximum Calibration Error with 15 bins:", mce_15)

    # Weight distribution with 50 bins of equal width
    bin_edges_50 = np.linspace(0, 1, 51)
    bin_ids_50 = np.searchsorted(bin_edges_50[1:-1], all_predict)
    weight_50 = np.bincount(bin_ids_50)
    weight_50 = weight_50[weight_50 != 0]
    prob_true, prob_pred = calibration_curve(all_label, all_predict, n_bins=50, strategy='uniform')

    # Calculation of ECE and MCE with 50 bins of equal width
    ece_50 = sum(weight_50 * abs(prob_true - prob_pred)) / len(all_label)
    mce_50 = max(abs(prob_true - prob_pred))
    print("Expected Calibration Error with 50 bins:", ece_50)
    print("Maximum Calibration Error with 50 bins:", mce_50)

    # Weight distribution with interactive binning schema (15 bins)
    quantiles_15 = np.linspace(0, 1, 16)
    bin_edges_ibc15 = np.percentile(all_predict, quantiles_15 * 100)
    bin_ids_ibc15 = np.searchsorted(bin_edges_ibc15[1:-1], all_predict)
    weight_ibc15 = np.bincount(bin_ids_ibc15)
    prob_true, prob_pred = calibration_curve(all_label, all_predict, n_bins=15, strategy='quantile')

    # Calculation of ECE and MCE with interactive binning schema (15 bins)
    ece_ibc15 = sum(weight_ibc15 * abs(prob_true - prob_pred)) / len(all_label)
    mce_ibc15 = max(abs(prob_true - prob_pred))
    print("Expected Calibration Error with interactive binning schema (15):", ece_ibc15)
    print("Maximum Calibration Error with interactive binning schema (15):", mce_ibc15)

    # Weight distribution with interactive binning schema (50 bins)
    quantiles_50 = np.linspace(0, 1, 51)
    bin_edges_ibc50 = np.percentile(all_predict, quantiles_50 * 100)
    bin_ids_ibc50 = np.searchsorted(bin_edges_ibc50[1:-1], all_predict)
    weight_ibc50 = np.bincount(bin_ids_ibc50)
    prob_true, prob_pred = calibration_curve(all_label, all_predict, n_bins=50, strategy='quantile')

    # Calculation of ECE and MCE with interactive binning schema (50 bins)
    ece_ibc50 = sum(weight_ibc50 * abs(prob_true - prob_pred)) / len(all_label)
    mce_ibc50 = max(abs(prob_true - prob_pred))
    print("Expected Calibration Error with interactive binning schema (50):", ece_ibc50)
    print("Maximum Calibration Error with interactive binning schema (50):", mce_ibc50)

    # Calculation of Brier Score
    brier = brier_score_loss(all_label, all_predict)
    print("Brier Score:", brier)
