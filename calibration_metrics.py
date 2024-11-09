from sklearn.calibration import calibration_curve as reliability_diagram
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from netcal.metrics import ECE

import csv
import numpy as np
# from conformal_prediction import apply_calibration, apply_conformal_prediction
import matplotlib.pyplot as plt

CLASSIFICATION_THRESHHOLD = 0.5
DATASET = 'op'
CALIBRATION = 'False'


# we experiment with bin size "n_bins" = 15 and "n_bins" = 50

def _full_accuracy_rel_dia(labels, preds, true_preds, n_bins, iteration, dataset, interactive_binning=False):
    # cf. https://github.com/scikit-learn/scikit-learn/blob/5491dc695/sklearn/calibration.py#L915 (extended for correct accuracy)
    if interactive_binning:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(preds, quantiles * 100)
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    binids = np.searchsorted(bins[1:-1], preds)

    bin_total = np.bincount(binids, minlength=len(bins))
    bin_sums = np.bincount(binids, weights=preds, minlength=len(bins))
    bin_true_pos_and_neg = np.bincount(binids, weights=true_preds, minlength=len(bins))

    nonzero = bin_total != 0
    confidence = bin_sums[nonzero] / bin_total[nonzero]
    accuracy = bin_true_pos_and_neg[nonzero] / bin_total[nonzero]

    plot_rel_dia(confidence, accuracy, n_bins, iteration, dataset, interactive_binning, False)

    return confidence, accuracy, bin_total


def plot_rel_dia(confidence, accuracy, bins, iteration, dataset, interactive_binning=False, scikit_impl=False,
                 calibration=CALIBRATION):
    # Plot the bar chart
    fig = plt.figure(figsize=(12, 6))
    bar_width = 0.01
    label = 'Accuracy'
    if scikit_impl: label = 'Fraction of defect-inducing commits'
    plt.bar(confidence - bar_width / 2, accuracy, bar_width, color='b', alpha=0.6, label=label)

    # Plot the reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

    # Add labels and title
    plt.xlabel('Confidence')
    plt.ylabel(label)
    plt.title(
        f'Reliability Diagram for iteration {iteration} of {dataset} dataset with {len(confidence)} actual bins (interactive binning {interactive_binning})')
    plt.legend()

    # Show plot
    plt.grid()
    plt.savefig(
        f'../experiment_results_oversampled_2/Rel_diagram_iteration {iteration}_{dataset}_{bins} bins interactive binning {interactive_binning}_scikit_impl_{scikit_impl}_calibration_{calibration}.png')
    plt.close(fig)


def evaluate_reliability_diagram(labels, preds, true_preds, n_bins, iteration, dataset, interactive_binning=False):
    # cf. https://github.com/scikit-learn/scikit-learn/blob/5491dc695/sklearn/calibration.py#L915 (extended for correct accuracy)
    if interactive_binning:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(preds, quantiles * 100)
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    binids = np.searchsorted(bins[1:-1], preds)

    bin_total = np.bincount(binids, minlength=len(bins))
    bin_sums = np.bincount(binids, weights=preds, minlength=len(bins))
    bin_true = np.bincount(binids, weights=labels, minlength=len(bins))

    nonzero = bin_total != 0
    confidence = bin_sums[nonzero] / bin_total[nonzero]
    accuracy = bin_true[nonzero] / bin_total[nonzero]

    # if interactive_binning:
    #     accuracy, confidence = reliability_diagram(labels, preds, n_bins=bins, strategy='quantile')
    # else:
    #     accuracy, confidence = reliability_diagram(labels, preds, n_bins=bins, strategy='uniform')

    plot_rel_dia(confidence, accuracy, n_bins, iteration, dataset, interactive_binning, True)

    return confidence, accuracy, bin_total


def calculate_ece(confidence, accuracy, verbose=False):
    total_error = 0
    number_actual_bins = len(confidence)
    for index, confidence_bin in enumerate(confidence):
        error = confidence_bin - accuracy[index]
        # print(f"[ECE bin {index}] confidence_bin = {confidence_bin} with accuracy[index] = {accuracy[index]}")
        if error < 0: error *= -1
        total_error += error
    return total_error / number_actual_bins


def calculate_mce(confidence, accuracy):
    max_error = 0
    max_error_index = 0
    for index, confidence_bin in enumerate(confidence):
        error = confidence_bin - accuracy[index]
        if error < 0: error *= -1
        if error > max_error:
            max_error = error
            max_error_index = index

    return max_error


def calculate_brier_score(labels, preds):
    return brier_score_loss(labels, preds, pos_label=1.0)
