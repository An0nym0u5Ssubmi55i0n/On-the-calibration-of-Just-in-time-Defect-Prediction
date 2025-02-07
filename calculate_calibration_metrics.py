from sklearn.calibration import calibration_curve as reliability_diagram
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from netcal.metrics import ECE

import csv
import numpy as np
import matplotlib.pyplot as plt


# we experiment with bin size "n_bins" = 15 and "n_bins" = 50

def compute_ece_mce(p, l, b, interactive_binning= False):
    """
    Compute the Expected Calibration Error (ECE) score and MCE (max calibration error)
    Args:
    p (np.array): Array of prediction probabilities (size N).
    l (np.array): Array of true labels (size N).
    b (int): Number of bins for calibration.

    Returns:
    2 float-s: ECE score and MCE score
    """
    ece = 0.0
    N = len(p)
    mce=0
    # Discretize probabilities into bins
    if interactive_binning:
        quantiles = np.linspace(0, 1, b + 1)
        bin_edges = np.percentile(p, quantiles * 100)
    else:
        bin_edges = np.linspace(0, 1, b + 1)

    for i in range(b):
        # Find the indices of predictions falling within the current bin
        in_bin = np.where((p >= bin_edges[i]) & (p < bin_edges[i + 1]))[0]

        if len(in_bin) == 0:
            continue

        # Calculate the accuracy of the predictions in this bin
        correct = []
        for i in in_bin:
            if l[i] == 1:
                correct.append(i)

        accuracy = len(correct)/len(in_bin)

        # Calculate the average predicted probability in this bin
        avg_prob = np.mean(p[in_bin])

        # Compute the absolute difference between accuracy and average probability
        if mce < abs(accuracy - avg_prob):
            mce= abs(accuracy - avg_prob)
        ece += len(in_bin) / N * abs(accuracy - avg_prob)

    return ece, mce


def calculate_brier_score(labels, preds):
    return brier_score_loss(labels, preds, pos_label=1.0)


def plot_rel_dia(confidence, accuracy, bins, iteration, dataset, interactive_binning=False, scikit_impl=False,
                 calibration=None):
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
        f'Reliability Diagram for iteration {iteration} of {dataset} dataset with {len(confidence)} actual bins (interactive binning {interactive_binning}) calibration_{calibration}')
    plt.legend()

    # Show plot
    plt.grid()
    plt.savefig(
        f'../{ADD_DIRECTORY}/Rel_diagram_{dataset}_{bins} bins interactive binning {interactive_binning}_calibration_{calibration}.png')
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

    plot_rel_dia(confidence, accuracy, n_bins, iteration, dataset, interactive_binning, True)

    return confidence, accuracy, bin_total, bin_total[nonzero]
