from sklearn.calibration import calibration_curve as reliability_diagram
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from netcal.metrics import ECE
from calibration_metrics import calculate_brier_score, calculate_ece, calculate_mce, evaluate_reliability_diagram, _full_accuracy_rel_dia

import csv
import numpy as np
import matplotlib.pyplot as plt

CLASSIFICATION_THRESHHOLD = 0.5
DATASET = 'op'
CALIBRATION = 'False'


def write_calibration_scores(scores_list, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for index, scores in enumerate(scores_list):
            if index == 0: writer.writerow(np.asarray(scores.keys()).tolist())
            writer.writerow(np.asarray(scores.values()).tolist())

def read_input(filename):
    predictions = []
    labels = []
    predicted_category = []
    confusion_matrix_category = []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index == 0: continue
            predictions.append(float(row[1]))
            labels.append(float(row[2]))
            predicted_category.append(row[4])
            confusion_matrix_category.append(row[5])
    return predictions, labels, predicted_category, confusion_matrix_category


if __name__ == '__main__':
    calibrations = ['True', 'False']
    for calibration in calibrations:
        CALIBRATION = calibration
        all_scores = []
        for iteration in range(100):
            filename = f'../Add_your_path/eval_calibration_{calibration}_iteration_{iteration}_evaluation_{DATASET}.csv'
            input_data = read_input(filename)
            predictions, labels, predicted_categories, confusion_matrix_categories = input_data
            labels = np.asarray(labels)
            predictions = np.asarray(predictions)

            predictions_true_class = []
            true_predictions = []
            for index, cm_cat in enumerate(confusion_matrix_categories):
                # adjust prediction to be prediction of the true class
                label = labels[index]
                prediction_true_class = predictions[index]
                if label == 0.0: prediction_true_class = 1-prediction_true_class
                predictions_true_class.append(prediction_true_class)
                # create weighting array for correct predictions
                if "True" in cm_cat:
                    true_predictions.append(1)
                else:
                    true_predictions.append(0)

            # true_predictions = [1 if "True" in cm_cat else 0 for cm_cat in confusion_matrix_categories]

            # RELIABILITY DIAGRAMS
            rel_diagram_bin_15 = evaluate_reliability_diagram(labels, predictions, true_predictions, 15, iteration, DATASET)
            rel_diagram_bin_15_interactive = evaluate_reliability_diagram(labels, predictions, true_predictions, 15, iteration, DATASET, interactive_binning=True)
            rel_diagram_bin_50 = evaluate_reliability_diagram(labels, predictions, true_predictions, 50, iteration, DATASET)
            
            # RELIABILITY DIAGRAMS using accuracy impl
            rel_diagram_bin_15_accuracy = _full_accuracy_rel_dia(labels, predictions_true_class, true_predictions, 15, iteration, DATASET)
            rel_diagram_bin_15_interactive_accuracy = _full_accuracy_rel_dia(labels, predictions_true_class, true_predictions, 15, iteration, DATASET, interactive_binning=True)
            rel_diagram_bin_50_accuracy = _full_accuracy_rel_dia(labels, predictions_true_class, true_predictions, 50, iteration, DATASET)

            # ECE Scores of the last iteration
            if iteration == 99:
                ece_bin_15 = calculate_ece(rel_diagram_bin_15[0], rel_diagram_bin_15[1], True)
                ece_bin_15_interactive = calculate_ece(rel_diagram_bin_15_interactive[0], rel_diagram_bin_15_interactive[1], True)
                ece_bin_50 = calculate_ece(rel_diagram_bin_50[0], rel_diagram_bin_50[1], True)
            else:
                ece_bin_15 = calculate_ece(rel_diagram_bin_15[0], rel_diagram_bin_15[1])
                ece_bin_15_interactive = calculate_ece(rel_diagram_bin_15_interactive[0], rel_diagram_bin_15_interactive[1])
                ece_bin_50 = calculate_ece(rel_diagram_bin_50[0], rel_diagram_bin_50[1])
            
            # ECE Scores using accuracy impl,
            if iteration == 99:
                ece_bin_15_accuracy = calculate_ece(rel_diagram_bin_15_accuracy[0], rel_diagram_bin_15_accuracy[1], True)
                ece_bin_15_interactive_accuracy = calculate_ece(rel_diagram_bin_15_interactive_accuracy[0], rel_diagram_bin_15_interactive_accuracy[1], True)
                ece_bin_50_accuracy = calculate_ece(rel_diagram_bin_50_accuracy[0], rel_diagram_bin_50_accuracy[1], True)
            else:
                ece_bin_15_accuracy = calculate_ece(rel_diagram_bin_15_accuracy[0], rel_diagram_bin_15_accuracy[1])
                ece_bin_15_interactive_accuracy = calculate_ece(rel_diagram_bin_15_interactive_accuracy[0], rel_diagram_bin_15_interactive_accuracy[1])
                ece_bin_50_accuracy = calculate_ece(rel_diagram_bin_50_accuracy[0], rel_diagram_bin_50_accuracy[1])

            # MCE Scores
            mce_bin_15 = calculate_mce(rel_diagram_bin_15[0], rel_diagram_bin_15[1])
            mce_bin_15_interactive = calculate_mce(rel_diagram_bin_15_interactive[0], rel_diagram_bin_15_interactive[1])
            mce_bin_50 = calculate_mce(rel_diagram_bin_50[0], rel_diagram_bin_50[1])
            
            # MCE Scores using accuracy impl
            mce_bin_15_accuracy = calculate_mce(rel_diagram_bin_15_accuracy[0], rel_diagram_bin_15_accuracy[1])
            mce_bin_15_interactive_accuracy = calculate_mce(rel_diagram_bin_15_interactive_accuracy[0], rel_diagram_bin_15_interactive_accuracy[1])
            mce_bin_50_accuracy = calculate_mce(rel_diagram_bin_50_accuracy[0], rel_diagram_bin_50_accuracy[1])

            # netcal ECE
            ece = ECE(15)
            ece_score_bin_15_netcal = ece.measure(predictions, labels)
            ece_50 = ECE(50)
            ece_score_bin_50_netcal = ece_50.measure(predictions, labels)

            # AUC score
            auc_score = roc_auc_score(labels, predictions)

            # Log Loss
            log_loss_calc = log_loss(labels, predictions)

            # Brier Score
            brier_score = calculate_brier_score(labels, predictions)

            # print(f"Len rel diagram = {len(rel_diagram_bin_15[1])} \n and its accuracy  = {rel_diagram_bin_15[1]}")
            # print(f"\nLen rel diagram = {len(rel_diagram_bin_50[1])} \n and its accuracy = {rel_diagram_bin_50[1]}")
            scores = {'rel_diagram_bin_15': rel_diagram_bin_15, 
                    'rel_diagram_bin_15_interactive': rel_diagram_bin_15_interactive,
                    "rel_diagram_bin_50": rel_diagram_bin_50, 
                    "ece_bin_15": ece_bin_15, 
                    "ece_bin_15_interactive": ece_bin_15_interactive,
                    "ece_score_bin_15_netcal": ece_score_bin_15_netcal,
                    "ece_bin_50": ece_bin_50,
                    "ece_score_bin_50_netcal": ece_score_bin_50_netcal,
                    "mce_bin_15": mce_bin_15, 
                    "mce_bin_15_interactive": mce_bin_15_interactive,
                    "mce_bin_50": mce_bin_50,
                    "auc_score": auc_score,
                    "log_loss": log_loss_calc,
                    "brier_score": brier_score,
                    'rel_diagram_bin_15_accuracy': rel_diagram_bin_15_accuracy,
                    'rel_diagram_bin_15_interactive_accuracy': rel_diagram_bin_15_interactive_accuracy,
                    "rel_diagram_bin_50_accuracy": rel_diagram_bin_50_accuracy,
                    "ece_bin_15_accuracy": ece_bin_15_accuracy,
                    "ece_bin_15_interactive_accuracy": ece_bin_15_interactive_accuracy,
                    "ece_bin_50_accuracy": ece_bin_50_accuracy,
                    "mce_bin_15_accuracy": mce_bin_15_accuracy,
                    "mce_bin_15_interactive_accuracy": mce_bin_15_interactive_accuracy,
                    "mce_bin_50_accuracy": mce_bin_50_accuracy,
                    }
            all_scores.append(scores)
            print(f"Iteration {iteration} done.")
        write_calibration_scores(all_scores, f'../Add_your_path/calibration_scores_aggregated_calibration_{calibration}_including_accuracy_{DATASET}.csv')


