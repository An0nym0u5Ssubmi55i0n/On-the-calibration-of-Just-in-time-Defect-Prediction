import math
import random
import time
import argparse
import csv

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict, KFold
import matplotlib.pyplot as plt
from LR import LR
from DBN import DBN
from sklearn.calibration import calibration_curve as reliability_diagram
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from netcal.metrics import ECE
from calibration_metrics import calculate_brier_score, calculate_ece, calculate_mce, _full_accuracy_rel_dia, evaluate_reliability_diagram, plot_rel_dia
from calibration_methods import invert_sigmoid_scores, custom_platt_scaling, platt_scaling, temperature_scaling
from calibration_process import calculate_calibrated_metrics, write_calibrated_metrics


DATASET = 'openstack'
CALIBRATIONMODE = "netcal_platt"
THRESHOLD = 0.3

parser = argparse.ArgumentParser()

parser.add_argument('-project', type=str,
                    default='qt')
parser.add_argument('-data', type=str,
                    default='k')
parser.add_argument('-algorithm', type=str,
                    default='lr')
parser.add_argument('-drop', type=str,
                    default='')
parser.add_argument('-only', nargs='+',
                    default=[])

def evaluation_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_pred = [1 if p >= THRESHOLD else 0 for p in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    # f1 = 2 * prc * rc / (prc + rc)
    f1 = 0
    return acc, prc, rc, f1, auc_

def roc_curve_write(y_true, y_pred, filename):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fpr', 'tpr', 'thresholds'])
        writer.writerow([fpr, tpr, thresholds])
            
def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    if args.drop:
        df = df.drop(columns=[args.drop])
    elif args.only:
        df = df[['Unnamed: 0','_id','date','bug','__'] + args.only]
    return df.values

def get_features(data):
    # return the features of yasu data
    return data[:, 5:]

def get_ids(data):
    # return the labels of yasu data
    return data[:, 1:2].flatten().tolist()

def get_label(data):
    data = data[:, 3:4].flatten().tolist()
    data = [1 if int(d) > 0 else 0 for d in data]
    return data

def load_df_yasu_data(path_data):
    data = pd.read_csv(path_data)
    data = replace_value_dataframe(df=data)
    ids, labels, features = get_ids(data=data), get_label(data=data), get_features(data=data)
    indexes = list()
    cnt_noexits = 0
    for i in range(0, len(ids)):
        try:
            indexes.append(i)
        except FileNotFoundError:
            print('File commit id no exits', ids[i], cnt_noexits)
            cnt_noexits += 1
    ids = [ids[i] for i in indexes]
    labels = [labels[i] for i in indexes]
    features = features[indexes]
    return (ids, np.array(labels), features)

def load_yasu_data(args):
  
    train_path_data = 'data/{}/{}_train.csv'.format(args.project, args.data)
    test_path_data = 'data/{}/{}_test.csv'.format(args.project, args.data)
    train, test = load_df_yasu_data(train_path_data), load_df_yasu_data(test_path_data)
    return train, test

def train_and_evl(data, label, args):
    size = int(label.shape[0]*0.2)
    auc_ = []
   
    for i in range(5):
        idx = size * i
        X_e = data[idx:idx+size]
        y_e = label[idx:idx+size]

        X_t = np.vstack((data[:idx], data[idx+size:]))
        y_t = np.hstack((label[:idx], label[idx+size:]))


        model = LogisticRegression(max_iter=7000).fit(X_t, y_t)
        y_pred = model.predict_proba(X_e)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_e, y_score=y_pred, pos_label=1)
        auc_.append(auc(fpr, tpr))

    return np.mean(auc_)

def mini_batches_update(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)
    
    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X, mini_batch_Y = shuffled_X[indexes], shuffled_Y[indexes]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)
    
    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y

    # Step 2: Partition (X, Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitioning
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def DBN_JIT(train_features, train_labels, test_features, test_labels, hidden_units=[20, 12, 12], num_epochs_LR=200):
    # training DBN model

    #################################################################################################
    starttime = time.time()
    dbn_model = DBN(visible_units=train_features.shape[1],
                    hidden_units=hidden_units,
                    use_gpu=False)
    dbn_model.train_static(train_features, train_labels, num_epochs=10)
    # Finishing the training DBN model
    # print('---------------------Finishing the training DBN model---------------------')
    # using DBN model to construct features
    DBN_train_features, _ = dbn_model.forward(train_features)
    DBN_test_features, _ = dbn_model.forward(test_features)
    DBN_train_features = DBN_train_features.numpy()
    DBN_test_features = DBN_test_features.numpy()

    train_features = np.hstack((train_features, DBN_train_features))
    test_features = np.hstack((test_features, DBN_test_features))


    if len(train_labels.shape) == 1:
        num_classes = 1
    else:
        num_classes = train_labels.shape[1]
    # lr_model = LR(input_size=hidden_units, num_classes=num_classes)
    lr_model = LR(input_size=train_features.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(lr_model.parameters(), lr=0.00001)
    steps = 0
    batches_test = mini_batches(X=test_features, Y=test_labels)
    for epoch in range(1, num_epochs_LR + 1):
        # building batches for training model
        batches_train = mini_batches_update(X=train_features, Y=train_labels)
        for batch in batches_train:
            x_batch, y_batch = batch
            x_batch, y_batch = torch.tensor(x_batch).float(), torch.tensor(y_batch).float()

            optimizer.zero_grad()
            predict = lr_model.forward(x_batch)
            loss = nn.BCELoss()
            loss = loss(predict, y_batch)
            loss.backward()
            optimizer.step()

            # steps += 1
            # if steps % 100 == 0:
            #     print('\rEpoch: {} step: {} - loss: {:.6f}'.format(epoch, steps, loss.item()))

    endtime = time.time()
    dtime = endtime - starttime
    print("Train Time: %.8s s" % dtime)  #

    starttime = time.time()
    y_pred, lables = lr_model.predict(data=batches_test)
    endtime = time.time()
    dtime = endtime - starttime
    print("Eval Time: %.8s s" % dtime)  #
    return y_pred

def baseline_crossVal(train, test):
    #Crossvalidation without calibration metrics
    _, y_train, X_train = train
    _, y_test, X_test = test
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    #acc, prc, rc, f1, auc_ = 0, 0, 0, 0, 0
    model = LogisticRegression(max_iter=7000)
    kf = KFold(n_splits = 10, shuffle=True, random_state=42)
    fold_accuracies=[]
    for train_index, val_index in kf.split(X_train):
        x_train, x_val = X_train[train_index], X_train[val_index]
        Y_train, Y_val = y_train[train_index], y_train[val_index]
        model.fit(x_train, Y_train)
        Y_pred = model.predict(x_val)
        accuracy = accuracy_score(Y_val, Y_pred)
        fold_accuracies.append(accuracy)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
    disp.plot()
    plt.show()

def write_calibration_scores(scores_list, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for index, scores in enumerate(scores_list):
            if index == 0: writer.writerow(np.asarray(scores.keys()).tolist())
            writer.writerow(np.asarray(scores.values()).tolist())

def extract_calibration_metrics(y_prediction_probabilities, y_true_labels, filename, fold, last=False):
    classification_threshold = THRESHOLD
    #Lists for Categories
    predicted_category =[]
    confusion_matrix_category =[]
    #Fill Categories
    for i in range(len(y_true_labels)):
        #Predicted Category based on threshold
        if y_prediction_probabilities[i] > classification_threshold:
            predicted_class = 1
        else:
            predicted_class = 0
        predicted_category.append(predicted_class)
        
        #Confusion Matrix Category
        if y_true_labels[i] == 1 and y_prediction_probabilities[i] > classification_threshold:
            confusion_matrix_category.append('True_Positive')
        elif y_true_labels[i] == 1 and y_prediction_probabilities[i] <= classification_threshold:
            confusion_matrix_category.append('False_Negative')
        elif y_true_labels[i] == 0 and y_prediction_probabilities[i] > classification_threshold:
            confusion_matrix_category.append('False_Negative')
        else:
            confusion_matrix_category.append('True_Negative')
    
    labels = np.asarray(y_true_labels)
    predictions = np.asarray(y_prediction_probabilities)
    #print("Calibration")
    all_scores = []
    predictions_true_class = []
    true_predictions = []
    for index, cm_cat in enumerate(confusion_matrix_category):
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
    rel_diagram_bin_15 = evaluate_reliability_diagram(labels, predictions, true_predictions, 15, fold, DATASET)
    rel_diagram_bin_15_interactive = evaluate_reliability_diagram(labels, predictions, true_predictions, 15, fold, DATASET, interactive_binning=True)
    rel_diagram_bin_50 = evaluate_reliability_diagram(labels, predictions, true_predictions, 50, fold, DATASET)
        
    # RELIABILITY DIAGRAMS using accuracy impl
    rel_diagram_bin_15_accuracy = _full_accuracy_rel_dia(labels, predictions_true_class, true_predictions, 15, fold, DATASET)
    rel_diagram_bin_15_interactive_accuracy = _full_accuracy_rel_dia(labels, predictions_true_class, true_predictions, 15, fold, DATASET, interactive_binning=True)
    rel_diagram_bin_50_accuracy = _full_accuracy_rel_dia(labels, predictions_true_class, true_predictions, 50, fold, DATASET)

    # ECE Scores of the last iteration
    if last:
        ece_bin_15 = calculate_ece(rel_diagram_bin_15[0], rel_diagram_bin_15[1], True)
        ece_bin_15_interactive = calculate_ece(rel_diagram_bin_15_interactive[0], rel_diagram_bin_15_interactive[1], True)
        ece_bin_50 = calculate_ece(rel_diagram_bin_50[0], rel_diagram_bin_50[1], True)
    else:
        ece_bin_15 = calculate_ece(rel_diagram_bin_15[0], rel_diagram_bin_15[1])
        ece_bin_15_interactive = calculate_ece(rel_diagram_bin_15_interactive[0], rel_diagram_bin_15_interactive[1])
        ece_bin_50 = calculate_ece(rel_diagram_bin_50[0], rel_diagram_bin_50[1])
        
    # ECE Scores using accuracy impl,
    if last:
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

    # Accuracy
    acc, prc, rc, f1, auc_ = evaluation_metrics(labels, predictions)

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
                "accuracy": acc,
                "prc": prc,
                "rc": rc,
                "f1": f1,
                "auc": auc_,
                }
    all_scores.append(scores)
        
    write_calibration_scores(all_scores, f'calibration_scores/calibration_scores_aggregated_fold_{fold}_{DATASET}.csv')

    
        #write results to csv
    with open(filename, 'w', newline='') as csvfile:
        writer= csv.writer(csvfile)

        #Header
        writer.writerow(['Index', 'Prediction', 'Label', 'Predicted_Category', 'Confusion_Matrix_Category'])

        #write Data
        for i in range(len(y_prediction_probabilities)):
            writer.writerow([i+1, y_prediction_probabilities[i], y_true_labels[i], predicted_category[i], confusion_matrix_category[i]])
#crossvalidation with calibration metrics
def baseline_crossVal_Calibrate(train, test):
    
    _, y_train, X_train = train
    _, y_test, X_test = test
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    acc, prc, rc, f1, auc_ = 0, 0, 0, 0, 0
    model = LogisticRegression(max_iter=7000)
    kf = KFold(n_splits = 10, shuffle=True, random_state=42)
    fold_accuracies=[]
    count = 0
    for train_index, val_index in kf.split(X_train):
        #print(type(train_index))
        filename = f'calibration_extraction/openstack/eval_calibration_training_fold_{count}_{DATASET}.csv'
        x_train, x_val = X_train[train_index], X_train[val_index]
        Y_train, Y_val = y_train[train_index], y_train[val_index]
        #Train Model
        model.fit(x_train, Y_train)
        #Validate Fold
        Y_pred = model.predict(x_val)
        #Predict Probabilities of only the positive label
        Y_pred_prob = model.predict_proba(x_val)[:,1]
        extract_calibration_metrics(Y_pred_prob, Y_val, filename, count)
        filename = f'roc_curve/roc_curve_fold_{count}_{DATASET}'
        roc_curve_write(Y_val,Y_pred_prob,filename)
        matrix = confusion_matrix(y_true=Y_val, y_pred=Y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
        disp.plot().figure_.savefig(f'calibration_extraction/openstack/eval_calibration_training_fold_{count}_confusion_matrix_{DATASET}.png')
        count+=1
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:,1]
    y_pred_labels = model.predict(X_test)
    matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
    disp.plot().figure_.savefig(f'calibration_extraction/openstack/eval_calibration_testset_confusion_matrix_{DATASET}.png')
    #print(y_pred)
    filename = f'calibration_extraction/openstack/eval_calibration_testset_evaluation_{DATASET}.csv'
    extract_calibration_metrics(y_pred, y_test, filename, count, last=True)
    filename = f'roc_curve/roc_curve_testset_{DATASET}'
    roc_curve_write(Y_val,Y_pred_prob,filename)

def baseline_crossval_with_Calibration_Fold(train, test):
    _, y_train, X_train = train
    _, y_test, X_test = test
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    acc, prc, rc, f1, auc_ = 0, 0, 0, 0, 0
    model = LogisticRegression(max_iter=7000)
    kf = KFold(n_splits = 10, shuffle=True, random_state=42)
    calibration_metrics=[]

    count =0
    # Cross-Validation with 8 Folds:
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        #10 Folds: 8 Training, 1 Calibration, 1 Validation
        train_val_split=KFold(n_splits=2, shuffle=True, random_state=42)
        train_fold_idx, calib_fold_idx = next(train_val_split.split(train_idx))
        
        #Division of Indices
        #8 for training
        xTrain, yTrain = X_train[train_idx[train_fold_idx]], y_train[train_idx[train_fold_idx]]
        #1 for Calibration
        xCalib, yCalib = X_train[train_idx[calib_fold_idx]], y_train[train_idx[calib_fold_idx]]
        #1 for Measuring
        xVal, yVal = X_train[val_idx], y_train[val_idx]

        #Train Model
        model.fit(xTrain, yTrain)

        # Calibrate Model
        # Fix Calibration with Fold 9:
        # TODO Fix Calibration with Fold 9
        Y_pred_prob_cal = model.predict_proba(xCalib)[:,1]
        # prediction threshold THRESHOLD
        y_pred_cal = (Y_pred_prob_cal >= THRESHOLD).astype(int)
        #prediction threshold 0.5
        #y_pred_cal = model.predict(xCalib)
        acc, prc, rc, f1, auc_ = evaluation_metrics(np.asarray(yCalib), np.asarray(Y_pred_prob_cal))
        file = f'accuracy/acc_prc_rc_f1_auc_calibration_fold_{fold}_dataset_{DATASET}_threshold_{THRESHOLD}_scaling_{CALIBRATIONMODE}.csv'
        with open(file, 'w', newline='') as csvfile:
            writer= csv.writer(csvfile)
            #Header
            writer.writerow(['acc', 'prc', 'rc', 'f1', 'auc_'])
            #write Data
            writer.writerow([acc, prc, rc, f1, auc_])
        
        rows=[]
        for i in range(len(xCalib)):
            row = [xCalib[i], yCalib[i], Y_pred_prob_cal[i], y_pred_cal[i]]
            rows.append(row)

        file = f'other_results/raw_data_calibration_fold_fold_{fold}_dataset_{DATASET}_threshold_{THRESHOLD}.csv'

        with open(file, 'w', newline='') as csvfile:
            writer= csv.writer(csvfile)
        
            #Header
            writer.writerow(['Index', 'Input', 'True_Label', 'Predicted_probability', 'Predicted_Label'])
            row = []
            #write Data
            for i in range(len(rows)):
                #print(rows[i])
                writer.writerow([i+1, rows[i][0], rows[i][1], rows[i][2], rows[i][3]])

        
        #Predict Probabilities of only the positive label
        Y_pred_prob_val = model.predict_proba(xVal)[:,1]
        #Validate Fold 10
        #With Threshold THRESHOLD
        Y_pred = (Y_pred_prob_val >= THRESHOLD).astype(int)
        #With Threshold 0.5
        #Y_pred = model.predict(xVal)

        acc, prc, rc, f1, auc_ = evaluation_metrics(np.asarray(yVal), np.asarray(Y_pred_prob_val))
        file = f'accuracy/acc_prc_rc_f1_auc_validation_fold_{fold}_dataset_{DATASET}_threshold_{THRESHOLD}_scaling_{CALIBRATIONMODE}.csv'
        with open(file, 'w', newline='') as csvfile:
            writer= csv.writer(csvfile)
            #Header
            writer.writerow(['acc', 'prc', 'rc', 'f1', 'auc_'])
            #write Data
            writer.writerow([acc, prc, rc, f1, auc_])
        
        rows =[]
        for i in range(len(xVal)):
            row = [xVal[i], yVal[i], Y_pred_prob_val[i], Y_pred[i]]
            rows.append(row)

        file = f'other_results/raw_data_validation_fold_{fold}_dataset_{DATASET}_threshold_{THRESHOLD}.csv'

        with open(file, 'w', newline='') as csvfile:
            writer= csv.writer(csvfile)
            
            #Header
            writer.writerow(['Index', 'Input', 'True_Label', 'Predicted_probability', 'Predicted_Label'])
            row = []
            #write Data
            for i in range(len(rows)):
                #print(rows[i])
                writer.writerow([i+1, rows[i][0], rows[i][1], rows[i][2], rows[i][3]])

        # Need: Predictions (Y_pred_prob_cal), Labels (yCalib)
        # Logits
        preds_cal_inverted = np.asarray(invert_sigmoid_scores(Y_pred_prob_cal))
        preds_val_inverted = np.asarray(invert_sigmoid_scores(Y_pred_prob_val))

        # 1. Platt Scaling
        # Custom Platt
        #p_calibrated_inv_ps, p_calibrated_inv_ps_cal = custom_platt_scaling(preds_cal_inverted, yCalib, preds_val_inverted, yVal)
        #"""
        # TODO Netcal Platt
        p_calibrated_nps, p_calibrated_nps_cal = platt_scaling(Y_pred_prob_cal, yCalib, Y_pred_prob_val, yVal)
        rows=[]

        #find longest array:
        if len(p_calibrated_nps)>len(p_calibrated_nps_cal):
            for i in range(p_calibrated_nps.shape[0]):
                default_value = 0
                if(i >= len(p_calibrated_nps_cal)):
                    row = [p_calibrated_nps[i], default_value]
                else:
                    row = [p_calibrated_nps[i], p_calibrated_nps_cal[i]]
                rows.append(row)
        else:
            for i in range(p_calibrated_nps_cal.shape[0]):
                default_value = 0
                if(i >= len(p_calibrated_nps)):
                    row = [default_value , p_calibrated_nps_cal[i]]
                else:
                    row = [p_calibrated_nps[i], p_calibrated_nps_cal[i]]
                rows.append(row)

        

        file = f'other_results/raw_data_platt_scaling_fold_{fold}_dataset_{DATASET}_threshold_{THRESHOLD}.csv'

        with open(file, 'w', newline='') as csvfile:
            writer= csv.writer(csvfile)
        
            #Header
            writer.writerow(['Index', 'p_calibrated_nps', 'p_calibrated_nps_cal'])
            row = []
            #write Data
            for i in range(len(rows)):
                #print(rows[i])
                writer.writerow([i+1, rows[i][0], rows[i][1]])

        
        """
        # TODO 2. Temperature Scaling
        p_calibrated_nts, p_calibrated_nts_cal = temperature_scaling(Y_pred_prob_cal, yCalib, Y_pred_prob_val, yVal)
        rows=[]
    
        #find longest array:
        if len(p_calibrated_nts)>len(p_calibrated_nts_cal):
            for i in range(p_calibrated_nts.shape[0]):
                default_value = 0
                if(i >= len(p_calibrated_nts_cal)):
                    row = [p_calibrated_nts[i], default_value]
                else:
                    row = [p_calibrated_nts[i], p_calibrated_nts_cal[i]]
                rows.append(row)
        else:
            for i in range(p_calibrated_nts_cal.shape[0]):
                default_value = 0
                if(i >= len(p_calibrated_nts)):
                    row = [default_value , p_calibrated_nts_cal[i]]
                else:
                    row = [p_calibrated_nts[i], p_calibrated_nts_cal[i]]
                rows.append(row)

        

        file = f'other_results/raw_data_temperature_scaling_fold_{fold}_dataset_{DATASET}_threshold_{THRESHOLD}.csv'

        with open(file, 'w', newline='') as csvfile:
            writer= csv.writer(csvfile)
        
            #Header
            writer.writerow(['Index', 'p_calibrated_nts_test', 'p_calibrated_nts_cal'])
            row = []
            #write Data
            for i in range(len(rows)):
                #print(rows[i])
                writer.writerow([i+1, rows[i][0], rows[i][1]])
        """

        # Rel Dias
        # uncalibrated
        rel_diagram_bin_15_test_uncalibrated_test_static = evaluate_reliability_diagram(yVal, Y_pred_prob_val, None, 15, count, 'Op_1_uncal_test_15_static_validation', False)
        rel_diagram_bin_50_test_uncalibrated_test_static = evaluate_reliability_diagram(yVal, Y_pred_prob_val, None, 50, count, 'Op_1_uncal_test_50_static_validation', False)
        rel_diagram_bin_15_test_uncalibrated_test_interactive = evaluate_reliability_diagram(yVal, Y_pred_prob_val, None, 15, count, 'Op_1_uncal_test_15_interactive_validation', True)
        rel_diagram_bin_50_test_uncalibrated_test_interactive = evaluate_reliability_diagram(yVal, Y_pred_prob_val, None, 50, count, 'Op_1_uncal_test_50_interactive_validation', True)

        rel_diagram_bin_15_test_uncalibrated_cal_static = evaluate_reliability_diagram(yCalib, Y_pred_prob_cal, None, 15, count, 'Op_1_uncal_cal_15_static_validation', False)
        rel_diagram_bin_50_test_uncalibrated_cal_static = evaluate_reliability_diagram(yCalib, Y_pred_prob_cal, None, 50, count, 'Op_1_uncal_cal_50_static_validation', False)
        rel_diagram_bin_15_test_uncalibrated_cal_interactive = evaluate_reliability_diagram(yCalib, Y_pred_prob_cal, None, 15, count, 'Op_1_uncal_cal_15_interactive_validation', True)
        rel_diagram_bin_50_test_uncalibrated_cal_interactive = evaluate_reliability_diagram(yCalib, Y_pred_prob_cal, None, 50, count, 'Op_1_uncal_cal_50_interactive_validation', True)

        # Platt Scaling
        #"""
        # netcal platt
        rel_diagram_bin_15_test_nps_static_test = evaluate_reliability_diagram(yVal, p_calibrated_nps, None, 15, count, 'Op_3_nps_test_15_static_validation', False)
        rel_diagram_bin_50_test_nps_static_test = evaluate_reliability_diagram(yVal, p_calibrated_nps, None, 50, count, 'Op_3_nps_test_50_static_validation', False)
        rel_diagram_bin_15_test_nps_interactive_test = evaluate_reliability_diagram(yVal, p_calibrated_nps, None, 15, count, 'Op_3_nps_test_15_interactive_validation', True)
        rel_diagram_bin_50_test_nps_interactive_test = evaluate_reliability_diagram(yVal, p_calibrated_nps, None, 50, count, 'Op_3_nps_test_50_interactive_validation', True)
        rel_diagram_bin_15_test_nps_static_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nps_cal, None, 15, count, 'Op_3_nps_cal_15_static_validation', False)
        rel_diagram_bin_50_test_nps_static_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nps_cal, None, 50, count, 'Op_3_nps_cal_50_static_validation', False)
        rel_diagram_bin_15_test_nps_interactive_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nps_cal, None, 15, count, 'Op_3_nps_cal_15_interactive_validation', True)
        rel_diagram_bin_50_test_nps_interactive_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nps_cal, None, 50, count, 'Op_3_nps_cal_50_interactive_validation', True)
        """
        # Temperature Scaling
        rel_diagram_bin_15_test_nts_static_test = evaluate_reliability_diagram(yVal, p_calibrated_nts, None, 15, count, 'Op_6_nhb_test_15_static_validation', False)
        rel_diagram_bin_50_test_nts_static_test = evaluate_reliability_diagram(yVal, p_calibrated_nts, None, 50, count, 'Op_6_nhb_test_50_static_validation', False)
        rel_diagram_bin_15_test_nts_interactive_test = evaluate_reliability_diagram(yVal, p_calibrated_nts, None, 15, count, 'Op_6_nhb_test_15_interactive_validation', True)
        rel_diagram_bin_50_test_nts_interactive_test = evaluate_reliability_diagram(yVal, p_calibrated_nts, None, 50, count, 'Op_6_nhb_test_50_interactive_validation', True)
        rel_diagram_bin_15_test_nts_static_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nts_cal, None, 15, count, 'Op_6_nhb_cal_15_static_validation', False)
        rel_diagram_bin_50_test_nts_static_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nts_cal, None, 50, count, 'Op_6_nhb_cal_50_static_validation', False)
        rel_diagram_bin_15_test_nts_interactive_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nts_cal, None, 15, count, 'Op_6_nhb_cal_15_interactive_validation', True)
        rel_diagram_bin_50_test_nts_interactive_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nts_cal, None, 50, count, 'Op_6_nhb_cal_50_interactive_validation', True)
        #"""
        rel_dias_netcalplatt = rel_diagram_bin_15_test_uncalibrated_test_static, rel_diagram_bin_50_test_uncalibrated_test_static, rel_diagram_bin_15_test_uncalibrated_test_interactive, rel_diagram_bin_50_test_uncalibrated_test_interactive, rel_diagram_bin_15_test_uncalibrated_cal_static, rel_diagram_bin_50_test_uncalibrated_cal_static, rel_diagram_bin_15_test_uncalibrated_cal_interactive, rel_diagram_bin_50_test_uncalibrated_cal_interactive, rel_diagram_bin_15_test_nps_static_test, rel_diagram_bin_50_test_nps_static_test, rel_diagram_bin_15_test_nps_interactive_test, rel_diagram_bin_50_test_nps_interactive_test, rel_diagram_bin_15_test_nps_static_cal, rel_diagram_bin_50_test_nps_static_cal, rel_diagram_bin_15_test_nps_interactive_cal, rel_diagram_bin_50_test_nps_interactive_cal
        #rel_dias_temperature = rel_diagram_bin_15_test_uncalibrated_test_static, rel_diagram_bin_50_test_uncalibrated_test_static, rel_diagram_bin_15_test_uncalibrated_test_interactive, rel_diagram_bin_50_test_uncalibrated_test_interactive, rel_diagram_bin_15_test_uncalibrated_cal_static, rel_diagram_bin_50_test_uncalibrated_cal_static, rel_diagram_bin_15_test_uncalibrated_cal_interactive, rel_diagram_bin_50_test_uncalibrated_cal_interactive, rel_diagram_bin_15_test_nts_static_test, rel_diagram_bin_50_test_nts_static_test, rel_diagram_bin_15_test_nts_interactive_test, rel_diagram_bin_50_test_nts_interactive_test, rel_diagram_bin_15_test_nts_static_cal, rel_diagram_bin_50_test_nts_static_cal, rel_diagram_bin_15_test_nts_interactive_cal, rel_diagram_bin_50_test_nts_interactive_cal
        
        # netcal ECE
        ece = ECE(15)
        ece_score_uncalibrated = ece.measure(Y_pred_prob_val, yVal)
        ece_score_uncalibrated_cal = ece.measure(Y_pred_prob_cal, yCalib)

        ece_score_calibrated_nps = ece.measure(p_calibrated_nps, yVal)
        ece_score_calibrated_nps_cal = ece.measure(p_calibrated_nps_cal, yCalib)

        #ece_score_calibrated_nts = ece.measure(p_calibrated_nts, yVal)
        #ece_score_calibrated_nts_cal = ece.measure(p_calibrated_nts_cal, yCalib)

        ece_50 = ECE(50)
        ece_50_score_uncalibrated = ece_50.measure(Y_pred_prob_val, yVal)
        ece_50_score_uncalibrated_cal = ece_50.measure(Y_pred_prob_cal, yCalib)

        ece_50_score_calibrated_nps = ece_50.measure(p_calibrated_nps, yVal)
        ece_50_score_calibrated_nps_cal = ece_50.measure(p_calibrated_nps_cal, yCalib)

        #ece_50_score_calibrated_nts = ece_50.measure(p_calibrated_nts, yVal)
        #ece_50_score_calibrated_nts_cal = ece_50.measure(p_calibrated_nts_cal, yCalib)

        # AUC score
        auc_score_uncalibrated = roc_auc_score(yVal, Y_pred_prob_val)
        auc_score_uncalibrated_cal = roc_auc_score(yCalib, Y_pred_prob_cal)

        auc_score_calibrated_nps = roc_auc_score(yVal, p_calibrated_nps)
        auc_score_calibrated_nps_cal = roc_auc_score(yCalib, p_calibrated_nps_cal)

        #auc_score_calibrated_nts = roc_auc_score(yVal, p_calibrated_nts)
        #auc_score_calibrated_nts_cal = roc_auc_score(yCalib, p_calibrated_nts_cal)

        # Log Loss
        log_loss_uncalibrated = log_loss(yVal, Y_pred_prob_val)
        log_loss_uncalibrated_cal = log_loss(yCalib, Y_pred_prob_cal)

        log_loss_calibrated_nps = log_loss(yVal, p_calibrated_nps)
        log_loss_calibrated_nps_cal = log_loss(yCalib, p_calibrated_nps_cal)
        
        #log_loss_calibrated_nts = log_loss(yVal, p_calibrated_nts)
        #log_loss_calibrated_nts_cal = log_loss(yCalib, p_calibrated_nts_cal)

        # Brier Score
        brier_score_uncalibrated = calculate_brier_score(yVal, Y_pred_prob_val)
        brier_score_uncalibrated_cal = calculate_brier_score(yCalib, Y_pred_prob_cal)

        brier_score_calibrated_nps = calculate_brier_score(yVal, p_calibrated_nps)
        brier_score_calibrated_nps_cal = calculate_brier_score(yCalib, p_calibrated_nps_cal)
        
        #brier_score_calibrated_nts = calculate_brier_score(yVal, p_calibrated_nts)
        #brier_score_calibrated_nts_cal = calculate_brier_score(yCalib, p_calibrated_nts_cal)

        # zipping
        metrics_netcalplatt = (ece_score_uncalibrated, ece_score_uncalibrated_cal, ece_score_calibrated_nps, ece_score_calibrated_nps_cal, ece_50_score_uncalibrated, ece_50_score_uncalibrated_cal, ece_50_score_calibrated_nps, ece_50_score_calibrated_nps_cal, auc_score_uncalibrated, auc_score_uncalibrated_cal, auc_score_calibrated_nps, auc_score_calibrated_nps_cal, log_loss_uncalibrated, log_loss_uncalibrated_cal, log_loss_calibrated_nps, log_loss_calibrated_nps_cal, brier_score_uncalibrated, brier_score_uncalibrated_cal, brier_score_calibrated_nps, brier_score_calibrated_nps_cal)
        #metrics_temperature = (ece_score_uncalibrated, ece_score_uncalibrated_cal, ece_score_calibrated_nts, ece_score_calibrated_nts_cal, ece_50_score_uncalibrated, ece_50_score_uncalibrated_cal, ece_50_score_calibrated_nts, ece_50_score_calibrated_nts_cal, auc_score_uncalibrated, auc_score_uncalibrated_cal, auc_score_calibrated_nts, auc_score_calibrated_nts_cal, log_loss_uncalibrated, log_loss_uncalibrated_cal, log_loss_calibrated_nts, log_loss_calibrated_nts_cal, brier_score_uncalibrated, brier_score_uncalibrated_cal, brier_score_calibrated_nts, brier_score_calibrated_nts_cal)

        metrics_names_netcalplatt = ["ece_score_uncalibrated", "ece_score_uncalibrated_cal", "ece_score_calibrated_nps", "ece_score_calibrated_nps_cal", "ece_50_score_uncalibrated", "ece_50_score_uncalibrated_cal", "ece_50_score_calibrated_nps", "ece_50_score_calibrated_nps_cal", "auc_score_uncalibrated", "auc_score_uncalibrated_cal", "auc_score_calibrated_nps", "auc_score_calibrated_nps_cal", "log_loss_uncalibrated", "log_loss_uncalibrated_cal", "log_loss_calibrated_nps", "log_loss_calibrated_nps_cal", "brier_score_uncalibrated", "brier_score_uncalibrated_cal", "brier_score_calibrated_nps", "brier_score_calibrated_nps_cal"]
        #metrics_names_temperature = ["ece_score_uncalibrated", "ece_score_uncalibrated_cal", "ece_score_calibrated_nts", "ece_score_calibrated_nts_cal", "ece_50_score_uncalibrated", "ece_50_score_uncalibrated_cal", "ece_50_score_calibrated_nts", "ece_50_score_calibrated_nts_cal", "auc_score_uncalibrated", "auc_score_uncalibrated_cal", "auc_score_calibrated_nts", "auc_score_calibrated_nts_cal", "log_loss_uncalibrated", "log_loss_uncalibrated_cal",  "log_loss_calibrated_nts", "log_loss_calibrated_nts_cal", "brier_score_uncalibrated", "brier_score_uncalibrated_cal", "brier_score_calibrated_nts", "brier_score_calibrated_nts_cal"]
        
        rel_dias_netcalplatt = rel_diagram_bin_15_test_uncalibrated_test_static, rel_diagram_bin_50_test_uncalibrated_test_static, rel_diagram_bin_15_test_uncalibrated_test_interactive, rel_diagram_bin_50_test_uncalibrated_test_interactive, rel_diagram_bin_15_test_uncalibrated_cal_static, rel_diagram_bin_50_test_uncalibrated_cal_static, rel_diagram_bin_15_test_uncalibrated_cal_interactive, rel_diagram_bin_50_test_uncalibrated_cal_interactive, rel_diagram_bin_15_test_nps_static_test, rel_diagram_bin_50_test_nps_static_test, rel_diagram_bin_15_test_nps_interactive_test, rel_diagram_bin_50_test_nps_interactive_test, rel_diagram_bin_15_test_nps_static_cal, rel_diagram_bin_50_test_nps_static_cal, rel_diagram_bin_15_test_nps_interactive_cal, rel_diagram_bin_50_test_nps_interactive_cal
        #rel_dias_temperature = rel_diagram_bin_15_test_uncalibrated_test_static, rel_diagram_bin_50_test_uncalibrated_test_static, rel_diagram_bin_15_test_uncalibrated_test_interactive, rel_diagram_bin_50_test_uncalibrated_test_interactive, rel_diagram_bin_15_test_uncalibrated_cal_static, rel_diagram_bin_50_test_uncalibrated_cal_static, rel_diagram_bin_15_test_uncalibrated_cal_interactive, rel_diagram_bin_50_test_uncalibrated_cal_interactive, rel_diagram_bin_15_test_nts_static_test, rel_diagram_bin_50_test_nts_static_test, rel_diagram_bin_15_test_nts_interactive_test, rel_diagram_bin_50_test_nts_interactive_test, rel_diagram_bin_15_test_nts_static_cal, rel_diagram_bin_50_test_nts_static_cal, rel_diagram_bin_15_test_nts_interactive_cal, rel_diagram_bin_50_test_nts_interactive_cal
        
        rel_dias_names_netcalplatt = ("rel_diagram_bin_15_test_uncalibrated_test_static", "rel_diagram_bin_50_test_uncalibrated_test_static", "rel_diagram_bin_15_test_uncalibrated_test_interactive", "rel_diagram_bin_50_test_uncalibrated_test_interactive", "rel_diagram_bin_15_test_uncalibrated_cal_static", "rel_diagram_bin_50_test_uncalibrated_cal_static", "rel_diagram_bin_15_test_uncalibrated_cal_interactive", "rel_diagram_bin_50_test_uncalibrated_cal_interactive", "rel_diagram_bin_15_test_nps_static_test", "rel_diagram_bin_50_test_nps_static_test", "rel_diagram_bin_15_test_nps_interactive_test", "rel_diagram_bin_50_test_nps_interactive_test", "rel_diagram_bin_15_test_nps_static_cal", "rel_diagram_bin_50_test_nps_static_cal", "rel_diagram_bin_15_test_nps_interactive_cal", "rel_diagram_bin_50_test_nps_interactive_cal")
        #rel_dias_names_temperature = ("rel_diagram_bin_15_test_uncalibrated_test_static", "rel_diagram_bin_50_test_uncalibrated_test_static", "rel_diagram_bin_15_test_uncalibrated_test_interactive", "rel_diagram_bin_50_test_uncalibrated_test_interactive", "rel_diagram_bin_15_test_uncalibrated_cal_static", "rel_diagram_bin_50_test_uncalibrated_cal_static", "rel_diagram_bin_15_test_uncalibrated_cal_interactive", "rel_diagram_bin_50_test_uncalibrated_cal_interactive", "rel_diagram_bin_15_test_nts_static_test", "rel_diagram_bin_50_test_nts_static_test", "rel_diagram_bin_15_test_nts_interactive_test", "rel_diagram_bin_50_test_nts_interactive_test", "rel_diagram_bin_15_test_nts_static_cal", "rel_diagram_bin_50_test_nts_static_cal", "rel_diagram_bin_15_test_nts_interactive_cal", "rel_diagram_bin_50_test_nts_interactive_cal")

        custom_eces = []
        custom_eces_names = []
        custom_mces = []
        custom_mces_names = []
        for index, rel_dia in enumerate(rel_dias_netcalplatt):
            ece_name = f"Custom_ECE_{rel_dias_names_netcalplatt[index]}"
            custom_ece = calculate_ece(rel_dia[0], rel_dia[1])
            custom_eces.append(custom_ece)
            custom_eces_names.append(ece_name)
            mce_name = f"Custom_MCE_{rel_dias_names_netcalplatt[index]}"
            custom_mce = calculate_mce(rel_dia[0], rel_dia[1])
            custom_mces.append(custom_mce)
            custom_mces_names.append(mce_name)

        calculated_metrics = metrics_netcalplatt, metrics_names_netcalplatt, custom_eces, custom_eces_names, custom_mces, custom_mces_names
        calibration_metrics.append(calculated_metrics)
            
        #measure:
        filename = f'other_results/eval_validation_training_fold_{count}_evaluation_{DATASET}_{CALIBRATIONMODE}_threshold_{THRESHOLD}.csv'
        #y_prediction_probabilities, y_true_labels, filename, iteration, last=False
        extract_calibration_metrics(Y_pred_prob_val, yVal, filename, fold=count, last=False)
        matrix = confusion_matrix(y_true=yVal, y_pred=Y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
        disp.plot().figure_.savefig(f'other_results/eval_validation_training_fold_{count}_evaluation_{DATASET}_confusion_matrix_{CALIBRATIONMODE}_threshold_{THRESHOLD}.png')
        file = f'other_results/eval_calibration_fold_{fold}_{DATASET}_{CALIBRATIONMODE}_threshold_{THRESHOLD}_aggregated_scaling_metrics.csv'
        write_calibrated_metrics(calibration_metrics, file)
        count+=1
    

    # Use Test Set and Measure everything    
    #model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:,1]
    #Threshold 0.5
    #y_pred_labels = model.predict(X_test)
    y_pred_labels = (y_pred >= THRESHOLD).astype(int)
    acc, prc, rc, f1, auc_ = evaluation_metrics(np.asarray(y_test), np.asarray(y_pred))
    file = f'accuracy/acc_prc_rc_f1_auc_testset_dataset_{DATASET}_threshold_{THRESHOLD}_scaling_{CALIBRATIONMODE}.csv'
    with open(file, 'w', newline='') as csvfile:
        writer= csv.writer(csvfile)
        #Header
        writer.writerow(['acc', 'prc', 'rc', 'f1', 'auc_'])
        #write Data
        writer.writerow([acc, prc, rc, f1, auc_])

    rows=[]
    for i in range(len(X_test)):
        row = [X_test[i], y_test[i], y_pred[i], y_pred_labels[i]]
        rows.append(row)

    file = f'other_results/raw_data_test_dataset_{DATASET}_threshold_{THRESHOLD}.csv'

    with open(file, 'w', newline='') as csvfile:
        writer= csv.writer(csvfile)
        
        #Header
        writer.writerow(['Index', 'Input', 'True_Label', 'Predicted_probability', 'Predicted_Label'])
        row = []
        #write Data
        for i in range(len(rows)):
            #print(rows[i])
            writer.writerow([i+1, rows[i][0], rows[i][1], rows[i][2], rows[i][3]])

    # Need: Predictions (Y_pred_prob_cal), Labels (yCalib)
    # Logits
    preds_test_inverted = np.asarray(invert_sigmoid_scores(Y_pred))

    # 1. Platt Scaling TODO
    # Custom Platt
    #p_calibrated_inv_ps, p_calibrated_inv_ps_cal = custom_platt_scaling(preds_cal_inverted, yCalib, preds_test_inverted, y_test)
    #"""
    # TODO Netcal Platt
    p_calibrated_nps, p_calibrated_nps_cal = platt_scaling(Y_pred_prob_cal, yCalib, y_pred, y_test)
    rows=[]

    #find longest array:
    if len(p_calibrated_nps)>len(p_calibrated_nps_cal):
        for i in range(p_calibrated_nps.shape[0]):
            default_value = 0
            if(i >= len(p_calibrated_nps_cal)):
                row = [p_calibrated_nps[i], default_value]
            else:
                row = [p_calibrated_nps[i], p_calibrated_nps_cal[i]]
            rows.append(row)
    else:
        for i in range(p_calibrated_nps_cal.shape[0]):
            default_value = 0
            if(i >= len(p_calibrated_nps)):
                row = [default_value , p_calibrated_nps_cal[i]]
            else:
                row = [p_calibrated_nps[i], p_calibrated_nps_cal[i]]
            rows.append(row)

        

    file = f'other_results/raw_data_platt_scaling_testset.csv'

    with open(file, 'w', newline='') as csvfile:
        writer= csv.writer(csvfile)
        
        #Header
        writer.writerow(['Index', 'p_calibrated_nps', 'p_calibrated_nps_cal'])
        row = []
        #write Data
        for i in range(len(rows)):
            writer.writerow([i+1, rows[i][0], rows[i][1]])

        
    """
    # TODO 2. Temperature Scaling
    p_calibrated_nts, p_calibrated_nts_cal = temperature_scaling(Y_pred_prob_cal, yCalib, y_pred, y_test)

    rows=[]

    #find longest array:
    if len(p_calibrated_nts)>len(p_calibrated_nts_cal):
        for i in range(p_calibrated_nts.shape[0]):
            default_value = 0
            if(i >= len(p_calibrated_nts_cal)):
                row = [p_calibrated_nts[i], default_value]
            else:
                row = [p_calibrated_nts[i], p_calibrated_nts_cal[i]]
            rows.append(row)
    else:
        for i in range(p_calibrated_nts_cal.shape[0]):
            default_value = 0
            if(i >= len(p_calibrated_nts)):
                row = [default_value , p_calibrated_nts_cal[i]]
            else:
                row = [p_calibrated_nts[i], p_calibrated_nts_cal[i]]
            rows.append(row)

        

    file = f'other_results/raw_data_temperature_scaling_testset.csv'

    with open(file, 'w', newline='') as csvfile:
        writer= csv.writer(csvfile)
        
        #Header
        writer.writerow(['Index', 'p_calibrated_nts_test', 'p_calibrated_nts_cal'])
        row = []
        #write Data
        for i in range(len(rows)):
            writer.writerow([i+1, rows[i][0], rows[i][1]])
    #"""

    
    # Rel Dias
    # uncalibrated
    rel_diagram_bin_15_test_uncalibrated_test_static = evaluate_reliability_diagram(y_test, y_pred, None, 15, 100, 'Op_1_uncal_test_15_static_test', False)
    rel_diagram_bin_50_test_uncalibrated_test_static = evaluate_reliability_diagram(y_test, y_pred, None, 50, 100, 'Op_1_uncal_test_50_static_test', False)
    rel_diagram_bin_15_test_uncalibrated_test_interactive = evaluate_reliability_diagram(y_test, y_pred, None, 15, 100, 'Op_1_uncal_test_15_interactive_test', True)
    rel_diagram_bin_50_test_uncalibrated_test_interactive = evaluate_reliability_diagram(y_test, y_pred, None, 50, 100, 'Op_1_uncal_test_50_interactive_test', True)

    rel_diagram_bin_15_test_uncalibrated_cal_static = evaluate_reliability_diagram(yCalib, Y_pred_prob_cal, None, 15, 100, 'Op_1_uncal_cal_15_static_test', False)
    rel_diagram_bin_50_test_uncalibrated_cal_static = evaluate_reliability_diagram(yCalib, Y_pred_prob_cal, None, 50, 100, 'Op_1_uncal_cal_50_static_test', False)
    rel_diagram_bin_15_test_uncalibrated_cal_interactive = evaluate_reliability_diagram(yCalib, Y_pred_prob_cal, None, 15, 100, 'Op_1_uncal_cal_15_interactive_test', True)
    rel_diagram_bin_50_test_uncalibrated_cal_interactive = evaluate_reliability_diagram(yCalib, Y_pred_prob_cal, None, 50, 100, 'Op_1_uncal_cal_50_interactive_test', True)
    #"""
    # netcal platt
    rel_diagram_bin_15_test_nps_static_test = evaluate_reliability_diagram(y_test, p_calibrated_nps, None, 15, 100, 'Op_3_nps_test_15_static_test', False)
    rel_diagram_bin_50_test_nps_static_test = evaluate_reliability_diagram(y_test, p_calibrated_nps, None, 50, 100, 'Op_3_nps_test_50_static_test', False)
    rel_diagram_bin_15_test_nps_interactive_test = evaluate_reliability_diagram(y_test, p_calibrated_nps, None, 15, 100, 'Op_3_nps_test_15_interactive_test', True)
    rel_diagram_bin_50_test_nps_interactive_test = evaluate_reliability_diagram(y_test, p_calibrated_nps, None, 50, 100, 'Op_3_nps_test_50_interactive_test', True)
    rel_diagram_bin_15_test_nps_static_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nps_cal, None, 15, 100, 'Op_3_nps_cal_15_static_test', False)
    rel_diagram_bin_50_test_nps_static_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nps_cal, None, 50, 100, 'Op_3_nps_cal_50_static_test', False)
    rel_diagram_bin_15_test_nps_interactive_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nps_cal, None, 15, 100, 'Op_3_nps_cal_15_interactive_test', True)
    rel_diagram_bin_50_test_nps_interactive_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nps_cal, None, 50, 100, 'Op_3_nps_cal_50_interactive_test', True)
    """
    # Temperature Scaling
    rel_diagram_bin_15_test_nts_static_test = evaluate_reliability_diagram(y_test, p_calibrated_nts, None, 15, 100, 'Op_6_nhb_test_15_static_test', False)
    rel_diagram_bin_50_test_nts_static_test = evaluate_reliability_diagram(y_test, p_calibrated_nts, None, 50, 100, 'Op_6_nhb_test_50_static_test', False)
    rel_diagram_bin_15_test_nts_interactive_test = evaluate_reliability_diagram(y_test, p_calibrated_nts, None, 15, 100, 'Op_6_nhb_test_15_interactive_test', True)
    rel_diagram_bin_50_test_nts_interactive_test = evaluate_reliability_diagram(y_test, p_calibrated_nts, None, 50, 100, 'Op_6_nhb_test_50_interactive', True)
    rel_diagram_bin_15_test_nts_static_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nts_cal, None, 15, 100, 'Op_6_nhb_cal_15_static_test', False)
    rel_diagram_bin_50_test_nts_static_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nts_cal, None, 50, 100, 'Op_6_nhb_cal_50_static_test', False)
    rel_diagram_bin_15_test_nts_interactive_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nts_cal, None, 15, 100, 'Op_6_nhb_cal_15_interactive_test', True)
    rel_diagram_bin_50_test_nts_interactive_cal = evaluate_reliability_diagram(yCalib, p_calibrated_nts_cal, None, 50, 100, 'Op_6_nhb_cal_50_interactive_test', True)
    #"""
    rel_dias_netcalplatt = rel_diagram_bin_15_test_uncalibrated_test_static, rel_diagram_bin_50_test_uncalibrated_test_static, rel_diagram_bin_15_test_uncalibrated_test_interactive, rel_diagram_bin_50_test_uncalibrated_test_interactive, rel_diagram_bin_15_test_uncalibrated_cal_static, rel_diagram_bin_50_test_uncalibrated_cal_static, rel_diagram_bin_15_test_uncalibrated_cal_interactive, rel_diagram_bin_50_test_uncalibrated_cal_interactive, rel_diagram_bin_15_test_nps_static_test, rel_diagram_bin_50_test_nps_static_test, rel_diagram_bin_15_test_nps_interactive_test, rel_diagram_bin_50_test_nps_interactive_test, rel_diagram_bin_15_test_nps_static_cal, rel_diagram_bin_50_test_nps_static_cal, rel_diagram_bin_15_test_nps_interactive_cal, rel_diagram_bin_50_test_nps_interactive_cal
    #rel_dias_temperature = rel_diagram_bin_15_test_uncalibrated_test_static, rel_diagram_bin_50_test_uncalibrated_test_static, rel_diagram_bin_15_test_uncalibrated_test_interactive, rel_diagram_bin_50_test_uncalibrated_test_interactive, rel_diagram_bin_15_test_uncalibrated_cal_static, rel_diagram_bin_50_test_uncalibrated_cal_static, rel_diagram_bin_15_test_uncalibrated_cal_interactive, rel_diagram_bin_50_test_uncalibrated_cal_interactive, rel_diagram_bin_15_test_nts_static_test, rel_diagram_bin_50_test_nts_static_test, rel_diagram_bin_15_test_nts_interactive_test, rel_diagram_bin_50_test_nts_interactive_test, rel_diagram_bin_15_test_nts_static_cal, rel_diagram_bin_50_test_nts_static_cal, rel_diagram_bin_15_test_nts_interactive_cal, rel_diagram_bin_50_test_nts_interactive_cal
    
    # netcal ECE
    ece = ECE(15)
    ece_score_uncalibrated = ece.measure(y_pred, y_test)
    ece_score_uncalibrated_cal = ece.measure(Y_pred_prob_cal, yCalib)

    ece_score_calibrated_nps = ece.measure(p_calibrated_nps, y_test)
    ece_score_calibrated_nps_cal = ece.measure(p_calibrated_nps_cal, yCalib)

    #ece_score_calibrated_nts = ece.measure(p_calibrated_nts, y_test)
    #ece_score_calibrated_nts_cal = ece.measure(p_calibrated_nts_cal, yCalib)

    ece_50 = ECE(50)
    ece_50_score_uncalibrated = ece_50.measure(y_pred, y_test)
    ece_50_score_uncalibrated_cal = ece_50.measure(Y_pred_prob_cal, yCalib)

    ece_50_score_calibrated_nps = ece_50.measure(p_calibrated_nps, y_test)
    ece_50_score_calibrated_nps_cal = ece_50.measure(p_calibrated_nps_cal, yCalib)

    #ece_50_score_calibrated_nts = ece_50.measure(p_calibrated_nts, y_test)
    #ece_50_score_calibrated_nts_cal = ece_50.measure(p_calibrated_nts_cal, yCalib)

    # AUC score
    auc_score_uncalibrated = roc_auc_score(y_test, y_pred)
    auc_score_uncalibrated_cal = roc_auc_score(yCalib, Y_pred_prob_cal)

    auc_score_calibrated_nps = roc_auc_score(y_test, p_calibrated_nps)
    auc_score_calibrated_nps_cal = roc_auc_score(yCalib, p_calibrated_nps_cal)

    #auc_score_calibrated_nts = roc_auc_score(y_test, p_calibrated_nts)
    #auc_score_calibrated_nts_cal = roc_auc_score(yCalib, p_calibrated_nts_cal)

    # Log Loss
    log_loss_uncalibrated = log_loss(y_test, y_pred)
    log_loss_uncalibrated_cal = log_loss(yCalib, Y_pred_prob_cal)

    log_loss_calibrated_nps = log_loss(y_test, p_calibrated_nps)
    log_loss_calibrated_nps_cal = log_loss(yCalib, p_calibrated_nps_cal)
    
    #log_loss_calibrated_nts = log_loss(y_test, p_calibrated_nts)
    #log_loss_calibrated_nts_cal = log_loss(yCalib, p_calibrated_nts_cal)

    # Brier Score
    brier_score_uncalibrated = calculate_brier_score(y_test, y_pred)
    brier_score_uncalibrated_cal = calculate_brier_score(yCalib, Y_pred_prob_cal)

    brier_score_calibrated_nps = calculate_brier_score(y_test, p_calibrated_nps)
    brier_score_calibrated_nps_cal = calculate_brier_score(yCalib, p_calibrated_nps_cal)
    
    #brier_score_calibrated_nts = calculate_brier_score(y_test, p_calibrated_nts)
    #brier_score_calibrated_nts_cal = calculate_brier_score(yCalib, p_calibrated_nts_cal)

    # zipping
    metrics_netcalplatt = (ece_score_uncalibrated, ece_score_uncalibrated_cal, ece_score_calibrated_nps, ece_score_calibrated_nps_cal, ece_50_score_uncalibrated, ece_50_score_uncalibrated_cal, ece_50_score_calibrated_nps, ece_50_score_calibrated_nps_cal, auc_score_uncalibrated, auc_score_uncalibrated_cal, auc_score_calibrated_nps, auc_score_calibrated_nps_cal, log_loss_uncalibrated, log_loss_uncalibrated_cal, log_loss_calibrated_nps, log_loss_calibrated_nps_cal, brier_score_uncalibrated, brier_score_uncalibrated_cal, brier_score_calibrated_nps, brier_score_calibrated_nps_cal)
    #metrics_temperature = (ece_score_uncalibrated, ece_score_uncalibrated_cal, ece_score_calibrated_nts, ece_score_calibrated_nts_cal, ece_50_score_uncalibrated, ece_50_score_uncalibrated_cal, ece_50_score_calibrated_nts, ece_50_score_calibrated_nts_cal, auc_score_uncalibrated, auc_score_uncalibrated_cal, auc_score_calibrated_nts, auc_score_calibrated_nts_cal, log_loss_uncalibrated, log_loss_uncalibrated_cal, log_loss_calibrated_nts, log_loss_calibrated_nts_cal, brier_score_uncalibrated, brier_score_uncalibrated_cal, brier_score_calibrated_nts, brier_score_calibrated_nts_cal)

    metrics_names_netcalplatt = ["ece_score_uncalibrated", "ece_score_uncalibrated_cal", "ece_score_calibrated_nps", "ece_score_calibrated_nps_cal", "ece_50_score_uncalibrated", "ece_50_score_uncalibrated_cal", "ece_50_score_calibrated_nps", "ece_50_score_calibrated_nps_cal", "auc_score_uncalibrated", "auc_score_uncalibrated_cal", "auc_score_calibrated_nps", "auc_score_calibrated_nps_cal", "log_loss_uncalibrated", "log_loss_uncalibrated_cal", "log_loss_calibrated_nps", "log_loss_calibrated_nps_cal", "brier_score_uncalibrated", "brier_score_uncalibrated_cal", "brier_score_calibrated_nps", "brier_score_calibrated_nps_cal"]
    #metrics_names_temperature = ["ece_score_uncalibrated", "ece_score_uncalibrated_cal", "ece_score_calibrated_nts", "ece_score_calibrated_nts_cal", "ece_50_score_uncalibrated", "ece_50_score_uncalibrated_cal", "ece_50_score_calibrated_nts", "ece_50_score_calibrated_nts_cal", "auc_score_uncalibrated", "auc_score_uncalibrated_cal", "auc_score_calibrated_nts", "auc_score_calibrated_nts_cal", "log_loss_uncalibrated", "log_loss_uncalibrated_cal",  "log_loss_calibrated_nts", "log_loss_calibrated_nts_cal", "brier_score_uncalibrated", "brier_score_uncalibrated_cal", "brier_score_calibrated_nts", "brier_score_calibrated_nts_cal"]
    
    rel_dias_netcalplatt = rel_diagram_bin_15_test_uncalibrated_test_static, rel_diagram_bin_50_test_uncalibrated_test_static, rel_diagram_bin_15_test_uncalibrated_test_interactive, rel_diagram_bin_50_test_uncalibrated_test_interactive, rel_diagram_bin_15_test_uncalibrated_cal_static, rel_diagram_bin_50_test_uncalibrated_cal_static, rel_diagram_bin_15_test_uncalibrated_cal_interactive, rel_diagram_bin_50_test_uncalibrated_cal_interactive, rel_diagram_bin_15_test_nps_static_test, rel_diagram_bin_50_test_nps_static_test, rel_diagram_bin_15_test_nps_interactive_test, rel_diagram_bin_50_test_nps_interactive_test, rel_diagram_bin_15_test_nps_static_cal, rel_diagram_bin_50_test_nps_static_cal, rel_diagram_bin_15_test_nps_interactive_cal, rel_diagram_bin_50_test_nps_interactive_cal
    #rel_dias_temperature = rel_diagram_bin_15_test_uncalibrated_test_static, rel_diagram_bin_50_test_uncalibrated_test_static, rel_diagram_bin_15_test_uncalibrated_test_interactive, rel_diagram_bin_50_test_uncalibrated_test_interactive, rel_diagram_bin_15_test_uncalibrated_cal_static, rel_diagram_bin_50_test_uncalibrated_cal_static, rel_diagram_bin_15_test_uncalibrated_cal_interactive, rel_diagram_bin_50_test_uncalibrated_cal_interactive, rel_diagram_bin_15_test_nts_static_test, rel_diagram_bin_50_test_nts_static_test, rel_diagram_bin_15_test_nts_interactive_test, rel_diagram_bin_50_test_nts_interactive_test, rel_diagram_bin_15_test_nts_static_cal, rel_diagram_bin_50_test_nts_static_cal, rel_diagram_bin_15_test_nts_interactive_cal, rel_diagram_bin_50_test_nts_interactive_cal
    
    rel_dias_names_netcalplatt = ("rel_diagram_bin_15_test_uncalibrated_test_static", "rel_diagram_bin_50_test_uncalibrated_test_static", "rel_diagram_bin_15_test_uncalibrated_test_interactive", "rel_diagram_bin_50_test_uncalibrated_test_interactive", "rel_diagram_bin_15_test_uncalibrated_cal_static", "rel_diagram_bin_50_test_uncalibrated_cal_static", "rel_diagram_bin_15_test_uncalibrated_cal_interactive", "rel_diagram_bin_50_test_uncalibrated_cal_interactive", "rel_diagram_bin_15_test_nps_static_test", "rel_diagram_bin_50_test_nps_static_test", "rel_diagram_bin_15_test_nps_interactive_test", "rel_diagram_bin_50_test_nps_interactive_test", "rel_diagram_bin_15_test_nps_static_cal", "rel_diagram_bin_50_test_nps_static_cal", "rel_diagram_bin_15_test_nps_interactive_cal", "rel_diagram_bin_50_test_nps_interactive_cal")
    #rel_dias_names_temperature = ("rel_diagram_bin_15_test_uncalibrated_test_static", "rel_diagram_bin_50_test_uncalibrated_test_static", "rel_diagram_bin_15_test_uncalibrated_test_interactive", "rel_diagram_bin_50_test_uncalibrated_test_interactive", "rel_diagram_bin_15_test_uncalibrated_cal_static", "rel_diagram_bin_50_test_uncalibrated_cal_static", "rel_diagram_bin_15_test_uncalibrated_cal_interactive", "rel_diagram_bin_50_test_uncalibrated_cal_interactive", "rel_diagram_bin_15_test_nts_static_test", "rel_diagram_bin_50_test_nts_static_test", "rel_diagram_bin_15_test_nts_interactive_test", "rel_diagram_bin_50_test_nts_interactive_test", "rel_diagram_bin_15_test_nts_static_cal", "rel_diagram_bin_50_test_nts_static_cal", "rel_diagram_bin_15_test_nts_interactive_cal", "rel_diagram_bin_50_test_nts_interactive_cal")

    custom_eces = []
    custom_eces_names = []
    custom_mces = []
    custom_mces_names = []
    for index, rel_dia in enumerate(rel_dias_netcalplatt):
        ece_name = f"Custom_ECE_{rel_dias_names_netcalplatt[index]}"
        custom_ece = calculate_ece(rel_dia[0], rel_dia[1])
        custom_eces.append(custom_ece)
        custom_eces_names.append(ece_name)
        mce_name = f"Custom_MCE_{rel_dias_names_netcalplatt[index]}"
        custom_mce = calculate_mce(rel_dia[0], rel_dia[1])
        custom_mces.append(custom_mce)
        custom_mces_names.append(mce_name)

    calculated_metrics = metrics_netcalplatt, metrics_names_netcalplatt, custom_eces, custom_eces_names, custom_mces, custom_mces_names
    calibration_metrics.append(calculated_metrics)

        

    matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
    disp.plot().figure_.savefig(f'other_results/eval_calibration_testset_{DATASET}_confusion_matrix_{CALIBRATIONMODE}_threshold_{THRESHOLD}.png')
    #print(y_pred)
    filename = f'other_results/eval_calibration_testset_{DATASET}_{CALIBRATIONMODE}_threshold_{THRESHOLD}.csv'
    extract_calibration_metrics(y_pred, y_test, filename, fold=100, last=True)
    file = f'other_results/eval_calibration_testset_{DATASET}_{CALIBRATIONMODE}_threshold_{THRESHOLD}_aggregated_scaling_metrics.csv'
    write_calibrated_metrics(calibration_metrics, file)


def baseline_algorithm(train, test, algorithm, only=False):
    _, y_train, X_train = train
    _, y_test, X_test = test
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    acc, prc, rc, f1, auc_ = 0, 0, 0, 0, 0
    if algorithm == 'lr':
        starttime = time.time()
        #model = LogisticRegression(max_iter=7000).fit(X_train, y_train)
        model = LogisticRegression(max_iter=7000)
        kf = KFold(n_splits = 10, shuffle=True, random_state=42)
        fold_accuracies=[]
        for train_index, val_index in kf.split(X_train):
            x_train, x_val = X_train[train_index], X_train[val_index]
            Y_train, Y_val = y_train[train_index], y_train[val_index]
            model.fit(x_train, Y_train)
            Y_pred = model.predict(x_val)
            accuracy = accuracy_score(Y_val, Y_pred)
            fold_accuracies.append(accuracy)
        #cv= cross_validate(model, X_train, y_train, scoring=['accuracy', 'precision'], cv=10, return_train_score=True)
        #print("Cross-Vaidation: ", cv)
        model.fit(X_train, y_train)
        endtime = time.time()
        dtime = endtime - starttime
        print("Train Time: %.8s s" % dtime)  #
      
        starttime = time.time()
        y_pred = model.predict_proba(X_test)[:, 1]
        y_pred_labels = model.predict(X_test)
        endtime = time.time()
        dtime = endtime - starttime
        print("Eval Time: %.8s s" % dtime)  #
        #matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_labels)
        #disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
        #disp.plot()
        #plt.show()
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        if only and not "cross" in args.data:
            auc_ = train_and_evl(X_train, y_train, args)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
      
    elif algorithm =='dbn':
        y_pred = DBN_JIT(X_train, y_train, X_test, y_test)
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        acc, prc, rc, f1 = 0, 0, 0, 0
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    else:
        print('You need to give the correct algorithm name')
        return

    return y_test, y_pred 


def save_result(labels, predicts, path):
  
    results = []
    for lable, predict in zip(labels, predicts):
        results.append('{}\t{}\n'.format(lable, predict))
   
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(results)    
        


if __name__ == '__main__':

    args = parser.parse_args()
    #print(args)
    save_path = 'result/{}/{}_{}_{}.result'.format(args.project, args.project, args.algorithm, args.data.replace("/","_"))
    only = True if args.only else False
    #print("In Baseline py")
    #print("only: ", only)
    #print("save_path:", save_path)
    if args.algorithm == 'la':
        args.algorithm = 'lr'
        args.only = ['la']
    if "all" in args.only:
        args.only.remove("all")

    train, test = load_yasu_data(args)
    
    #print("Algo: ", args.algorithm, "Only: ", only)    
    #labels, predicts = baseline_algorithm(train=train, test=test, algorithm=args.algorithm, only=only)
    #baseline_crossVal_Calibrate(train=train, test=test)
    baseline_crossval_with_Calibration_Fold(train=train, test=test)
    #if not only:
   
        #save_result(labels, predicts, save_path)
