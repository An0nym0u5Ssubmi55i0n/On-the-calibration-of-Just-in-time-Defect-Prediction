from model import DeepJIT
from utils import mini_batches_test
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import tqdm
from measure import calibration_score
from calibration import TemperatureScaling
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import numpy as np
import trustscore_evaluation
import pickle

def evaluation_model(data, params, inv_temp=None, p_model=None, signals=None):
    pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_test(X_msg=pad_msg, X_code=pad_code, Y=labels)

    # Define data for determining distance to classes
    valid_pad_code = pad_code.reshape(len(pad_code), -1)
    X_test = np.column_stack((pad_msg, valid_pad_code))

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # set up parameters
    print("Parameter test: ", params.calibration)
    if params.predict is True:
        params.cuda = (not params.no_cuda) and torch.cuda.is_available()
        del params.no_cuda
        params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

        # Calibration settings
        print("Parameter test: ", params.calibration)
        inv_temp = torch.load(params.temp)
        with open(params.platt, 'rb') as f:
            p_model = pickle.load(f)
        signals = []
        with open(params.trustscore, 'rb') as f:
            ats_model = pickle.load(f)
        signals.append(ats_model)

    model = DeepJIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):
            pad_msg, pad_code, label = batch
            if torch.cuda.is_available():
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(label)
            else:
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():
                predict = model.forward(pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                predict = model.forward(pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()
    calibration_score(all_label, all_predict)
    auc_score = roc_auc_score(y_true=all_label, y_score=all_predict)
    if params.train is True:
        print('Validation data -- AUC score:', auc_score)
    elif params.predict is True:
        print('Test data -- AUC score:', auc_score)

    all_tprob = all_predict
    if params.calibration_opt is True:
        # Perform Optimization of Temperature Scaling
        calib_model = TemperatureScaling()
        calib_model.fit(all_predict, all_label)
        all_tprob = calib_model.calibrate_prob(all_predict)
        calibration_score(all_label, all_tprob)
        calib_auc_score = roc_auc_score(y_true=all_label, y_score=all_tprob)
        print('Calibration data (Temp) -- AUC score:', calib_auc_score)

        # Perform Optimization of Platt Scaling
        platt_model = LogisticRegression()
        np_predict = np.array(all_predict)
        all_logit = np.log(np_predict / (1 - np_predict))
        platt_model.fit(all_logit.reshape(-1, 1), all_label)
        all_pprob = platt_model.predict_proba(all_logit.reshape(-1, 1))[:, 1]
        calibration_score(all_label, all_pprob)
        calib_auc_score = roc_auc_score(y_true=all_label, y_score=all_pprob)
        print('Calibration data (Platt) -- AUC score:', calib_auc_score)
        return calib_model.inv_temperature, platt_model

    elif params.calibration is True:
        # Perform Temperature Scaling
        calib_model = TemperatureScaling()
        calib_model.inv_temperature = inv_temp
        all_tprob = calib_model.calibrate_prob(all_predict)
        calibration_score(all_label, all_tprob)
        calib_auc_score = roc_auc_score(y_true=all_label, y_score=all_tprob)
        if params.train is True:
            print('Validation data (Temp) -- AUC score:', calib_auc_score)
        elif params.predict is True:
            print('Test data (Temp) -- AUC score:', calib_auc_score)

        # Perform Platt Scaling
        platt_model = p_model
        np_predict = np.array(all_predict)
        all_logit = np.log(np_predict / (1 - np_predict))
        all_pprob = platt_model.predict_proba(all_logit.reshape(-1, 1))[:, 1]
        calibration_score(all_label, all_pprob)
        calib_auc_score = roc_auc_score(y_true=all_label, y_score=all_pprob)
        if params.train is True:
            print('Validation data (Platt) -- AUC score:', calib_auc_score)
        elif params.predict is True:
            print('Test data (Platt) -- AUC score:', calib_auc_score)

    # Reliability plot for 15 bins, 50 bins and interactive binning schema
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    disp = CalibrationDisplay.from_predictions(all_label, all_predict, n_bins=15, strategy='uniform',
                                               name='Uncalibrated', ax=ax1)
    disp = CalibrationDisplay.from_predictions(all_label, all_predict, n_bins=50, strategy='uniform',
                                               name='Uncalibrated', ax=ax2)
    disp = CalibrationDisplay.from_predictions(all_label, all_predict, n_bins=15, strategy='quantile',
                                               name='Uncalibrated', ax=ax3)
    disp = CalibrationDisplay.from_predictions(all_label, all_predict, n_bins=50, strategy='quantile',
                                               name='Uncalibrated', ax=ax4)

    if params.calibration is True:

        # Plot graph for Temperature Scaling
        disp = CalibrationDisplay.from_predictions(all_label, all_tprob, n_bins=15, strategy='uniform',
                                                   name='Temperature', ax=ax1)
        disp = CalibrationDisplay.from_predictions(all_label, all_tprob, n_bins=50, strategy='uniform',
                                                   name='Temperature', ax=ax2)
        disp = CalibrationDisplay.from_predictions(all_label, all_tprob, n_bins=15, strategy='quantile',
                                                   name='Temperature', ax=ax3)
        disp = CalibrationDisplay.from_predictions(all_label, all_tprob, n_bins=50, strategy='quantile',
                                                   name='Temperature', ax=ax4)

        # Plot graph for Platt Scaling
        disp = CalibrationDisplay.from_predictions(all_label, all_pprob, n_bins=15, strategy='uniform',
                                                   name='Platt', ax=ax1)
        disp = CalibrationDisplay.from_predictions(all_label, all_pprob, n_bins=50, strategy='uniform',
                                                   name='Platt', ax=ax2)
        disp = CalibrationDisplay.from_predictions(all_label, all_pprob, n_bins=15, strategy='quantile',
                                                   name='Platt', ax=ax3)
        disp = CalibrationDisplay.from_predictions(all_label, all_pprob, n_bins=50, strategy='quantile',
                                                   name='Platt', ax=ax4)

    plt.show()

    if params.calibration is True:
        raw_predictions = (np.array(all_predict) > 0.5).astype(int)
        print('Raw confusion matrix: ', confusion_matrix(raw_predictions, all_label))
        all_clean_predict = 1 - np.array(all_predict)
        all_class_predict = np.column_stack((all_clean_predict, all_predict))
        class_predict = all_class_predict[range(len(raw_predictions)), raw_predictions]

        temp_predictions = (np.array(all_tprob) > 0.5).astype(int)
        print('Temperature confusion matrix: ', confusion_matrix(temp_predictions, all_label))
        all_clean_tprob = 1 - all_tprob
        all_class_tprob = np.column_stack((all_clean_tprob, all_tprob))
        class_tprob = all_class_tprob[range(len(temp_predictions)), temp_predictions]

        platt_predictions = (np.array(all_pprob) > 0.5).astype(int)
        print('Platt confusion matrix: ', confusion_matrix(platt_predictions, all_label))
        all_clean_pprob = 1 - all_pprob
        all_class_pprob = np.column_stack((all_clean_pprob, all_pprob))
        class_pprob = all_class_pprob[range(len(platt_predictions)), platt_predictions]

        # Defining parameters for precision percentile curve
        signal_names = ["Adjusted Trust Score"]
        signals = signals
        percentile_list = [15, 50, 100]
        for n in percentile_list:

            percentile_levels = [0 + float(100/n) * i for i in range(n)]

            # Illustrate if high scores correspond to correct predictions
            extra_plot_title = "DeepJIT Identify Correct"
            all_auc, _, _, _, _ = trustscore_evaluation.run_precision_recall_experiment(
                X_test,
                all_label,
                raw_predictions,
                class_predict,
                temp_predictions,
                class_tprob,
                platt_predictions,
                class_pprob,
                percentile_levels=percentile_levels,
                signal_names=signal_names,
                signals=signals,
                extra_plot_title=extra_plot_title,
                skip_print=False,
                predict_when_correct=True,
                fig_path='dump/plots/RQ3/Correct/' + str(n) + '_bins/'
            )
            print("p-p curve auc (Correct): ", all_auc)

            # Illustrate if low scores correspond to incorrect predictions
            extra_plot_title = "DeepJIT Identify Incorrect"
            all_auc, _, _, _, _ = trustscore_evaluation.run_precision_recall_experiment(
                X_test,
                all_label,
                raw_predictions,
                class_predict,
                temp_predictions,
                class_tprob,
                platt_predictions,
                class_pprob,
                percentile_levels=percentile_levels,
                signal_names=signal_names,
                signals=signals,
                extra_plot_title=extra_plot_title,
                skip_print=False,
                predict_when_correct=False,
                fig_path='dump/plots/RQ3/Incorrect/' + str(n) + '_bins/'
            )
            print("p-p curve auc (Incorrect): ", all_auc)
