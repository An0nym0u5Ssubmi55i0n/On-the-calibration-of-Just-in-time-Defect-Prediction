import numpy as np
from scipy import stats
from scipy.stats import monte_carlo_test
import ast
import csv
import pandas as pd
import matplotlib.pyplot as plt

def write_statistics(all_statistics_t_test, all_statistics_wilcoxon, header, normal_p_values, dataset, model, is_brier_only=False):
    """ Method used to write the calculated statistics to CSV file.
    """
    filename=f"aggregated_statistics_{dataset}_{model}_brier_appended.csv"
    normal_p_values_uncal = []
    normal_p_values_cal = []

    for p_value_cal, p_value_uncal in normal_p_values:
        normal_p_values_cal.append(p_value_cal)
        normal_p_values_uncal.append(p_value_uncal)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test"] + header)
        writer.writerow(["paired_t_test"] + all_statistics_t_test)
        writer.writerow(["wilcox_test"] + all_statistics_wilcoxon)
        writer.writerow(["p_values for normality_cal"] + normal_p_values_cal)
        writer.writerow(["p_values for normality_uncal"] + normal_p_values_uncal)


def write_to_csv(all_statistics_t_test_val, all_statistics_t_test_test, all_statistics_wilcoxon_val, all_statistics_wilcoxon_test, all_p_values_normality_val, all_p_values_normality_test, headers, dataset, model="LApredict"):
    """ Method used to write the calculated statistics to CSV file.
    """
    # Validation FOlds
    filename=f"aggregated_statistics_validation_{dataset}_{model}.csv"
    new_headers = []
    p_values_cal_val = []
    p_values_uncal_val = []
    p_values_cal_test = []
    p_values_uncal_test = []
    for index, header in enumerate(headers):
        suffix_platt = "_platt"
        suffix_temp = "_temp"
        new_headers.append(header + suffix_platt)
        new_headers.append(header + suffix_temp)

    for p_value_cal, p_value_uncal in all_p_values_normality_val:
        p_values_cal_val.append(p_value_cal)
        p_values_uncal_val.append(p_value_uncal)

    for p_value_cal, p_value_uncal in all_p_values_normality_test:
        p_values_cal_test.append(p_value_cal)
        p_values_uncal_test.append(p_value_uncal)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test"] + new_headers)
        writer.writerow(["paired_t_test"] + all_statistics_t_test_val)
        writer.writerow(["wilcox_test"] + all_statistics_wilcoxon_val)
        writer.writerow(["p_values for normality_cal"] + p_values_cal_val)
        writer.writerow(["p_values for normality_uncal"] + p_values_uncal_val)

    # Test data
    filename=f"aggregated_statistics_test_{dataset}_{model}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test"] + new_headers)
        writer.writerow(["paired_t_test"] + all_statistics_t_test_test)
        writer.writerow(["wilcox_test"] + all_statistics_wilcoxon_test)
        writer.writerow(["p_values for normality_cal"] + p_values_cal_test)
        writer.writerow(["p_values for normality_uncal"] + p_values_uncal_test)



def calculate_significance(uncalibrated_samples, calibrated_samples, all_statistics_t_test, all_statistics_wilcoxon, all_p_values):
    """
    Calculate significance for calibrated samples compared to uncalibrated sample either via t_test_paired or wilcoxon, depending on normality.

    Params:
        uncalibrated_samples (numpy.arr)
        calibrated_samples (numpy.arr)
        all_statistics_t_test (arr): array holding all repetitions, calculated values should be added here
        all_statistics_wilcoxon (arr): array holding all repetitions, calculated values should be added here
        all_p_values (arr): array holding tuples of p_values for normality
    """
    p_value_t_test = paired_t_test(uncalibrated_samples, calibrated_samples)
    p_value_wilcoxon = -1
    calibrated_samples_normal = is_normal_distributed(calibrated_samples, is_uncal=False)
    uncalibrated_samples_normal = is_normal_distributed(uncalibrated_samples, is_uncal=True)
    if not (calibrated_samples_normal[1] and uncalibrated_samples_normal[1]):
        print(f"[ERROR] not normally distributed")
        p_value_wilcoxon = wilcox_test(uncalibrated_samples, calibrated_samples)
        p_value_t_test = -1
    all_statistics_t_test.append(p_value_t_test)
    all_statistics_wilcoxon.append(p_value_wilcoxon)
    all_p_values.append((calibrated_samples_normal[0], uncalibrated_samples_normal[0]))


def run_calculations(all_val_data, all_test_data, headers):
    """ Method used to orchestrate the calculations for all configurations of the measured metrics.
    """
    all_statistics_t_test_val = []
    all_statistics_t_test_test = []
    all_statistics_wilcoxon_val = []
    all_statistics_wilcoxon_test = []
    all_p_values_normality_val = []
    all_p_values_normality_test = []

    for index, header in enumerate(headers):
        uncal_samples_val = all_val_data[0][index]
        cal_platt_samples_val = all_val_data[1][index]
        cal_temp_samples_val = all_val_data[2][index]

        uncal_samples_test = all_test_data[0][index]
        cal_platt_samples_test = all_test_data[1][index]
        cal_temp_samples_test = all_test_data[2][index]

        print(f"[RUN Calc] header = {header} with index = {index} for VALIDATION FOLDS\nuncal = {uncal_samples_val}\ncal_platt = {cal_platt_samples_val}\ncal temp = {cal_temp_samples_val}\n\n FOR TEST: \nnucal = {uncal_samples_test}\ncal_platt = {cal_platt_samples_test}\ncal_temp = {cal_temp_samples_test}")

        print(f"[RUN CALC] uncal, cal_platt VALIDATION")
        calculate_significance(uncal_samples_val, cal_platt_samples_val, all_statistics_t_test_val, all_statistics_wilcoxon_val, all_p_values_normality_val)
        print(f"[RUN CALC] uncal, cal_temp VALIDATION")
        calculate_significance(uncal_samples_val, cal_temp_samples_val, all_statistics_t_test_val, all_statistics_wilcoxon_val, all_p_values_normality_val)

        print(f"[RUN CALC] uncal, cal_temp TEST")
        calculate_significance(uncal_samples_test, cal_platt_samples_test, all_statistics_t_test_test, all_statistics_wilcoxon_test, all_p_values_normality_test)
        print(f"[RUN CALC] uncal, cal_temp TEST")
        calculate_significance(uncal_samples_test, cal_temp_samples_test, all_statistics_t_test_test, all_statistics_wilcoxon_test, all_p_values_normality_test)

    return all_statistics_t_test_val, all_statistics_t_test_test, all_statistics_wilcoxon_val, all_statistics_wilcoxon_test, all_p_values_normality_val, all_p_values_normality_test



def statistic(x, axis):
    """
    Taken from https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_normaltest.html
    """
    # Get only the `normaltest` statistic; ignore approximate p-value
    # still requires n >= 20
    return stats.normaltest(x, axis=axis).statistic


def is_normal_distributed_mc(samples, is_uncal, required_p_value=0.05):
    """
    Caclualte normality based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.monte_carlo_test.html#scipy.stats.monte_carlo_test, example from https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_normaltest.html

    Functioning is similar to other func: H_0 = is normally distributed, p_value <= 0.05 significant rejection of H_0
    """
    calibration = "uncalibrated" if is_uncal else "calibrated"
    res = stats.monte_carlo_test(samples, stats.norm.rvs, statistic)
    print(f"REsult = {res}")
    if res.pvalue < required_p_value:
        print(f"[NORMAL DISTRIBUTION MC] Likely that {calibration} distribution is not normal with p_value of {res.pvalue}")
        return (res.pvalue, False)
    print(f"[NORMAL DISTRIBUTION MC] for {calibration} p-value = {res.pvalue}")
    return (res.pvalue, True)


def is_normal_distributed_dagostino(samples, is_uncal, required_p_value=0.05):
    """
    Method used to determine if samples are drawn from a normal distribution.
    H_0_ = it is normally distributed
    Hence, a low p_value can be seen as evidence, that this sample distribution is not drawn from a normal distribution.
    Sufficient "high" p_value currently (randomly) set as 0.2.
    Noteworthy: only works properly for at least 50 samples.

    More infos https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_normaltest.html#hypothesis-normaltest
    """
    res = stats.normaltest(samples)
    calibration = "uncalibrated" if is_uncal else "calibrated"
    if res.pvalue < required_p_value:
        print(f"[NORMAL DISTRIBUTION] Likely that {calibration} distribution is not normal with p_value of {res.pvalue}")
        return (res.pvalue, False)
    print(f"[NORMAL DISTRIBUTION] statistic for {calibration} distribution: {res.statistic}\np-value = {res.pvalue}")
    return (res.pvalue, True)

def is_normal_distributed(samples, is_uncal, required_p_value=0.05):
    if len(samples) >= 50:
        print(f"[NORMAL DISTRIBUTION] Sample size >= 50 using default normality for is_uncal = {is_uncal}")
        return is_normal_distributed_dagostino(samples, is_uncal, required_p_value)
    print(f"[NORMAL DISTRIBUTION] Sample size < 50 using monte carlo normality for is uncal = {is_uncal}")
    return is_normal_distributed_mc(samples, is_uncal=False, required_p_value=0.05)

def paired_t_test(samples_uncal, samples_cal):
    """
    Compute the paired t test for the uncalibrated and calibrated ECE scores.
    This assumes H_0_ = identical average expected values for both samples

    More infos: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    """
    result = stats.ttest_rel(samples_uncal, samples_cal)
    print(f"[PAIRED T_TEST]] results = {result}\n with conf interval = {result.confidence_interval(0.95)}")
    return result.pvalue

def wilcox_test(uncalibrated_samples, calibrated_samples):
    """ TODO: Question of rounding for proper analysis -> else simply uncal - cal will be used. This might lead to some weird behaviour, as due to floating inacurracy the some numbers are treated (un)equally, that should not.
    H_0: samples are from same distribution 

    More infos: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    """
    res = stats.wilcoxon(uncalibrated_samples, calibrated_samples)
    print(f"[WILCOXON]] results = {res}\n")
    return res.pvalue


