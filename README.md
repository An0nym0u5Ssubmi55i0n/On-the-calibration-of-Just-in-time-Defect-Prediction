# On-the-calibration-of-Just-in-time-Defect-Prediction

## Getting started

This folder contains 4 files

The calibration_metrics files contains the implementation of the calibration metrics (ECE, MCE, Brier score) used to measure the (mis) calibration of each JIT DP model

The calculate_calibration_metrics file is a script for computing all the calibration metrics, given a set of predictions

The calibration_methods file contains the implementation of the calibration methods (Platt and Temperature scaling) used to adjust the calibration of the JIT DP models.

The calibration process is another script for running the whole training, calibration and evaluation process.

## JIT DP Models:
For these experiments we use three JIT DP techniques, whose implementation is publicly available

DeepJIT: https://github.com/soarsmu/DeepJIT/tree/master

LApredict: https://github.com/ZZR0/ISSTA21-JIT-DP

CodeBERT4JIT: https://github.com/Xin-Zhou-smu/Assessing-generalizability-of-CodeBERT

## DATASETS:

We conduct our experiments with the QT and OPENSTACK datasets, which can are publicly available under: https://zenodo.org/records/3965246#.XyEDVnUzY5k


## Models & Results:
You can find three sub-directories, one for each JIT DP model.
Inside each for these sub-directories, is the implementation as well as the results.
The results are split in three folders:
- the original model: folder contains the calibration measurements of the original (uncalibrated) JIT DP
- the temp-scaled: folder contains the calibration measurements of the JIT DP calibrated with Temperature scaling
- the Platt-scaled: folder contains the calibration measurements of the JIT DP calibrated with Platt scaling
