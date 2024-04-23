# Time Series and Vitality testing scripts
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white.svg "pre-commit")](https://github.com/pre-commit/pre-commit) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg "black")](https://github.com/python/black)


<details>
<summary><b> Table of Contents </b></summary>

[TOC]

</details>

## Overview

This repository contains the main finalized scripts used for vitality research in Pablo de la Mora's Thesis, for the Kalman Filter, NBEATS, Break Detection based on BFAST and a script to create synthetic time series datasets. Each script contains most of the necessary functions to run independently.
<details>
<summary>

## Running Kalman Filter

In order to run the Kalman filter, one must have the script in the appropriate repository, and use it as such

```
from kalman_vitality_simple import kalman_executor
kalman_results, linear_predictions = kalman_executor(observations,history_period_length,dates,measurement_error,h_term)

```

The output are two arrays, one with the predictions of the Kalman filter and one with those from the Linear Harmonic Regression. The input is meant to be the full AOI's worth of observations without NaN values, in the shape (pixels,time steps), the length of the history period, the list of time series' dates, the measurement error and the number of harmonic herms for the filter. It is possible to set the history_period_length as the length of the whole time series to have as many predictions as there are observations, like in BFAST-Monitor. The measurement error is a crucial element, and can be set based on experimentation.

If the measurement error is set to = -1, the error will be the residual between the linear harmonic models and the observations, and usually very low.

## Running NBEATS

For this, it is best to use the notebook and follow the advice written there on the markdown cells

## Running Break Detection and Metric
This script has multiple useful functions that can be imported and used independently. Of those, the ones meant for import onto other work are as follow.
```
import break_detection as bd

updated_disturbances = bd.update_disturbances(ground_truth_disturbances,dates,start_monitor=dates[30])

detected_breaks = bd.break_detection(observations,kalman_predictions,dates[30],dates,hfrac=0.25,level=0.02,k=1)

break_metric_score = bd.break_metric(updated_disturbances,detected_breaks,30)

updated_detected_disturbances = bd.update_detected_disturbances(breaks_detected_nbeats,dates,dates[30])

```
To start, one can used the update_disturbances function to change the time step date of the disturbances based on the start of the monitoring period, to better compare the ground truth disturbances with those detected. The inputs are the ground truth disturbances, a list of time series dates, and the starting date of the monitoring period.

The break detection is based on BFAST, and needs the observations, predictions, dates, start of the monitoring period, and some parameters from BFAST such as hfrac, level and k. These last three can have the same value range as in BFAST. The output is an array of the time steps of the detected breaks within the monitoring period. indices with -1 have no detected break.

The break detection metric requires the ground truth disturbances, the detected disturbances and the size of the monitoring period. The output is the accuracy score of the detected disturbances throughout the entire set, based on the distance of the detections from the ground truths.

Finally, to set the detected disturbances in the time range of the entire time series, for graph visualization or any other purpose, use updated_detected_disturbances with the breaks array, the full time series dates list and the start of the monitoring period.

## Creating Synthetic Time Series

```
params_testing = {'a':[0.1,0.3,0.5], 'noise': [0.03,0.05,0.08,0.10,0.2],'m':[-0.2,-0.4,-0.6,-0.8],'fixed_disturbance':[False]}
series, p_sets_test,disturbs= timeseries_parameter_runner(params_testing,t_disturbance=50,set_size=500)
test_disturbances =  test_disturbances.reshape(test_disturbances.shape[0]*test_disturbances.shape[1])

bfast_testseries = np.moveaxis(testseries,-1,0)
testseries = testseries.reshape(testseries.shape[0]*testseries.shape[1],testseries.shape[2])

dates = np.arange(datetime(2017,4,4), datetime(2022,3,4), timedelta(days=30)).astype(datetime)
```
To create synthetic datasets, it is necessary to first create a dictionary as shown in the example, where `a` is the amplitude, `noise` is the random noise, `m` is the magnitude of the time series disturbances and `fixed_disturbance` is in case the user wants to have disturbances in the same time step in every pixel. When this is run, the dictionary, the time step of the fixed disturbances and the size of each set of time series (per parameter combination) are the inputs. It is important to add the disturbance time step even if you choose `False` for the fixed_disturbance, and if so, the disturbances will be in random places of the time series, and some wont have one.
The second sets of likes are for reshaping the time series for use in classic BFAST-Monitor(time steps,height,width), as well as on the Kalman filter (pixels,time steps)

## Dependency management

Dependencies are very simple and should be covered in most repositories, as one can see from the modules used in each script.
