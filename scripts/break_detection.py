"""Break Detection and Metric Based on BFAST."""
from datetime import datetime
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from constants import __critvals

__critval_h = np.array([0.25, 0.5, 1])
__critval_period = np.arange(2, 12, 2)
__critval_level = np.arange(0.95, 0.999, 0.001)
__critval_mr = np.array(["max", "range"])


def check(h: float, period: float, level: float, mr: str) -> None:
    """Check for validity of entered parameters within the approrpiate value ranges."""
    if not h in __critval_h:
        raise ValueError("h can only be one of", __critval_h)

    # if not period in __critval_period:
    #    raise ValueError("period can only be one of", __critval_period)

    if not level in __critval_level:
        raise ValueError("level can only be one of", __critval_level)

    if not mr in __critval_mr:
        raise ValueError("mr can only be one of", __critval_mr)


def get_critval(h: float, period: float, level: float, mr: str) -> Any:
    """Take  the critical value for break detection fromthe appririate critval list."""
    check(h, period, level, mr)
    index = np.zeros(4, dtype=int)
    # Get index into table from arguments
    index[0] = np.where(mr == __critval_mr)[0][0]
    index[1] = np.where(level == __critval_level)[0][0]
    # index[2] = np.where(period == __critval_period)[0][0]
    # print((np.abs(__critval_period - period)).argmin())
    index[2] = (np.abs(__critval_period - period)).argmin()
    index[3] = np.where(h == __critval_h)[0][0]
    # For historical reasons, the critvals are scaled by sqrt(2)

    return __critvals[tuple(index)] * np.sqrt(2)


def map_indices(dates: List[datetime]) -> Any:
    """Create a list of indices for the dates."""
    start = dates[0]
    end = dates[-1]
    start = datetime(start.year, 1, 1)
    end = datetime(end.year, 12, 31)

    drange = pd.date_range(start, end, freq="d")
    ts = pd.Series(np.ones(len(dates)), dates)
    ts = ts.reindex(drange)
    indices = np.argwhere(~np.isnan(ts).to_numpy()).T[0]

    return indices


def _log_plus(a: np.array) -> Any:
    """Return value for normalization on the break detection bounds."""
    retval = np.ones(a.shape, dtype=float)
    fl = a > np.e
    retval[fl] = np.log(a[fl])

    return retval


def compute_lam(N: int, hfrac: float, level: float, period: float) -> Any:
    """Return values for normalization of the break detection bounds."""
    check(hfrac, period, 1 - level, "max")
    return get_critval(hfrac, period, 1 - level, "max")


def compute_end_history(dates: List[datetime], start_monitor: datetime) -> int:
    """Return value of the first date of the monitoring period.

    Raises exception it the start of the monitoring period is not witin the history period.

    """
    for i in range(len(dates)):
        if start_monitor <= dates[i]:
            return i

    raise Exception("Date 'start' not within the range of dates!")


def break_detection(
    measurements: np.array,
    predictions: np.array,
    start_monitor: datetime,
    dates: List[datetime],
    hfrac: float,
    level: float,
    k: int,
    full_predictions: bool = True,
) -> Tuple[np.array, np.array]:
    """Detect breaks in time seroes, returns the breaks in the monitoring period and the mean values of the residuals."""
    n = compute_end_history(dates, start_monitor)
    # measurements = measurements[:,measurements.shape[1] - n:]
    # predictions = predictions[:,predictions.shape[1] - n:]
    pred_residuals = measurements - predictions

    if full_predictions:
        mapped_indices = map_indices(dates)
        h = int(float(n) * hfrac)
        N = mapped_indices.shape[0]
        err_cs = np.cumsum(pred_residuals[:, n - h : N + 1], axis=1)
        mosum = err_cs[:, h:] - err_cs[:, :-h]
        a = mapped_indices[n:] / mapped_indices[n - 1].astype(float)
        # compute magnitude
        # magnitudes = np.median(pred_residuals[:,n:],axis=1)
        # magnitude_tstep = np.where(pred_residuals[:,n:] ==magnitudes)[0]

    else:
        mapped_indices = map_indices(dates[n:])
        h = int(float(n) * hfrac)
        N = mapped_indices.shape[0]
        err_cs = np.cumsum(pred_residuals, axis=1)
        mosum = err_cs
        a = mapped_indices[0:] / mapped_indices[0].astype(float)
        # compute magnitude
        # magnitudes = np.median(pred_residuals,axis=1)
        # magnitude_tstep = np.where(pred_residuals == magnitudes)[0]

    sigma = np.sqrt(np.sum(pred_residuals[:, :n] ** 2, axis=1) / (n - (2 + 2 * k)))
    sigma = np.expand_dims(sigma, 1)
    mosum = 1.0 / (sigma * np.sqrt(n)) * mosum
    period = measurements.shape[0] / float(n)

    lam = compute_lam(measurements.shape[1], hfrac, level, period)
    bounds = lam * np.sqrt(_log_plus(a))

    # compute mean
    means = []

    all_breaks = []
    for i in range(mosum.shape[0]):
        means.append(np.mean(mosum[i]))

        breaks = np.abs(mosum[i]) > bounds

        first_break = np.where(breaks)[0]
        if first_break.shape[0] > 0:
            first_break = first_break[0]
        else:
            first_break = -1
        all_breaks.append(first_break)

    means = np.array(means)
    all_breaks = np.array(all_breaks)

    return all_breaks, means


def update_disturbances(disturbances: np.array, dates: List[datetime], start_monitor: datetime) -> np.array:
    """Update the ground truth disturbances time steps so the indexing works realtive to the monitoring period and the detected disturbances, for the sake of the break detection metric."""
    n = compute_end_history(dates, start_monitor)
    updated_disturbances = []
    for i in disturbances:
        if i != -1:
            new_disturbance = i - n
            if new_disturbance > 0:
                updated_disturbances.append(new_disturbance)
            else:
                updated_disturbances.append(-1)
        else:
            updated_disturbances.append(i)

    new_disturbances = np.array(updated_disturbances)

    return new_disturbances


def update_detected_disturbances(
    detected_disturbances: np.array, dates: List[datetime], start_monitor: datetime
) -> np.array:
    """Update the detected breaks to match the full time series indexing for visualization and proper date logging."""
    n = compute_end_history(dates, start_monitor)
    updated_disturbances = []
    for i in detected_disturbances:
        if i != -1:
            new_disturbance = i + n

            updated_disturbances.append(new_disturbance)
        else:
            updated_disturbances.append(-1)

    new_disturbances = np.array(updated_disturbances)

    return new_disturbances


def break_metric(y_true: np.ndarray, y_pred: np.ndarray, monitoring_period: int) -> Any:
    """
    Calculate the break metric, which checks for the break detection time step accuracy of the entire set.

    y_true: 2D array of integers.
    y_pred: 2D array of integers.

    """
    # Compute difference
    difference = np.abs(y_true - y_pred) / monitoring_period

    # Convert from break points to binary 0, 1
    y_true = (y_true != -1).astype(int)
    y_pred = (y_pred != -1).astype(int)

    # compute binary difference
    binary_difference = np.abs(y_true - y_pred)

    # Put these together
    total_diff = np.where(binary_difference == 1, 1, difference)
    # Return mean
    return total_diff.mean()
