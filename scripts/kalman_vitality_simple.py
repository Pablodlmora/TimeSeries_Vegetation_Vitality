"""Kalman Filter for an entire AOI."""
import concurrent.futures
from datetime import datetime
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
from tqdm import tqdm


def map_indices(dates: List[datetime]) -> np.ndarray:
    """Create index array for time series dates."""
    start = dates[0]
    end = dates[-1]
    start = datetime(start.year, 1, 1)
    end = datetime(end.year, 12, 31)

    drange = pd.date_range(start, end, freq="d")
    ts = pd.Series(np.ones(len(dates)), dates)
    ts = ts.reindex(drange)
    indices = np.argwhere(~np.isnan(ts).to_numpy()).T[0]

    return indices


def omega(i: int) -> Any:
    """Create frequency for time series."""
    return 2 * np.pi * i / 365.25


def interpolate_band(band: np.ndarray) -> np.ndarray:
    """Interpolate missing values in series."""
    band = band.copy()
    is_nan = np.isnan(band)
    nan_index = is_nan.nonzero()[0]
    value_index = (~is_nan).nonzero()[0]
    band[is_nan] = np.interp(nan_index, value_index, band[value_index])
    return band


def surprise_editing(surprise: float, Ck: float) -> Any:
    """Calculate new residual based on threshold crossing."""
    alpha = 0.05
    dof = 1
    thresh = stats.chi2.ppf(1 - alpha, dof)

    if surprise >= 0:
        surprise_edited = min((surprise / np.sqrt(Ck)), np.sqrt(thresh))
    else:
        surprise_edited = max((surprise / np.sqrt(Ck)), -np.sqrt(thresh))

    return surprise_edited


def kalman_simple(
    pixel_ts: np.ndarray,
    in_val: float,
    m_error: float,
    tsteps: np.ndarray,
    d: int,
    model: np.ndarray,
    h_terms: int,
    cu_type: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the Kalman filter for an individual pixel time series."""
    preds = np.zeros_like(pixel_ts)

    preds[0] = in_val

    R = np.array([m_error])

    if len(model) == 8:
        if h_terms == 1:
            x_0 = np.array([model[0], -model[1], model[5], model[2]])
            H = lambda dt: np.array([1, dt, np.sin(omega(1) * dt), np.cos(omega(1) * dt)])
            F = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # State transition
            P_0 = np.array(
                [[0.05, 0, 0, 0], [0, 0.05, 0, 0], [0, 0, 0.05, 0], [0, 0, 0, 0.12]]
            )  # state covariance            # Observation model
        elif h_terms == 2:
            x_0 = np.array([model[0], -model[1], model[5], model[2], model[6], model[3]])
            H = lambda dt: np.array(
                [1, dt, np.sin(omega(1) * dt), np.cos(omega(1) * dt), np.sin(omega(2) * dt), np.cos(omega(2) * dt)]
            )
            F = np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )  # State transition
            P_0 = np.array(
                [
                    [0.05, 0, 0, 0, 0, 0],
                    [0, 0.05, 0, 0, 0, 0],
                    [0, 0, 0.05, 0, 0, 0],
                    [0, 0, 0, 0.12, 0, 0],
                    [0, 0, 0, 0, 0.05, 0],
                    [0, 0, 0, 0, 0, 0.12],
                ]
            )  # state covariance            # Observation model
        elif h_terms == 3:
            x_0 = np.array([model[0], -model[1], model[5], model[2], model[6], model[3], model[7], model[4]])
            H = lambda dt: np.array(
                [
                    1,
                    dt,
                    np.sin(omega(1) * dt),
                    np.cos(omega(1) * dt),
                    np.sin(omega(2) * dt),
                    np.cos(omega(2) * dt),
                    np.sin(omega(3) * dt),
                    np.cos(omega(3) * dt),
                ]
            )
            F = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )  # State transition
            P_0 = np.array(
                [
                    [0.05, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0.05, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0.05, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0.12, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.05, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0.12, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0.05, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0.12],
                ]
            )  # state covariance            # Observation model

    elif len(model) != 8:
        if h_terms == 1:
            x_0 = np.array([model[0], -model[1], model[5]])
            H = lambda dt: np.array([1, np.sin(omega(1) * dt), np.cos(omega(1) * dt)])  # Observation model
            F = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # State transition
            P_0 = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.12]])  # state covariance
        elif h_terms == 2:
            x_0 = np.array([model[0], -model[1], model[5], model[2], model[6]])
            H = lambda dt: np.array(
                [1, np.sin(omega(1) * dt), np.cos(omega(1) * dt), np.sin(omega(2) * dt), np.cos(omega(2) * dt)]
            )
            F = np.array(
                [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
            )  # State transition
            P_0 = np.array(
                [[0.05, 0, 0, 0, 0], [0, 0.05, 0, 0, 0], [0, 0, 0.05, 0, 0], [0, 0, 0, 0.12, 0], [0, 0, 0, 0, 0.05]]
            )  # state covariance            # Observation model
        elif h_terms == 3:
            x_0 = np.array([model[0], -model[1], model[5], model[2], model[6], model[3]])
            H = lambda dt: np.array(
                [
                    1,
                    np.sin(omega(1) * dt),
                    np.cos(omega(1) * dt),
                    np.sin(omega(2) * dt),
                    np.cos(omega(2) * dt),
                    np.sin(omega(3) * dt),
                ]
            )
            F = np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )  # State transition
            P_0 = np.array(
                [
                    [0.05, 0, 0, 0, 0, 0],
                    [0, 0.05, 0, 0, 0, 0],
                    [0, 0, 0.05, 0, 0, 0],
                    [0, 0, 0, 0.12, 0, 0],
                    [0, 0, 0, 0, 0.05, 0],
                    [0, 0, 0, 0, 0, 0.12],
                ]
            )  # state covariance

    x_k = x_0
    P_k = P_0
    cusum = np.zeros((pixel_ts.shape[0],))
    surprise_list = []
    observation = []

    for k, dt in enumerate(tsteps[:]):
        z = pixel_ts[k]
        # dt = 31

        x_pred = np.matmul(F, x_k)
        P_pred = np.matmul(np.matmul(F, P_k), F.T)

        # Calculate the Kalman gain
        PH = np.dot(P_pred, H(dt).T)  # type: ignore
        Ck = np.dot(H(dt), PH) + R  # type: ignore
        K = PH / Ck

        z_pred = np.dot(H(dt), x_pred)  # type: ignore
        observation.append(z)
        # Surprise
        surprise = z - z_pred
        Ks = np.dot(K, surprise)

        surprise_edited = surprise_editing(surprise, Ck)

        surprise_list.append(surprise_edited)
        if cu_type:
            cusum[k] = max(0, cusum[k - 1] + surprise_edited - d)
        else:
            cusum[k] = np.cumsum(surprise)

        x_k = x_pred + np.where(np.isnan(Ks), 0, Ks)

        preds[k] = np.dot(H(dt), np.dot(F, x_k))  # type: ignore

        P_k = P_pred - np.where(np.isnan(Ks), 0, Ks) * np.dot(H(dt), P_pred)  # type: ignore # np.where(np.isnan(Ks), 0, Ks)

        # estimates = estimates.at[k].set(x_k)

    return preds, cusum


def linear_regression(
    pixel_ts: np.ndarray, dates: List[datetime], trend: bool, history_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create linear model timerseries for the initialization of the Kalman filter."""
    mapped_indices = map_indices(dates).astype(np.int32)
    y = pixel_ts
    N = mapped_indices.shape[0]

    # n = pixel_ts[history_length:].shape[0]
    h = int(float(N) * 0.25)
    temp = 2 * np.pi * mapped_indices / float(365)

    if trend == True:
        X = np.vstack((np.ones(N), mapped_indices))
    else:
        X = np.ones(N)

    for j in np.arange(1, 4 + 1):
        X = np.vstack((X, np.sin(j * temp)))
        X = np.vstack((X, np.cos(j * temp)))

    if trend == True:
        indxs = np.array([0, 1, 3, 5, 7, 2, 4, 6])
    else:
        indxs = np.array([0, 2, 4, 6, 1, 3, 5])

    X_h = X[:, :history_length]
    # X_m = X[:, history_length:]
    y_h = y[:history_length]
    # y_m = y[history_length:]

    model = linear_model.LinearRegression(fit_intercept=False)

    model.fit(X_h.T, y_h)
    h = model.coef_[indxs]

    y_pred = model.predict(X.T)

    y_error = y - y_pred

    return y_pred, y_error, h


def kalman_runner(
    index: np.ndarray,
    pixel_ts: np.ndarray,
    history_length: int,
    dates_historic: List[datetime],
    measurement_error: float,
    h_terms: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run all the pixels of the aoi on the kalman filter and output the predictions."""
    forecast_length = pixel_ts.shape[0] - history_length

    if forecast_length == 0:
        forecast_length = history_length
        tsteps_hist = np.linspace(1, 30.436875 * history_length, history_length)
        tsteps_monitoring = tsteps_hist
        monitoring_ts = pixel_ts
        historic_ts = pixel_ts

    else:
        historic_ts = pixel_ts[:history_length]
        monitoring_ts = pixel_ts[history_length:]
        tsteps_hist = np.linspace(1, 30.436875 * history_length, history_length)
        tsteps_monitoring = np.linspace(tsteps_hist[-1], 30.436875 * forecast_length, forecast_length)

    try:
        predictions, error, model = linear_regression(
            historic_ts, dates_historic[:history_length], trend=False, history_length=history_length
        )

        if measurement_error < 0:
            measurement_error = error.mean()

        preds, cusum = kalman_simple(
            monitoring_ts,
            monitoring_ts[0],
            measurement_error,
            tsteps_monitoring,
            0,
            model,
            h_terms=h_terms,
            cu_type=False,
        )
        linear_predictions = predictions
        kalman_predictions = preds
    except ValueError:
        kalman_predictions = np.zeros(monitoring_ts.shape)
        error = np.zeros(historic_ts.shape)
        # kalman_predictions.append(surprises)

    return kalman_predictions, linear_predictions


def kalman_executor(
    data: np.ndarray, history_length: int, dates_historic: List[datetime], measurement_error: float, h_term: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Executre the kalman runner function for the entire dataset."""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        index = np.arange(data.shape[0])
        history_lengths = [history_length for i in range(data.shape[0])]
        dates_historic_list = [dates_historic for i in range(data.shape[0])]
        measurment_errors = [measurement_error for i in range(data.shape[0])]
        h_terms = [h_term for i in range(data.shape[0])]
        kalman_output = []
        linear_output = []
        for index, output in tqdm(
            zip(
                index,
                executor.map(
                    kalman_runner, index, data, history_lengths, dates_historic_list, measurment_errors, h_terms
                ),
            )
        ):
            kalman_predictions, linear_preds = output
            kalman_output.append(kalman_predictions)
            linear_output.append(linear_preds)

        kalman_output = np.array(kalman_output)
        linear_output = np.array(linear_output)
        return kalman_output, linear_output
