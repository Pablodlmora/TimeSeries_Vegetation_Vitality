"""Synthetic Time Series Data Generator."""
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tqdm import tqdm


def create_timeseries(
    params: Dict[str, float], years: int, points: int, frequency: int, t_disturbance: int, num_timeseries: int
) -> Tuple[List[float], List[int]]:
    """Dataset of time series of a chosen size with one set of parameters for all of them.

    params: a -> amplitude of the time series,
            noise -> external noise component
            m -> magnitude of the disturbance,


    """
    recovery_rate = 0.003
    noise, m, fixed_disturbance, a = params.values()  # whenever using the Sklearn Parametergrid

    # Time points
    time_points = np.arange(0, points)
    b, n = 0, -0.0006580936821823804
    synthetic_dataset = []
    disturbances = []

    for i in range(num_timeseries):
        if fixed_disturbance == False:
            t_disturbance = np.random.randint(points / 2, points + 10)

        # Create each component
        seasonal = seasonal_component(time_points, a, frequency, years)
        trend = trend_component(time_points, b, n, t_disturbance, m, recovery_rate)  # Define recovery_rate

        # Combine components to form the synthetic time series
        synthetic_time_series = seasonal + trend + np.random.normal(0, noise, points)
        synthetic_dataset.append(synthetic_time_series)

        if t_disturbance >= points:
            t_disturbance = -1

        disturbances.append(t_disturbance)
    disturbances = np.array(disturbances)

    return synthetic_dataset, disturbances


def timeseries_parameter_runner(
    param_dict: Dict[str, float],
    num_years: int = 5,
    num_points: int = 60,
    f: int = 12,
    t_disturbance: int = 30,
    ParallelUnits: int = 1,
    set_size: int = 100,
) -> np.array:
    """Run to create a Dictionary with parameters for amplitude, magnitude of disturbance and noise."""
    param_sorted_sets = ParameterGrid(param_dict)
    ParamSets = ParameterSampler(param_dict, len(param_sorted_sets))
    parameters = []
    synthetic_datasets = []
    disturbance_dates = []
    try:
        with ThreadPoolExecutor(max_workers=ParallelUnits) as executor:
            for params in tqdm(ParamSets):
                dataset = executor.submit(create_timeseries, params, num_years, num_points, f, t_disturbance, set_size)
                # results_keys = list(params.values())
                synthetic_datasets.append(dataset.result()[0])
                disturbance_dates.append(dataset.result()[1])
                parameters.append(params)
    # Handle keyboard interrupt to stop the execution
    except KeyboardInterrupt:
        print("CTRL+C .")
        executor.shutdown(wait=False)
    synthetic_datasets = np.array(synthetic_datasets)
    disturbance_dates = np.array(disturbance_dates)
    return synthetic_datasets, parameters, disturbance_dates


def omega(i: int) -> Any:
    """Periodic function of a yearly cycle."""
    return 2 * np.pi * i / 365.25


def seasonal_component(t: np.array, a: float, f: int, num_years: int) -> np.array:
    """Seasonal component for the time series.

    t: array of time points.
    a: amplitude
    f: sampling period
    num_years: number of years in the time series.

    """
    steps = np.linspace(0, 365.25 * num_years, num_years * f)
    return a * np.sin(omega(1) * steps) + 0.1 * np.cos(omega(1) * steps) + 0.5


def step_function(t: np.array, t_0: int, m: float) -> np.array:
    """Step function to add a disturbance to the time series."""
    # Returns m if t is greater than or equal to t_0, otherwise 0

    return np.where(t >= t_0, m, 0)


def linear_recovery(t: np.array, t_0: int, recovery_rate: float) -> np.array:
    """Constant recovery of a fixed magnitude after the disturbance."""
    # Only starts recovery after t_0, and the recovery is linear

    return np.where(t > t_0, recovery_rate * (t - t_0), 0)


def trend_component(t: np.array, b: float, n: float, t_disturbance: int, m: float, recovery_rate: float) -> np.array:
    """Trend component for the time series, usually very small when it comes to vegetation."""
    linear_trend = b + n * t
    disturbance = step_function(t, t_disturbance, m) - linear_recovery(t, t_disturbance, recovery_rate)
    return linear_trend + disturbance
