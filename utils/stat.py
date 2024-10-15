from typing import Tuple
import random

import numpy as np
import pandas as pd


def calculate_proportion(df: pd.DataFrame, col_name: str) -> float:
    """Calculate the proportion of each dataframe column where the value is equal to 1.
    :param df: dataframe
    :param col_name: column name that contains a binary variable with 1 as a value
    :return: sample proportion
    """
    sample_counts = df[col_name].value_counts()
    sample_yes = sample_counts[sample_counts.index == 1].values[0]
    sample_total = len(df)
    return sample_yes / sample_total


def bootstrap_confidence_interval_two_proportions(
    data_1, data_2, alpha: float = 0.05, n_iterations: int = 1000
) -> Tuple[float, float]:
    """
    Calculate the confidence interval of proportion differences.
    :param data_1: group 1
    :dtype data_1: array-like
    :param data_2: group 2
    :dtype data_2: array-like
    :param alpha: significance level (default 0.05)
    :param n_iterations: number of bootstrap iterations (default 1000)
    :return: confidence interval of proportion differences (upper, lower)
    """
    proportion_means_diff = []
    alpha = alpha
    n_iterations = n_iterations

    travel_insurance_1 = data_1
    travel_insurance_2 = data_2

    for _ in range(n_iterations):
        bootstrap_sample1 = np.random.choice(
            travel_insurance_1, size=len(travel_insurance_1), replace=True
        )
        bootstrap_sample2 = np.random.choice(
            travel_insurance_2, size=len(travel_insurance_2), replace=True
        )

        proportion1 = np.mean(bootstrap_sample1)
        proportion2 = np.mean(bootstrap_sample2)

        proportion_diff = proportion1 - proportion2
        proportion_means_diff.append(proportion_diff)

    lower_bound = np.percentile(proportion_means_diff, 100 * alpha / 2)
    upper_bound = np.percentile(proportion_means_diff, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound


def proportion_diff_permutation(labels: pd.Series, n_obs_a: int, n_obs_b: int) -> float:
    """
    Calculate the proportion difference for a single permutation of data from two groups of observations.

    :param labels: a series of category labels for all observations from two independent groups.
    :type labels: pd.Series
    :param n_obs_a: The number of observations in group A.
    :type n_obs_a: int
    :param n_obs_b: The number of observations in group B.
    :type n_obs_b: int

    :return proportion_diff: The proportion difference for a single permutation of the data
                             from two groups of observations.
    :rtype: float
    """
    total_obs = n_obs_a + n_obs_b
    idx_a = set(random.sample(range(total_obs), n_obs_a))
    idx_b = set(range(total_obs)) - idx_a
    group_a = labels.iloc[list(idx_a)]
    group_b = labels.iloc[list(idx_b)]
    proportion_a = group_a.value_counts().iloc[0] / group_a.value_counts().sum()
    proportion_b = group_b.value_counts().iloc[1] / group_b.value_counts().sum()

    return proportion_a - proportion_b


def bootstrap_confidence_interval_two_means(
    obs1: pd.Series, obs2: pd.Series, alpha: float = 0.05, n_bootstrap: int = 1000
) -> tuple:
    """
    Calculate the bootstrap confidence interval for the difference between two means
    :param obs1: dataset number 1
    :type obs1: pd.Series
    :param obs2: dataset number 2
    :type obs2: pd.Series
    :param alpha: significance level (default 0.05)
    :type alpha: float
    :param n_bootstrap: number of bootstrap iterations (default 1000)
    :type n_bootstrap: int
    :return: confidence interval of proportion differences (upper, lower)
    """
    n_obs1, n_obs2 = len(obs1), len(obs2)

    bootstrap_means_diff = []
    for _ in range(n_bootstrap):
        bootstrap_sample1 = np.random.choice(obs1, size=n_obs1, replace=True)
        bootstrap_sample2 = np.random.choice(obs2, size=n_obs2, replace=True)

        mean1 = np.mean(bootstrap_sample1)
        mean2 = np.mean(bootstrap_sample2)

        means_diff = mean1 - mean2
        bootstrap_means_diff.append(means_diff)

    lower_bound = np.percentile(bootstrap_means_diff, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_means_diff, 100 * (1 - alpha / 2))

    return lower_bound, upper_bound


def mean_diff_permutation(values: pd.Series, n_obs_a: int, n_obs_b: int) -> float:
    """
    Calculate the mean difference for a single permutation of data from two groups of observations.
    :param values: a Sequence such as a pandas series or list of values for all observations from
                   two independent groups.
    :type values: Sequence
    :param n_obs_a: The number of observations in group A.
    :type n_obs_a: int
    :param n_obs_b: The number of observations in group B.
    :type n_obs_b: int

    :return mean_diff: The mean difference for a single permutation of the data
    from two groups of observations.
    :rtype: float
    """
    total_obs = n_obs_a + n_obs_b
    idx_a = set(random.sample(range(total_obs), n_obs_a))
    idx_b = set(range(total_obs)) - idx_a
    return values.iloc[list(idx_a)].mean() - values.iloc[list(idx_b)].mean()
