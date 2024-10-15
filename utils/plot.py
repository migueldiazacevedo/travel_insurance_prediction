from typing import Tuple, List

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def plot_numeric_feature_boxplot(
    df: pd.DataFrame, y_label: str = None, title: str = None
) -> plt.Axes:
    """
    Plot boxplot of columns in a dataframe containing numerical type data.
    :param df: features containing numerical data only.
    :type df: pd.DataFrame
    :param y_label: label for y-axis.
    :type y_label: str
    :param title: A title for the graph.
    :type title: str

    :return: A matplotlib axes object.
    :rtype: plt.Axes (matplotlib.axes.Axes)
    """
    ax = sns.boxplot(data=df)
    plt.title(title)
    plt.ylabel(y_label)
    plt.show()
    return ax


def plot_numerical_features_pairgrid(
    df: pd.DataFrame,
    hue: str = None,
    height: float = None,
    aspect: float = None,
    title: str = None,
) -> sns.PairGrid:
    """
    Plot pairgrid of numerical columns in a dataframe containing.
    :param df: A dataframe with numerical features to plot.
    :type df: pd.DataFrame
    :param hue: column name of hue to divide data by.
    :type hue: str
    :param height: height of grid.
    :type height: int
    :param aspect: aspect of grid.
    :type aspect: int
    :param title: A title for the graph.
    :type title: str

    :return: A matplotlib axes object.
    :rtype: seaborn.PairGrid
    """
    pg = sns.pairplot(df, hue=hue, height=height, aspect=aspect)
    pg.fig.suptitle(title, y=1.02)
    plt.show()
    return pg


def plot_feature_qq_plot(
    df: pd.DataFrame,
    title: str = "Note: Red diagonal lines represent a normal distribution",
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Plot Q-Q plots of features to assess normality
    :param df: dataframe containing features to plot.
    :type df: pd.DataFrame
    :param title: inforative title for the data
    :type title: str

    :return: A matplotlib features and axes object.
    :rtype: Tuple[matplotlib.figure.Figure, plt.Axes]
    """
    fig, axes = plt.subplots(
        nrows=1, ncols=int(len(df.columns.values)), figsize=(10, 5)
    )
    for i, feature in enumerate(df.columns.values):
        stats.probplot(df[feature], plot=axes.flatten()[i])
        axes.flatten()[i].set_title("Q-Q plots for " + feature)
    plt.suptitle(title)
    plt.show()
    return fig, axes


def plot_correlation_matrix(corr_mat: pd.DataFrame, title: str = None) -> plt.Axes:
    """
    Plot correlation matrix of features avoiding redundancy.
    :param corr_mat: Correlation matrix to plot
    :type corr_mat: pd.DataFrame
    :param title: title for your plot
    :type title: str

    :return: axes object of correlation matrix.
    :rtype: plt.Axes
    """
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    ax = sns.heatmap(corr_mat, cmap="crest", annot=True, mask=mask)
    plt.gca().grid(False)
    plt.title(title)
    plt.show()
    return ax


def plot_counts_of_categorical_features(
    df: pd.DataFrame, stat="probability"
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Plot counts of categorical features as a bunch of bar charts separated on a binary categorical feature.
    :param df: the dataframe containing features to plot.
    :type df: pd.DataFrame
    :param stat: {‘count’, ‘percent’, ‘proportion’, ‘probability’}
    :type stat: str

    :return: A matplotlib features and axes object.
    :rtype: Tuple[matplotlib.figure.Figure, plt.Axes]
    """
    fig, axs = plt.subplots(nrows=df.shape[1], ncols=1, figsize=(12, 4 * df.shape[1]))

    for ax, col in zip(axs, df.columns):
        sns.countplot(y=df[col], stat=stat, hue=df["TravelInsurance"], ax=ax)
        ax.set_title(f"Proportion of {col}", fontsize=14)
        ax.set_ylabel(col, fontsize=12)
        ax.set_xlabel("")

    plt.tight_layout()
    plt.show()
    return fig, axs


def plot_distribution_of_feature(
    feature: pd.Series, title: str = None
) -> Tuple[matplotlib.figure.Figure, plt.Axes, plt.Axes]:
    """
    Plot distribution of feature as side by side boxplot and violin plot.
    :param feature: a column of a feature to plot
    :type feature: pd.Series
    :param title: a title for the plot
    :type title: str

    :return: Figure object, and a boxplot and a violin plot
    :rtype: Tuple[matplotlib.figure.Figure, plt.Axes, plt.Axes]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    sns.boxplot(feature, ax=ax1)
    sns.violinplot(feature, ax=ax2)
    plt.suptitle(title)
    plt.show()
    return fig, ax1, ax2


def plot_proportion_diff_permutation(
    null_distribution: List[float],
    best_estimate: float,
    bins: int = 50,
    kde: bool = True,
    text_x: int = 11,
    text_y: int = 800,
) -> plt.Axes:
    """
    Plot permutation distribution proportion difference between two distributions.
    :param null_distribution: a list with the null distribution of proportion differences calculated
           from permutation test
    :type null_distribution: List[float]
    :param best_estimate: point estimate of permutation differences
    :type best_estimate: float
    :param bins: number of bins for null distribution histogram
    :type bins: int
    :param kde: Do you want a KDE plot?
    :type kde: bool
    :param text_x: x coordinate of text boxplot
    :type text_x: int
    :param text_y: y coordinate of text boxplot
    :type text_y: int

    :return: a matplotlib axes object to access plot
    """
    ax = sns.histplot(data=null_distribution, bins=bins, kde=kde)
    ax.axvline(x=best_estimate * 100, linestyle="--", color="black")

    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_tick_params(width=1, length=5)
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_tick_params(width=1, length=5)
    ax.text(
        text_x,
        text_y,
        f"Observed\ndifference: \n{best_estimate * 100:.1f}%",
        bbox={"facecolor": "white"},
    )

    plt.title("Permutation Test of Proportion Difference")
    plt.xlabel("Difference in Proportion")
    plt.ylabel("Count")
    plt.show()
    return ax


def plot_numerical_column_two_groups_histogram(
    df_group1: pd.DataFrame,
    df_group2: pd.DataFrame,
    column_name: str,
    title_group1: str,
    title_group2: str,
    x_label: str,
    y_label: str,
) -> Tuple[matplotlib.figure.Figure, plt.Axes, plt.Axes]:
    """
    Plot histograms two groups of numerical data with a common column .
    :param df_group1: data for group 1
    :type df_group1: pd.DataFrame
    :param df_group2: data for group 2
    :type df_group2: pd.DataFrame
    :param column_name: name of common column
    :type column_name: str
    :param title_group1: title for graph 1
    :type title_group1: str
    :param title_group2: title for graph 2
    :type title_group2: str
    :param x_label: label for x axes
    :type x_label: str
    :param y_label: label for y-axis
    :type y_label: str

    :return: figure and axes objects for plotting
    :rtype: Tuple[matplotlib.figure.Figure, plt.Axes, plt.Axes]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    df_group1[column_name].hist(ax=ax1)
    ax1.set_title(title_group1)
    df_group2[column_name].hist(ax=ax2)
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    ax2.set_title(title_group2)
    ax2.set_xlabel(x_label)
    plt.show()
    return fig, ax1, ax2
