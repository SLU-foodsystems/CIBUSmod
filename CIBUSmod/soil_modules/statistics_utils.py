#!/usr/bin/env python

"""
Statistical utility functions for the CIBUSmod soil_modules

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def detect_outliers(x: pd.Series,
                    z_threshold=3,
                    y: pd.Series = None,
                    plot_fig: bool = False,
                    test: str = 'box',
                    xlabel: str = 'Unset label',
                    ylabel: str = 'Unset label',
                    verbose: bool = False):
    """
    Choose between a number of plots and functions that can help detect the presence of outliers

    Parameters
    ----------
    x:      Main dataseries to be used for plotting. Mandatory
    y:      (optional) dataseries to be used for plotting in scatter plots
    test:   (optional) The name of the plot type to be plotted. Available options are 'box', 'scatter', 'z-plot'.
    xlabel: (optional) The label for the x-axis in scatterplots.
    ylabel: (optional) The label for the x-axis in scatterplots.
    z_threshold: (optional) The value used to detect outliers with the Z-method (default: 3)
    verbose: (optional) If true, prints information from the Z-test to screen

    Returns
    -------
    None: Generates a plot of the choosen type in the notebook
    """
    # Z-score outlier detection
    # Run a Z-test, do a line plot and return detected outliers
    threshold = z_threshold
    z = np.abs(stats.zscore(x))
    ar = np.where(z > threshold)[0]
    drop_map = z.index[ar]
    new_set = x.drop(drop_map)
    z_new = np.abs(stats.zscore(new_set))
    outlier_dict = {'num_outliers': len(ar), 'outlier_index': drop_map, 'outlier_positions': ar, 'dirty_series': x, 'cleaned_series': new_set}

    if verbose:
        print(f'Potential outliers: {ar}')
        for i in ar:
            print(f'sko {z.index[i]} has a Z-score of {z.iloc[i]}')  # prints the sko no. and zscore of the identified outliers
            print(f"It's nominal value is {x.iloc[i]}")
        print(f"For comparison, the summary statistics of the remaining dataset is:\n{new_set.describe()}\nand it's Z-scores are:\n {z_new.describe()}")

    if plot_fig == True:
        if test == 'box':
            # Create a boxplot
            sns.boxplot(x=x)
        elif test == 'scatter':
            # Create a scatterplot
            if y is not None:
                fig, ax = plt.subplots(figsize=(16, 8))
                ax.scatter(x, y, color='green')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                plt.show()
            else:
                print("'A series is require for the 'y' coordinate when chosing type='scatter'")
        elif test == 'z-plot':
            z.plot()
        else:
            print("The available options for type are 'box', 'scatter' and 'z-plot'")
    return outlier_dict