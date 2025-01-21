import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from seaborn import axes_style

def eval_woe(df, feature_name, y_df, bins=2):
    """
    Evaluate weight of evidence (WoE) for a given feature in a DataFrame.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the feature to be evaluated.
    feature_name (str): The name of the feature/column in the DataFrame to evaluate.
    y_df (pd.Series or pd.DataFrame): The target variable (binary) corresponding to the feature.
    bins (int, optional): The number of bins to use for discretizing the feature if it is numeric. Default is 2. 
                          If the feature is categorical, this parameter is ignored.
    Returns:
    pd.DataFrame: A DataFrame containing the following columns:
        - feature_name or feature_name_bin: The binned feature values.
        - nbr_obs: The number of observations in each bin.
        - prob_event: The probability of the event in each bin.
        - prob_obs: The probability of observations in each bin.
        - nbr_event: The number of events in each bin.
        - nbr_nevent: The number of non-events in each bin.
        - prob_event_only: The probability of events only in each bin.
        - prob_nevent_only: The probability of non-events only in each bin.
        - WoE: The weight of evidence for each bin.
        - diff_prob_event: The absolute difference in probability of event between consecutive bins.
        - diff_WoE: The absolute difference in WoE between consecutive bins.
        - IV: The information value for each bin.
    Notes:
    - WoE is calculated as the natural logarithm of the ratio of the probability of non-events to the probability of events.
    - IV is calculated as the product of the difference in probabilities of non-events and events, and the WoE.
    - Infinite WoE values are replaced with 0.
    """
    if feature_name not in df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in DataFrame.")
    
    df[feature_name] = df[feature_name].fillna(-99999)  # Fill NaN values with a specific value

    if feature_name in df.select_dtypes(include='number').columns:
        bin_feature_name = f"{feature_name}_bin"
        df[bin_feature_name] = pd.cut(df[feature_name], bins)
        feature_name = bin_feature_name

        df[bin_feature_name] = pd.cut(df[feature_name], bins, labels=False)
    
    # Sort the DataFrame by WoE values if bins are numeric
    if pd.api.types.is_numeric_dtype(df[feature_name]):
        df = df.sort_values([feature_name])
        df = df.reset_index(drop=True)
    
    # Group by the feature and calculate the number of observations and mean of the target variable
    df = df.groupby(feature_name, as_index=False).agg(
        nbr_obs=(y_df.name, 'count'),
        prob_event=(y_df.name, 'mean')
    )
    df.columns = [df.columns.values[0], 'nbr_obs', 'prob_event']
    
    # Calculate the probability of observations in each bin
    total_nbr_obs = df['nbr_obs'].sum()
    df['prob_obs'] = df['nbr_obs'] / total_nbr_obs
    
    # Calculate the number of events and non-events in each bin
    df['nbr_event'] = df['nbr_event'].round().astype(int)
    df['nbr_event'] = df['nbr_event'].astype(int)
    df['nbr_nevent'] = (1 - df['prob_event']) * df['nbr_obs']
    df['nbr_nevent'] = df['nbr_nevent'].round().astype(int)
    
    # Calculate the number of events and non-events in each bin
    total_nbr_event = df['nbr_event'].sum()
    
    # Calculate the probability of events and non-events only in each bin
    if total_nbr_event == 0:
        df['prob_event_only'] = 0
    else:
        df['prob_event_only'] = df['nbr_event'] / total_nbr_event
    df['prob_nevent_only'] = df['nbr_nevent'] / df['nbr_nevent'].sum()
    
    # Calculate the weight of evidence (WoE) for each bin
    df['prob_event_only'] = df['prob_event_only'].replace(0, 1e-10)
    df['prob_nevent_only'] = df['prob_nevent_only'].replace(0, 1e-10)
    df['WoE'] = np.log(df['prob_nevent_only'] / df['prob_event_only'])
    df['WoE'] = df['WoE'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.reset_index(drop=True)
    
    # Calculate the absolute difference in probability of event and WoE between consecutive bins
    df['diff_prob_event'] = df['prob_event'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    
    # Calculate the information value (IV) for each bin
    df['IV'] = (df['prob_nevent_only'] - df['prob_event_only']) * df['WoE']
    
    return df

# ------------
# - PLOTTING -
# ------------


def plot_woe_bar(df_woe, x_rotation=0):
    """
    Plot weight of evidence bar graph.
    
    Parameters:
    df_woe (pd.DataFrame): DataFrame containing WoE values.
    x_rotation (int, optional): Rotation angle for x-axis labels. Default is 0.
    """
    style_dict = {'axes.facecolor': 'gainsboro', 'grid.color': 'white'}
    x = df_woe.iloc[:, 0].astype(str).values
    x = np.array(df_woe.iloc[:, 0].apply(str))
    y = df_woe['WoE']

    mask_y_below_zero = y < 0
    mask_y_above_zero = y >= 0

    plt.figure(figsize=figsize) # type: ignore
    plt.bar(x[mask_y_below_zero], y[mask_y_below_zero], color="lightcoral")
    plt.bar(x[mask_y_above_zero], y[mask_y_above_zero], color="steelblue")
    plt.xlabel(df_woe.columns[0].replace('_bin',''))
    plt.ylabel('Weight of Evidence')
    plt.xticks(rotation=x_rotation)


def plot_woe_line(df_woe, figsize=(18, 6), x_rotation=0):
    """
    Purpose:
    This function generates a line plot to visualize the WoE values for different bins of a feature.
    The x-axis represents the bins, and the y-axis represents the WoE values.
    
    Plot weight of evidence (WoE) line graph.
    
    Parameters:
    df_woe (pd.DataFrame): DataFrame containing WoE values.
    x_rotation (int, optional): Rotation angle for x-axis labels. Default is 0.
    figsize (tuple, optional): Size of the figure. Default is (18, 6).
    """
    style_dict = {'axes.facecolor': 'gainsboro', 'grid.color': 'white'}
    with axes_style(style_dict):
        x = np.array(df_woe.iloc[:, 0].apply(str))
        y = df_woe['WoE']

        plt.figure(figsize=(18, 6))
        plt.plot(x, y, marker='o', linestyle='--', color='steelblue')
        plt.xlabel(df_woe.columns[0].replace('_bin',''))
        plt.ylabel('Weight of Evidence')
        plt.xticks(rotation=x_rotation)
