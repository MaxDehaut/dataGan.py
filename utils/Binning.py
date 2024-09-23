# -----------
# - CONTENT -
# -----------

# Imports
# Weight of evidence
# Plotting

# ------------------------------

# -----------
# - IMPORTS -
# -----------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from seaborn import axes_style


# ----------------------
# - WEIGHT OF EVIDENCE -
# ----------------------

def eval_woe(df, feature_name, y_df, bins=2):
    """Evaluate weight of evidence"""
    if (feature_name in df.select_dtypes(include='number').columns):
        bin_feature_name = feature_name + '_bin'
        df[bin_feature_name] = pd.cut(df[feature_name], bins)
        feature_name = bin_feature_name

    df = pd.concat([df[feature_name], y_df], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()],axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'nbr_obs', 'prob_event']
    df['prob_obs'] = df['nbr_obs'] / df['nbr_obs'].sum()
    df['nbr_event'] = df['prob_event'] * df['nbr_obs']
    df['nbr_event'] = df['nbr_event'].astype(int)

    df['nbr_nevent'] = (1 - df['prob_event']) * df['nbr_obs']
    df['nbr_nevent'] = df['nbr_nevent'].astype(int)

    df['prob_event_only'] = df['nbr_event'] / df['nbr_event'].sum()
    df['prob_nevent_only'] = df['nbr_nevent'] / df['nbr_nevent'].sum()
    df['WoE'] = np.log(df['prob_nevent_only'] / df['prob_event_only'])
    df['WoE'] = df['WoE'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop=True)

    # Calculates the difference of a Dataframe element compared with 
    # another element in the Dataframe (default is element in previous row)
    df['diff_prob_event'] = df['prob_event'].diff().abs()

    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prob_nevent_only'] - df['prob_event_only']) * df['WoE']

    return df

# ------------
# - PLOTTING -
# ------------


def plot_woe_bar(df_woe, x_rotation=0):
    """Plot weight of evidence bar graph"""
    with axes_style({'axes.facecolor': 'gainsboro', 'grid.color': 'white'}):
        x = np.array(df_woe.iloc[:, 0].apply(str))
        y = df_woe['WoE']

        mask_y_bz = y < 0
        mask_y_az = y >= 0

        plt.figure(figsize=(18, 6))
        plt.bar(x[mask_y_bz], y[mask_y_bz], color="lightcoral")
        plt.bar(x[mask_y_az], y[mask_y_az], color="steelblue")
        plt.xlabel(df_woe.columns[0].replace('_bin',''))
        plt.ylabel('Weight of Evidence')
        plt.xticks(rotation=x_rotation)


def plot_woe_line(df_woe, x_rotation=0):
    """Plot weight of evidence line graph"""
    with axes_style({'axes.facecolor': 'gainsboro', 'grid.color': 'white'}):
        x = np.array(df_woe.iloc[:, 0].apply(str))
        y = df_woe['WoE']

        plt.figure(figsize=(18, 6))
        plt.plot(x, y, marker='o', linestyle='--', color='steelblue')
        plt.xlabel(df_woe.columns[0].replace('_bin',''))
        plt.ylabel('Weight of Evidence')
        plt.xticks(rotation=x_rotation)
