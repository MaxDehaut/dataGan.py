# -----------
# - CONTENT -
# -----------

# Imports
# Configuration
# Categorical
# Continuous
# Features
# Donuts
# Planes

# ------------------------------

# -----------
# - IMPORTS -
# -----------

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

from seaborn import axes_style
from pandas.plotting import parallel_coordinates

# ----------
# - CONFIG -
# ----------

sns.set(font_scale=1.2)
sns.set_context({"figure.figsize": (10, 6)})
sns.set_style("whitegrid")


# ---------------
# - CATEGORICAL -
# ---------------


def plot_categorical(df, variable, ax=None):
    """Plot a categorical variable"""
    variable_df = df[variable].value_counts(normalize=True).reset_index()
    with axes_style({'axes.facecolor': 'lightsteelblue', 'grid.color': 'white'}):
        variable_df.set_index('index').T.plot(kind='barh', stacked=True, width=0.15)


def plot_categoricals(df, list_categorical):
    """Plot all categorical variables"""
    for i, variable in enumerate(list_categorical):
        plot_categorical(df, variable)
        plt.show()


# --------------
# - CONTINUOUS -
# --------------


def plot_continuous(data, list_continuous):
    """Plot a boxplot of the continuous variables"""
    palette = sns.color_palette()

    for i, variable in enumerate(list_continuous):
        with axes_style({'axes.facecolor': 'lightsteelblue', 'grid.color': 'white'}):
            a = sns.boxplot(x=variable, data=data, orient='h', palette=[palette[i]])
        plt.ylabel('        ', labelpad=45)
        plt.show()
        del a


# ------------
# - FEATURES -
# ------------


def plot_feature_comparison_box(data, feature1, feature2):
    """Plot boxed comparison of features/variables"""
    fig = px.box(data, y=feature2, x=feature1,
                 color='label',
                 hover_data=['label','inCust','ipHop','weekDay','appHour','inList'],
                 template="seaborn")
    fig.update_layout(plot_bgcolor='lightsteelblue')
    fig.show()


def plot_feature_comparison_scatter(data, feature1, feature2 , slider):
    """Plot scattered comparison of features/variables"""
    fig = px.scatter(data, x=feature1, y=feature2,
                     color='label', animation_frame=slider,
                     template="seaborn")
    fig.update_layout(plot_bgcolor='lightsteelblue')
    fig.show()


# ----------
# - DONUTS -
# ----------


def plot_summary_donuts(pivot_df):
    """Plot a summary in a donught shape"""
    d = pivot_df.T.to_dict()

    # Create colors
    cGr, cPr, cBu, cRd = [plt.cm.Greens, plt.cm.Purples, plt.cm.Blues, plt.cm.Reds]

    # nested donut, adapted from https://python-graph-gallery.com/163-donut-plot-with-subgroups/
    outer_counts = [d['Train']['size', 'wt', 1], d['Validation']['size', 'wt', 1]]
    inner_counts = [d['Train']['size', 'wt', 0], d['Validation']['size', 'wt', 0]]

    outer_names = ["{}:{}\n{:,.0f}".format(*x)
                   for x in [('Train', 1, outer_counts[0]), ('Validation', 1, outer_counts[1])]]
    inner_names = ["{}:{}\n{:,.0f}".format(*x)
                   for x in [('Train', 0, inner_counts[0]), ('Validation', 0, inner_counts[1])]]

    realized_test_size = d['Validation']['size', 'wt', 'All'] / d['All']['size', 'wt', 'All']

    # plot pair
    ax = plt.subplot(121)
    ax.axis('equal')

    # outer ring
    mypie1, _ = ax.pie(outer_counts, radius=1.35, labels=outer_names,
                       labeldistance=1.05, colors=[cBu(0.7), cBu(0.5)], startangle=90 )
    plt.setp(mypie1, width=0.3, edgecolor='white')

    # inner ring
    mypie2, _ = ax.pie(inner_counts, radius=1.33-0.3, labels=inner_names,
                       labeldistance=0.1, colors=[cPr(0.7), cPr(0.5)], startangle=90)
    plt.setp(mypie2, width=0.4, edgecolor='white')

    # show nested donut
    title = f"Train/Validation Record Counts\n({realized_test_size:.1%} Test)"
    plt.title(title, loc='left')
    plt.margins(0.1, 0.1)

    # create simpler pie chart from weighted counts
    total_model_cts = d['All']['sum', 'wt', 'All']
    realized_1_rate = d['All']['sum', 'wt', 1] / total_model_cts
    title = f"Target Rate (Weighted)\n({realized_1_rate:.2%} : 1)"

    outer_counts = [d['All']['sum', 'wt', 0], d['All']['sum', 'wt', 1]]
    outer_names = ["{}: {:,.1f}\n({:.1%})".format(i,
                   outer_counts[i], outer_counts[i]/total_model_cts ) for i in [0,1]]

    ax = plt.subplot(122)
    ax.axis('equal')

    # ld=0.5, 0.3, 0.15, 0.1, 0.08
    mypie3, _ = ax.pie(outer_counts, radius=1.35, labels=outer_names,
                       labeldistance=0.1, colors=[cRd(0.5), cGr(0.7)], startangle=65)
    plt.setp(mypie3, width=0.9, edgecolor='white')
    plt.setp(mypie3, width=0.5, edgecolor='white')
    plt.setp(mypie3, width=0.6, edgecolor='white')
    plt.title(title, loc='left')
    plt.margins(0.1, 0.1)

    plt.show()
    return plt


# ----------
# - PLANES -
# ----------


def plot_parallel_coordinates(df, num_clusters):
    '''Plot a parallel coordinates plot for the clusters in df'''
    palette = sns.color_palette("bright", 10)

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster==i])

    # Create the plot
    fig = plt.figure(figsize=(12, 15))
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster
    # appearing over the lines for the other clusters
    for i in range(num_clusters):
        plt.subplot(num_clusters, 1, i+1)
        for j, c in enumerate(cluster_points):
            if i != j:
                pc = parallel_coordinates(c, 'cluster', color=[add_alpha(palette[j],0.2)])
        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[add_alpha(palette[i],0.5)])
        del pc

        # Stagger the axes
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)


def add_alpha(colour, alpha):
    '''Add an alpha to the RGB colour'''
    return (colour[0], colour[1], colour[2], alpha)


def plot_factorial_planes(x_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    '''Plot a scatter plot on a factorial plane, one for each factorial plane'''
    # For each factorial plane
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # Initialise the matplotlib figure
            fig = plt.figure(figsize=(7, 6))

            # Display the points
            if illustrative_var is None:
                plt.scatter(x_projected[:, d1], x_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(x_projected[selected, d1],
                                x_projected[selected, d2],
                                alpha=alpha, label=value)
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i, (x, y) in enumerate(x_projected[:, [d1, d2]]):
                    plt.text(x, y, labels[i],
                             fontsize='14', ha='center', va='center')

            # Define the limits of the chart
            boundary = np.max(np.abs(x_projected[:, [d1, d2]])) * 1.1
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])

            # Display grid lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel(f"PC{d1+1} ({round(100*pca.explained_variance_ratio_[d1],1)}%)")
            plt.ylabel(f"PC{d2+1} ({round(100*pca.explained_variance_ratio_[d2],1)}%)")

            plt.title(f"Projection of points (on PC{d1+1} and PC{d2+1})")

            del fig
