import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import datetime
import pandas as pd
import seaborn as sns

import utils


def graph_day_info(dates_info, file_name, title='Subject'):
    FIG_SIZE = (20, 20)

    fig, axs = plt.subplots(
        4, 3, figsize=FIG_SIZE, layout='constrained')

    hours = [i for i in range(24)]

    # Avg Remove Outliers
    avg_cough_count_per_hour_no_outliers = np.zeros((24,))
    avg_cough_activity_per_hour_no_outliers = np.zeros((24,))
    avg_activity_per_hour_no_outliers = np.zeros((24,))

    for hour in range(24):
        if dates_info["total_usage_per_hour"][hour] > 3:
            avg_cough_count_per_hour_no_outliers[hour] = utils.mean_remove_outliers(
                dates_info["cough_count_per_hour"][hour])
            avg_cough_activity_per_hour_no_outliers[hour] = utils.mean_remove_outliers(
                dates_info["cough_activity_per_hour"][hour])
            avg_activity_per_hour_no_outliers[hour] = utils.mean_remove_outliers(
                dates_info["activity_per_hour"][hour])

    axs[0, 0].bar(hours, avg_cough_count_per_hour_no_outliers, label=hours)
    axs[0, 0].set_ylabel('Cough Count')
    axs[0, 0].set_title(
        f'{title} - Average Cough Count per Hour (No Outliers)')
    axs[0, 0].yaxis.grid(True)

    axs[1, 0].bar(
        hours, avg_cough_activity_per_hour_no_outliers, label=hours)
    axs[1, 0].set_ylabel('Cough Activity')
    axs[1, 0].set_title(
        f'{title} - Average Cough Activity per Hour (No Outliers)')
    axs[1, 0].yaxis.grid(True)

    axs[2, 0].bar(hours, avg_activity_per_hour_no_outliers, label=hours)
    axs[2, 0].set_ylabel('Activity')
    axs[2, 0].set_title(
        f'{title} - Average Activity per Hour (No Outliers)')
    axs[2, 0].yaxis.grid(True)

    axs[3, 0].bar(hours, dates_info["total_usage_per_hour"], label=hours)
    axs[3, 0].set_ylabel('Hours')
    axs[3, 0].set_title(f'{title} - Total Estimated Usage per Hour')
    axs[3, 0].yaxis.grid(True)

    # Avg
    axs[0, 1].bar(hours, dates_info["avg_cough_count_per_hour"], label=hours)
    axs[0, 1].set_ylabel('Cough Count')
    axs[0, 1].set_title(
        f'{title} - Average Cough Count per Hour')
    axs[0, 1].yaxis.grid(True)

    axs[1, 1].bar(
        hours, dates_info["avg_cough_activity_per_hour"], label=hours)
    axs[1, 1].set_ylabel('Cough Activity')
    axs[1, 1].set_title(
        f'{title} - Average Cough Activity per Hour')
    axs[1, 1].yaxis.grid(True)

    axs[2, 1].bar(hours, dates_info["avg_activity_per_hour"], label=hours)
    axs[2, 1].set_ylabel('Activity')
    axs[2, 1].set_title(
        f'{title} - Average Activity per Hour')
    axs[2, 1].yaxis.grid(True)

    axs[3, 1].bar(hours, dates_info["total_usage_per_hour"], label=hours)
    axs[3, 1].set_ylabel('Hours')
    axs[3, 1].set_title(f'{title} - Total Estimated Usage per Hour')
    axs[3, 1].yaxis.grid(True)

    # Total
    axs[0, 2].bar(hours, dates_info["total_cough_count_per_hour"], label=hours)
    axs[0, 2].set_ylabel('Cough Count')
    axs[0, 2].set_title(
        f'{title} - Total Cough Count per Hour')
    axs[0, 2].yaxis.grid(True)

    axs[1, 2].bar(
        hours, dates_info["total_cough_activity_per_hour"], label=hours)
    axs[1, 2].set_ylabel('Cough Activity')
    axs[1, 2].set_title(
        f'{title} - Total Cough Activity per Hour')
    axs[1, 2].yaxis.grid(True)

    axs[2, 2].bar(hours, dates_info["total_activity_per_hour"], label=hours)
    axs[2, 2].set_ylabel('Activity')
    axs[2, 2].set_title(
        f'{title} - Total Activity per Hour')
    axs[2, 2].yaxis.grid(True)

    axs[3, 2].bar(hours, dates_info["total_usage_per_hour"], label=hours)
    axs[3, 2].set_ylabel('Hours')
    axs[3, 2].set_title(f'{title} - Total Estimated Usage per Hour')
    axs[3, 2].yaxis.grid(True)

    plt.savefig(file_name)
    plt.close()


def estimate_box_plot_day_info(dates_info, file_name, title='Subject'):
    FIG_SIZE = (10, 20)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=FIG_SIZE, layout='constrained')

    hours = [i for i in range(24)]

    ax1.boxplot(dates_info["cough_count_per_hour"], labels=hours)
    ax1.set_ylabel('Cough Count')
    ax1.set_title(
        f'{title} - Cough Count per Hour')
    ax1.yaxis.grid(True)

    ax2.boxplot(dates_info["cough_activity_per_hour"], labels=hours)
    ax2.set_ylabel('Cough Activity')
    ax2.set_title(
        f'{title} - Cough Activity per Hour')
    ax2.yaxis.grid(True)

    ax3.boxplot(dates_info["activity_per_hour"], labels=hours)
    ax3.set_ylabel('Activity')
    ax3.set_title(
        f'{title} - Activity per Hour')
    ax3.yaxis.grid(True)

    ax4.bar(hours, dates_info["total_usage_per_hour"], label=hours)
    ax4.set_ylabel('Hours')
    ax4.set_title(f'{title} - Total Estimated Usage per Hour')
    ax4.yaxis.grid(True)

    plt.savefig(file_name)
    plt.close()


def graph_day_summary(dates_summary, file_name, title='Subject'):
    FIG_SIZE = (10, 20)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=FIG_SIZE, layout='constrained')

    sns.regplot(ax=ax1, data=dates_summary, x="days_from_start",
                y="cough_count_avg_per_day", robust=True)
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Cough Count')
    ax1.set_title(
        f'{title} - Cough Count')

    sns.regplot(ax=ax2, data=dates_summary, x="days_from_start",
                y="cough_activity_avg_per_day", robust=True)
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Cough Activity')
    ax2.set_title(
        f'{title} - Cough Activity')

    sns.regplot(ax=ax3, data=dates_summary, x="days_from_start",
                y="activity_avg_per_day", robust=True)
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Activity')
    ax3.set_title(
        f'{title} - Activity')

    plt.savefig(file_name)
    plt.close()


def changes_between_chunks(changes_between_chunks, file_name, title='Subject'):
    FIG_SIZE = (20, 20)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=FIG_SIZE, sharex=True, sharey=False)

    num_chunks = len(changes_between_chunks)

    cough_count_data = np.zeros((num_chunks, 24))
    cough_activity_data = np.zeros((num_chunks, 24))
    activity_data = np.zeros((num_chunks, 24))
    usage_mask = np.zeros((num_chunks, 24))

    row_labels = []
    col_labels = list(range(0, 24))

    for i, chunk in enumerate(changes_between_chunks):
        chunk_cough_count = chunk["avg_change_cough_count_per_hour"]
        chunk_activity_count = chunk["avg_change_cough_activity_per_hour"]
        chunk_activity = chunk["avg_change_activity_per_hour"]
        chunk_usage_mask = chunk["usage_mask"]
        row_labels.append(chunk["start"])

        for j in range(24):
            if ~np.isnan(chunk_cough_count[j]):
                cough_count_data[i, j] = chunk_cough_count[j]

            if ~np.isnan(chunk_activity_count[j]):
                cough_activity_data[i, j] = chunk_activity_count[j]

            if ~np.isnan(chunk_activity[j]):
                activity_data[i, j] = chunk_activity[j]

            if ~np.isnan(chunk_usage_mask[j]):
                usage_mask[i, j] = chunk_usage_mask[j]

    cough_count_df = pd.DataFrame(
        cough_count_data,
        columns=col_labels,
        index=row_labels)
    sns.heatmap(cough_count_df, annot=True,
                linewidth=.5, ax=ax1, vmin=0, vmax=2,
                cmap="viridis", mask=usage_mask)
    ax1.set_xlabel('Hours')
    ax1.set_title(
        f'{title} - Cough Count')

    cough_activity_df = pd.DataFrame(
        cough_activity_data,
        columns=col_labels,
        index=row_labels)
    sns.heatmap(cough_activity_df, annot=True,
                linewidth=.5, ax=ax2, vmin=0, vmax=2,
                cmap="viridis", mask=usage_mask)
    ax2.set_xlabel('Hours')
    ax2.set_title(
        f'{title} - Cough Activity')

    activity_df = pd.DataFrame(
        activity_data,
        columns=col_labels,
        index=row_labels)
    sns.heatmap(activity_df, annot=True,
                linewidth=.5, ax=ax3, vmin=0, vmax=2,
                cmap="viridis", mask=usage_mask)
    ax3.set_xlabel('Hours')
    ax3.set_title(
        f'{title} - Activity')

    for a in fig.axes:
        a.tick_params(
            axis='x',           # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom=True,
            top=False,
            labelbottom=True)    # labels along the bottom edge are on

    fig.tight_layout(pad=5.0)

    plt.savefig(file_name)
    plt.close()


def changes_between_chunks_avg(changes_between_chunks, file_name, title='Subject'):
    FIG_SIZE = (10, 20)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=FIG_SIZE, layout='constrained')

    dates = []
    cough_count_avg = []
    cough_activity_avg = []
    activity_avg = []

    for chunk in changes_between_chunks:
        dates.append(chunk["start"])
        cough_count_avg.append(chunk["avg"]["cough_count"])
        cough_activity_avg.append(chunk["avg"]["cough_activity"])
        activity_avg.append(chunk["avg"]["activity"])

    ax1.plot(dates, cough_count_avg, linewidth=2,
             color='black', label='Average')
    ax1.set_title(
        f'{title} - Cough Count')
    ax1.yaxis.grid(True)

    ax2.plot(dates, cough_activity_avg, linewidth=2,
             color='black', label='Average')
    ax2.set_ylabel('Percentage')
    ax2.set_title(
        f'{title} - Cough Activity')
    ax2.yaxis.grid(True)

    ax3.plot(dates, activity_avg, linewidth=2, color='black', label='Average')
    ax3.set_ylabel('Percentage')
    ax3.set_title(
        f'{title} - Activity')
    ax3.yaxis.grid(True)

    plt.savefig(file_name)
    plt.close()


def plot_usage_clusters(cluster_infos, subject, file_name):
    FIG_SIZE = (40, 20)
    n_clusters = len(cluster_infos)

    # Plot clusters
    fig = plt.figure(figsize=FIG_SIZE)
    height_ratios = (1.0 - 1/(n_clusters+1),)
    for i in range(n_clusters):
        height_ratios += (1,)
    gs = fig.add_gridspec(n_clusters+1, 2,  width_ratios=(30, 1), height_ratios=height_ratios,
                          left=0.15, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.15, hspace=n_clusters*0.15)

    ax1 = plt.subplot(gs[0, :1])

    totalNumDays = (subject.get_last_day() - subject.get_first_day()).days
    clusterLineData = np.zeros((1, totalNumDays))
    clusterLineLabels = []

    for days in range(totalNumDays):
        date = subject.get_first_day() + datetime.timedelta(days=days)
        clusterLineLabels.append(date)

    numOfDays = 0
    for cluster, cluster_info in enumerate(cluster_infos):
        numOfDays += cluster_info.size()

        for dayInfo in cluster_info.dates():
            dayIndex = (dayInfo.date() - subject.get_first_day()).days
            clusterLineData[0, dayIndex] = cluster + 1

    colors = [(1, 1, 1), (0.71, 0.84, 0.77), (0.69, 0.55, 0.73),
              (0.93, 0.66, 0.41), (0.40, 0.42, 0.78), (0.80, 0.42, 0.78)]
    colors = colors[:(n_clusters+1)]
    cmap = LinearSegmentedColormap.from_list(
        "CustomPastel", colors, N=n_clusters+1)

    values = np.unique(clusterLineData.ravel())
    im = ax1.imshow(
        clusterLineData, interpolation='none', cmap=cmap)
    ax1.set_xticks(np.arange(len(clusterLineLabels)),
                   labels=clusterLineLabels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=90,
             ha="right", rotation_mode="anchor")

    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = []  # mpatches.Patch(color=colors[0], label="No Cough Data")
    for i in range(n_clusters):
        patches.append(mpatches.Patch(
            color=colors[i+1], label=f"Cluster {i+1}"))
    # put those patched as legend-handles into the legend
    ax1.legend(handles=patches, facecolor='white',
               bbox_to_anchor=(1.05, 1), borderaxespad=0)

    hours = [i for i in range(24)]
    lastCoughCountAx = None
    lastCoughActivityAx = None
    for cluster in range(n_clusters):
        # Save time of day cough count box plot
        if (lastCoughCountAx == None):
            ax = plt.subplot(gs[cluster+1, 0])
            lastCoughCountAx = ax
        else:
            ax = plt.subplot(gs[cluster+1, 0],
                             sharey=lastCoughCountAx)

        usage = np.zeros((24,))

        for i, dayInfo in enumerate(cluster_infos[cluster].dates()):
            for hour in range(24):
                usage[hour] += dayInfo.estimated_usage()[hour]

        ax.bar(hours, usage, label=hours)
        ax.set_ylabel('Usage')
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_title(
            f'Cluster {cluster+1} - Usage Time of Day')
        ax.yaxis.grid(True)
        # Save cluster size
        ax = plt.subplot(gs[cluster+1, 1])

        ax.bar([f'Cluster {cluster+1}'],
               [cluster_infos[cluster].size()])
        ax.set_ylabel('Days')
        ax.set_ylim([0, numOfDays])
        ax.yaxis.grid(True)
    # plt.subplots_adjust(right=0.7)
    # plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def plot_cough_clusters(cluster_infos, subject, file_name):
    FIG_SIZE = (40, 20)
    n_clusters = len(cluster_infos)

    # Plot clusters
    fig = plt.figure(figsize=FIG_SIZE)
    height_ratios = (1.0 - 1/(n_clusters+1),)
    for i in range(n_clusters):
        height_ratios += (1,)
    gs = fig.add_gridspec(n_clusters+1, 2,  width_ratios=(30, 1), height_ratios=height_ratios,
                          left=0.15, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.15, hspace=n_clusters*0.15)

    ax1 = plt.subplot(gs[0, :1])

    totalNumDays = (subject.get_last_day() - subject.get_first_day()).days
    clusterLineData = np.zeros((1, totalNumDays))
    clusterLineLabels = []

    for days in range(totalNumDays):
        date = subject.get_first_day() + datetime.timedelta(days=days)
        clusterLineLabels.append(date)

    numOfDays = 0
    for cluster, cluster_info in enumerate(cluster_infos):
        numOfDays += cluster_info.size()

        for dayInfo in cluster_info.dates():
            dayIndex = (dayInfo.date() - subject.get_first_day()).days
            clusterLineData[0, dayIndex] = cluster + 1

    colors = [(1, 1, 1), (0.71, 0.84, 0.77), (0.69, 0.55, 0.73),
              (0.93, 0.66, 0.41), (0.40, 0.42, 0.78), (0.80, 0.42, 0.78)]
    colors = colors[:(n_clusters+1)]
    cmap = LinearSegmentedColormap.from_list(
        "CustomPastel", colors, N=n_clusters+1)

    values = np.unique(clusterLineData.ravel())
    im = ax1.imshow(
        clusterLineData, interpolation='none', cmap=cmap)
    ax1.set_xticks(np.arange(len(clusterLineLabels)),
                   labels=clusterLineLabels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=90,
             ha="right", rotation_mode="anchor")

    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = []  # mpatches.Patch(color=colors[0], label="No Cough Data")
    for i in range(n_clusters):
        patches.append(mpatches.Patch(
            color=colors[i+1], label=f"Cluster {i+1}"))
    # put those patched as legend-handles into the legend
    ax1.legend(handles=patches, facecolor='white',
               bbox_to_anchor=(1.05, 1), borderaxespad=0)

    hours = [i for i in range(24)]
    lastCoughCountAx = None
    lastCoughActivityAx = None
    for cluster in range(n_clusters):
        # Save time of day cough count box plot
        if (lastCoughCountAx == None):
            ax = plt.subplot(gs[cluster+1, 0])
            lastCoughCountAx = ax
        else:
            ax = plt.subplot(gs[cluster+1, 0],
                             sharey=lastCoughCountAx)

        cough_count = np.zeros((24,))

        for i, dayInfo in enumerate(cluster_infos[cluster].dates()):
            for hour in range(24):
                cough_count[hour] += dayInfo.coughCount()[hour]

        ax.bar(hours, cough_count, label=hours)
        ax.set_ylabel('Cough Count')
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_title(
            f'Cluster {cluster+1} - Cough Count Time of Day')
        ax.yaxis.grid(True)
        # Save cluster size
        ax = plt.subplot(gs[cluster+1, 1])

        ax.bar([f'Cluster {cluster+1}'],
               [cluster_infos[cluster].size()])
        ax.set_ylabel('Days')
        ax.set_ylim([0, numOfDays])
        ax.yaxis.grid(True)
    # plt.subplots_adjust(right=0.7)
    # plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
