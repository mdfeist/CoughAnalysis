import numpy as np
import matplotlib.pyplot as plt
import heatmap


def graph_day_info(dates_info, file_name, title='Subject'):
    FIG_SIZE = (20, 20)

    fig, axs = plt.subplots(
        4, 2, figsize=FIG_SIZE, layout='constrained')

    hours = [i for i in range(24)]

    axs[0, 0].bar(hours, dates_info["avg_cough_count_per_hour"], label=hours)
    axs[0, 0].set_ylabel('Cough Count')
    axs[0, 0].set_title(
        f'{title} - Cough Count per Hour Normalized')
    axs[0, 0].yaxis.grid(True)

    axs[1, 0].bar(
        hours, dates_info["avg_cough_activity_per_hour"], label=hours)
    axs[1, 0].set_ylabel('Cough Activity')
    axs[1, 0].set_title(
        f'{title} - Cough Activity per Hour Normalized')
    axs[1, 0].yaxis.grid(True)

    axs[2, 0].bar(hours, dates_info["avg_activity_per_hour"], label=hours)
    axs[2, 0].set_ylabel('Activity')
    axs[2, 0].set_title(
        f'{title} - Activity per Hour Normalized')
    axs[2, 0].yaxis.grid(True)

    axs[3, 0].bar(hours, dates_info["total_usage_per_hour"], label=hours)
    axs[3, 0].set_ylabel('Hours')
    axs[3, 0].set_title(f'{title} - Total Estimated Usage per Hour')
    axs[3, 0].yaxis.grid(True)

    axs[0, 1].bar(hours, dates_info["total_cough_count_per_hour"], label=hours)
    axs[0, 1].set_ylabel('Cough Count')
    axs[0, 1].set_title(
        f'{title} - Total Cough Count per Hour')
    axs[0, 1].yaxis.grid(True)

    axs[1, 1].bar(
        hours, dates_info["total_cough_activity_per_hour"], label=hours)
    axs[1, 1].set_ylabel('Cough Activity')
    axs[1, 1].set_title(
        f'{title} - Total Cough Activity per Hour')
    axs[1, 1].yaxis.grid(True)

    axs[2, 1].bar(hours, dates_info["total_activity_per_hour"], label=hours)
    axs[2, 1].set_ylabel('Activity')
    axs[2, 1].set_title(
        f'{title} - Total Activity per Hour')
    axs[2, 1].yaxis.grid(True)

    axs[3, 1].bar(hours, dates_info["total_usage_per_hour"], label=hours)
    axs[3, 1].set_ylabel('Hours')
    axs[3, 1].set_title(f'{title} - Total Estimated Usage per Hour')
    axs[3, 1].yaxis.grid(True)

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


def changes_over_time_day_info(dates_changes, file_name, title='Subject'):
    FIG_SIZE = (10, 20)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=FIG_SIZE, layout='constrained')

    ax1.plot(dates_changes["dates"],
             dates_changes["cough_count_avg_per_day"], linewidth=2, color='black', label='Average')
    ax1.plot(dates_changes["dates"],
             dates_changes["cough_count_max_per_day"], alpha=0.5, color='red', label='Maximum')
    ax1.plot(dates_changes["dates"],
             dates_changes["cough_count_min_per_day"], alpha=0.5, color='blue', label='Minimum')
    ax1.set_ylabel('Percentage')
    ax1.set_title(
        f'{title} - Cough Count')
    ax1.legend()
    ax1.yaxis.grid(True)

    ax2.plot(dates_changes["dates"],
             dates_changes["cough_activity_avg_per_day"], linewidth=2, color='black', label='Average')
    ax2.plot(dates_changes["dates"],
             dates_changes["cough_activity_max_per_day"], alpha=0.5, color='red', label='Maximum')
    ax2.plot(dates_changes["dates"],
             dates_changes["cough_activity_min_per_day"], alpha=0.5, color='blue', label='Minimum')
    ax2.set_ylabel('Percentage')
    ax2.set_title(
        f'{title} - Cough Activity')
    ax2.legend()
    ax2.yaxis.grid(True)

    ax3.plot(dates_changes["dates"],
             dates_changes["activity_avg_per_day"], linewidth=2, color='black', label='Average')
    ax3.plot(dates_changes["dates"],
             dates_changes["activity_max_per_day"], alpha=0.5, color='red', label='Maximum')
    ax3.plot(dates_changes["dates"],
             dates_changes["activity_min_per_day"], alpha=0.5, color='blue', label='Minimum')
    ax3.set_ylabel('Percentage')
    ax3.set_title(
        f'{title} - Activity')
    ax3.legend()
    ax3.yaxis.grid(True)

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

    row_labels = []
    col_labels = list(range(0, 24))

    for i, chunk in enumerate(changes_between_chunks):
        chunk_cough_count = chunk["avg_change_cough_count_per_hour"]
        chunk_activity_count = chunk["avg_change_cough_activity_per_hour"]
        chunk_activity = chunk["avg_change_activity_per_hour"]
        row_labels.append(chunk["start"])

        for j in range(24):
            if ~np.isnan(chunk_cough_count[j]):
                cough_count_data[i, j] = chunk_cough_count[j]

            if ~np.isnan(chunk_activity_count[j]):
                cough_activity_data[i, j] = chunk_activity_count[j]

            if ~np.isnan(chunk_activity[j]):
                activity_data[i, j] = chunk_activity[j]

    im, _ = heatmap.heatmap(cough_count_data, row_labels,
                            col_labels, ax1, vmin=0.0, vmax=2.0)
    heatmap.annotate_heatmap(im, cough_count_data,
                             textcolors=("white", "black"))
    ax1.autoscale(False)
    ax1.set_xlabel('Hours')
    # ax1.set_ylabel('Dates')
    ax1.set_title(
        f'{title} - Cough Count')

    im, _ = heatmap.heatmap(cough_activity_data, row_labels,
                            col_labels, ax2, vmin=0.0, vmax=2.0)
    heatmap.annotate_heatmap(im, cough_activity_data,
                             textcolors=("white", "black"))
    ax2.autoscale(False)
    ax2.set_xlabel('Hours')
    ax2.set_title(
        f'{title} - Cough Activity')

    im, _ = heatmap.heatmap(activity_data, row_labels,
                            col_labels, ax3, vmin=0.0, vmax=2.0)
    heatmap.annotate_heatmap(im, activity_data,
                             textcolors=("white", "black"))
    ax3.autoscale(False)
    ax3.set_xlabel('Hours')
    ax3.set_title(
        f'{title} - Activity')

    plt.savefig(file_name)
    plt.close()
