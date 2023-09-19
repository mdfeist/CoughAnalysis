import matplotlib.pyplot as plt


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
    axs[3, 0].set_ylabel('Usage')
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
    axs[3, 1].set_ylabel('Usage')
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
    # ax1.tick_params(axis='x', labelrotation=45)
    ax1.set_title(
        f'{title} - Cough Count per Hour')
    ax1.yaxis.grid(True)

    ax2.boxplot(dates_info["cough_activity_per_hour"], labels=hours)
    ax2.set_ylabel('Cough Activity')
    # ax2.tick_params(axis='x', labelrotation=45)
    ax2.set_title(
        f'{title} - Cough Activity per Hour')
    ax2.yaxis.grid(True)

    ax3.boxplot(dates_info["activity_per_hour"], labels=hours)
    ax3.set_ylabel('Activity')
    # ax3.tick_params(axis='x', labelrotation=45)
    ax3.set_title(
        f'{title} - Activity per Hour')
    ax3.yaxis.grid(True)

    ax4.boxplot(dates_info["usage_per_hour"], labels=hours)
    ax4.set_ylabel('Usage')
    # ax4.tick_params(axis='x', labelrotation=45)
    ax4.set_title(f'{title} - Estimated Usage per Hour')
    ax4.yaxis.grid(True)

    plt.savefig(file_name)
    plt.close()
