from os import makedirs

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu, ttest_ind

from tqdm import tqdm

import calendar
import datetime

from subject import Subject, SUBJECT_ACTIVE_TIME

import utils
import time_utils


DATA_DIR = "data_oct_2022/"
CSV_PATH = DATA_DIR + "cleaned.csv"
RESULTS_DIR = "results/cluster/"

makedirs(RESULTS_DIR, exist_ok=True)

CHANGES_FIG_SIZE = (40, 20)
FIG_SIZE = (40, 20)
FEATURE_SIZE = 48


class DayInfo:
    def __init__(self, date, day_df, valid_events_df) -> None:
        self._date = date
        self._day_df = day_df
        self._valid_events = np.zeros(24)
        self._cough_count = np.zeros(24)
        self._cough_activity = np.zeros(24)
        self._activity = np.zeros(24)

        self._features = None

        self._valid_events_distributions = []
        self._cough_count_distributions = []
        self._cough_activity_distributions = []
        self._activity_distributions = []

        self._total_cough_count = 0
        self._cough_activity_events = []

        day_cough_count_df = day_df.loc[(day_df['event type'] == 'cough')]
        day_cough_activity_df = day_df.loc[(
            day_df['event type'] == 'cough activity')]
        day_activity_df = day_df.loc[(day_df['event type'] == 'activity')]

        start_time = time_utils.convert_to_unix(date)
        for _, row in valid_events_df.iterrows():
            cough_count = row['cough count']
            time = row['corrected_start_time']
            time_of_day = time - start_time

            self._valid_events_distributions.append(time_of_day)

        for _, row in day_cough_count_df.iterrows():
            cough_count = row['cough count']

            self._total_cough_count += cough_count

            time = row['corrected_start_time']
            time_of_day = time - start_time

            for _ in range(int(cough_count)):
                self._cough_count_distributions.append(time_of_day)

        for _, row in day_cough_activity_df.iterrows():
            cough_activity = row['cough activity']

            self._cough_activity_events.append(cough_activity)

            time = row['corrected_start_time']
            time_of_day = time - start_time

            for _ in range(int(cough_activity)):
                self._cough_activity_distributions.append(time_of_day)

        for _, row in day_activity_df.iterrows():
            activity = row['activity']
            time = row['corrected_start_time']
            time_of_day = time - start_time

            for _ in range(int(activity)):
                self._activity_distributions.append(time_of_day)

    def calculateFeatures(self, max_valid_events, max_cough_count, max_cough_activity, max_activity):
        ve = self._valid_events / max_valid_events
        ve = np.clip(ve, 0, 1)

        cc = self._cough_count / max_cough_count
        cc = np.clip(cc, 0, 1)

        ca = self._cough_activity / max_cough_activity
        ca = np.clip(ca, 0, 1)

        a = self._activity / max_activity
        a = np.clip(a, 0, 1)

        self._features = np.concatenate([cc, ca])

    def setHour(self, hour, valid_events, cough_count, cough_activity, activity):
        self._valid_events[hour] = valid_events
        self._cough_count[hour] = cough_count
        self._cough_activity[hour] = cough_activity
        self._activity[hour] = activity

    def features(self):
        return self._features

    def date(self):
        return self._date

    def dayDataFrame(self):
        return self._day_df

    def validEvents(self):
        return self._valid_events

    def coughCount(self):
        return self._cough_count

    def coughActivity(self):
        return self._cough_activity

    def activity(self):
        return self._activity


def graphDistributions(x, labels, dir):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    df = pd.DataFrame({
        'x': x,
        'labels': labels
    })

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="labels", hue="labels", aspect=5,
                      height=2, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "x", clip_on=False,
          color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.savefig(dir)
    plt.close()


def calculateMax(list):
    size = len(list)
    n = size // 2
    x = np.array(list)

    x = np.sort(x)

    Q1 = np.median(x[:n])
    Q3 = np.median(x[n:])

    IQR = Q3 - Q1

    upper = Q3 + 1.5*IQR

    # Remove outliers
    x = x[x <= upper]

    return np.max(x)


def calculateDistributionInfo(x):
    size = x.shape[0]
    n = size // 2

    x = np.sort(x)

    median = np.median(x)

    Q1 = np.median(x[:n])
    Q3 = np.median(x[n:])

    IQR = Q3 - Q1

    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    return median, lower, upper


def getCost(dates):
    n = len(dates)

    costs = 100.0*np.ones((n, n))

    for i in range(n):
        features_i = dates[i].features()
        for j in range(i+1, n):
            features_j = dates[j].features()
            costs[i, j] = -np.sum(features_i*np.log(features_j+1e-9))

    return costs


def getClusterError(dates, clusters, cluster_centers):
    n = len(dates)

    costs = np.zeros(n)

    for i in range(n):
        features = dates[i].features()
        j = clusters[i]
        center = cluster_centers[j]
        costs[i] = -np.sum(center*np.log(features+1e-9))

    return costs


def calculate(subject):
    # Get subject data
    subject_df = subject.get_data()

    min_time = SUBJECT_ACTIVE_TIME[subject.get_id()][0] * 3600
    max_time = SUBJECT_ACTIVE_TIME[subject.get_id()][1] * 3600

    dates = []
    valid_events_list = []
    cough_count_list = []
    cough_activity_list = []
    activity_list = []

    for day_start in time_utils.daterange(subject.get_first_day(), subject.get_last_day()):
        day_start_time = time_utils.convert_to_unix(day_start)
        day_end = day_start + datetime.timedelta(days=1)
        day_end_time = time_utils.convert_to_unix(day_end)

        day_df = subject_df.loc[(subject_df['corrected_start_time'] >= (day_start_time)) & (
            subject_df['corrected_start_time'] < (day_end_time))]
        day_cough_count_df = day_df.loc[(day_df['event type'] == 'cough')]

        day_cough_count_total = day_cough_count_df['cough count'].sum()

        if np.isnan(day_cough_count_total):
            day_cough_count_total = 0.0

        valid_events_df = utils.get_valid_events(day_df)
        valid_events_count = valid_events_df.shape[0]

        # Calculate Time of Day
        if day_cough_count_total > 2 and valid_events_count > 10:
            dayInfo = DayInfo(day_start, day_df, valid_events_df)
            for hour in range(24):
                start_time = datetime.datetime.combine(
                    day_start, datetime.time(hour=hour))
                end_time = start_time + datetime.timedelta(hours=1)

                start_time = time_utils.convert_to_unix(start_time)
                end_time = time_utils.convert_to_unix(end_time)

                hour_valid_events = valid_events_df.loc[(valid_events_df['corrected_start_time'] >= (start_time)) & (
                    valid_events_df['corrected_start_time'] < (end_time))]

                hour_df = subject_df.loc[(subject_df['corrected_start_time'] >= (start_time)) & (
                    subject_df['corrected_start_time'] < (end_time))]
                cough_df = hour_df.loc[(hour_df['event type'] == 'cough')]
                cough_activity_df = hour_df.loc[(
                    hour_df['event type'] == 'cough activity')]
                activity_df = hour_df.loc[(
                    hour_df['event type'] == 'activity')]

                valid_events = hour_valid_events.shape[0]
                cough_count = cough_df['cough count'].sum()
                cough_activity = utils.get_mean(
                    cough_activity_df['cough activity'])
                activity = activity_df['activity'].sum()

                dayInfo.setHour(hour, valid_events, cough_count,
                                cough_activity, activity)

                if valid_events > 0:
                    valid_events_list.append(valid_events)

                if cough_count > 0:
                    cough_count_list.append(cough_count)

                if cough_activity > 0:
                    cough_activity_list.append(cough_activity)

                if activity > 0:
                    activity_list.append(activity)

            dates.append(dayInfo)

    max_valid_events = calculateMax(valid_events_list)
    max_cough_count = calculateMax(cough_count_list)
    max_cough_activity = calculateMax(cough_activity_list)
    max_activity = calculateMax(activity_list)

    numOfDays = len(dates)

    # Mann-Whitney U-Teset
    numOfDays_2 = numOfDays//2
    cough_count_per_hour = [[[]
                             for _ in range(24)] for _ in range(2)]

    for i in range(numOfDays_2):
        dayInfo = dates[i]
        for hour in range(24):
            cough_count_per_hour[0][hour].append(dayInfo.coughCount()[hour])

    for i in range(numOfDays_2, numOfDays):
        dayInfo = dates[i]
        for hour in range(24):
            cough_count_per_hour[1][hour].append(dayInfo.coughCount()[hour])

    dist_0 = np.transpose(
        np.array(cough_count_per_hour[0]))
    dist_1 = np.transpose(
        np.array(cough_count_per_hour[1]))
    _, p = mannwhitneyu(dist_0, dist_1, method="exact")
    psum = sum(list(map(lambda x: 1 if x < 0.05 else 0, p)))
    print(f"Change over time p-value {psum}")

    features = np.zeros((numOfDays, FEATURE_SIZE))

    for i in range(numOfDays):
        dayInfo = dates[i]
        dayInfo.calculateFeatures(
            max_valid_events, max_cough_count, max_cough_activity, max_activity)

        features[i] = dayInfo.features()

    print(subject.get_id())
    # Calculate 14 - day chuncks
    one_day = datetime.timedelta(1)
    changes_over_time_start_dates = [subject.get_first_day()]
    changes_over_time_end_dates = []
    today = subject.get_first_day()
    days_in_group = 0
    days_group_size = 14
    while today <= subject.get_last_day():
        # print(today)
        tomorrow = today + one_day
        days_in_group += 1

        if days_in_group >= days_group_size:
            changes_over_time_start_dates.append(tomorrow)
            changes_over_time_end_dates.append(today)
            days_in_group = 0
        today = tomorrow

    changes_over_time_end_dates.append(subject.get_last_day())
    changes_over_time_chunks = list(
        zip(changes_over_time_start_dates, changes_over_time_end_dates))

    # Calculate 3 - month chuncks
    one_day = datetime.timedelta(1)
    months_3_start_dates = [subject.get_first_day()]
    months_3_end_dates = []
    today = subject.get_first_day()
    months_in_group = 0
    month_group_size = 3
    start_day_of_month = today.day
    cut_off_day = start_day_of_month
    # calendar.monthrange(2022, 9)
    while today <= subject.get_last_day():
        # print(today)
        tomorrow = today + one_day
        if tomorrow.month != today.month:
            months_in_group += 1

        if months_in_group == month_group_size:
            last_day_of_month = calendar.monthrange(
                tomorrow.year, tomorrow.month)[1]
            if start_day_of_month > last_day_of_month:
                cut_off_day = last_day_of_month

            if tomorrow.day >= cut_off_day:
                months_3_start_dates.append(tomorrow)
                months_3_end_dates.append(today)
                months_in_group = 0
        today = tomorrow

    months_3_end_dates.append(subject.get_last_day())

    month_3_chunks = list(zip(months_3_start_dates, months_3_end_dates))
    n_month_chunks = len(months_3_start_dates)

    month_cough_count_distributions = [[] for _ in range(n_month_chunks)]

    # T Test 3 - Month
    for i in range(numOfDays):
        dayInfo = dates[i]
        month_idx = 0
        for start, end in month_3_chunks:
            if dayInfo.date() >= start and dayInfo.date() <= end:
                break
            month_idx += 1

        month_cough_count_distributions[month_idx].extend(
            dayInfo._cough_count_distributions)

    month_t_tests = np.zeros((n_month_chunks, n_month_chunks))
    for i in range(n_month_chunks):
        month_i_dist = month_cough_count_distributions[i]

        for j in range(i, n_month_chunks):
            if i == j:
                month_t_tests[i][j] = 1.0
                continue

            month_j_dist = month_cough_count_distributions[j]

            _, p = ttest_ind(month_i_dist, month_j_dist)

            month_t_tests[i][j] = p
            month_t_tests[j][i] = p

    month_t_tests_columns = []
    for start, end in month_3_chunks:
        date_str = f"{start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}"
        month_t_tests_columns.append(date_str)

    month_t_tests_df = pd.DataFrame(
        month_t_tests, columns=month_t_tests_columns)

    month_t_tests_df.insert(
        loc=0, column='date range', value=month_t_tests_columns)

    month_t_tests_df.to_csv(
        f'{RESULTS_DIR}{subject.get_id()}_3_month_ttest.csv')

    # Clusters
    for n_clusters in range(2, 6):
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10)
        try:
            kmeans.fit(features)
        except:
            continue

        clusters = kmeans.fit_predict(features)
        clusterError = getClusterError(
            dates, clusters, kmeans.cluster_centers_)
        clusterSizes = np.unique(clusters, return_counts=True)[1]

        print(f'Clusters: {n_clusters}')
        print(f'Mean: {np.mean(clusterError)}')
        print(f'Max: {np.max(clusterError)}')
        print(f'Cluster Sizes: {clusterSizes}')

        clustersToSmall = np.vectorize(
            lambda x: 1 if x <= 1 else 0)(clusterSizes).sum()

        if clustersToSmall > 0:
            print('Skipping because cluster is to small!')
            continue

        # pca_num_components = 2

        # reduced_data = PCA(
        #     n_components=pca_num_components).fit_transform(features)
        # results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])

        # sns.scatterplot(x="pca1", y="pca2", hue=clusters, data=results)
        # plt.title('K-means Clustering with 2 dimensions')
        # plt.savefig(
        #     f'{RESULTS_DIR}{subject.get_id()}_{n_clusters}_clusters.jpg')
        # plt.close()

        valid_events_distributions = []
        cough_count_distributions = []
        cough_activity_distributions = []
        activity_distributions = []
        valid_events_cluster_labels = []
        cough_count_cluster_labels = []
        cough_activity_cluster_labels = []
        activity_cluster_labels = []

        cough_count_per_hour = [[[]
                                 for _ in range(24)] for _ in range(n_clusters)]

        cough_activity_per_hour = [[[]
                                    for _ in range(24)] for _ in range(n_clusters)]

        valid_events_per_hour = [[[]
                                  for _ in range(24)] for _ in range(n_clusters)]

        cough_count_distributions_cmb = [[] for _ in range(n_clusters)]

        month_chunk_info = [
            {
                "start": start,
                "end": end,
                "cough_count_per_hour": [[[]
                                          for _ in range(24)] for _ in range(n_clusters)],
                "cough_activity_per_hour": [[[]
                                             for _ in range(24)] for _ in range(n_clusters)],
                "cluster_breakdown": np.zeros(n_clusters)
            } for start, end in month_3_chunks
        ]

        changes_over_time_chunk_info = [
            {
                "start": start,
                "end": end,
                "days": 0,
                "cough_count_per_day": [],
                "avg_cough_activity_per_day": [],
                "cluster_breakdown": np.zeros(n_clusters)
            } for start, end in changes_over_time_chunks
        ]

        for i in range(numOfDays):
            dayInfo = dates[i]
            cluster = clusters[i]

            # Changes over time
            changes_over_time_info = None
            for chunk in changes_over_time_chunk_info:
                start = chunk["start"]
                end = chunk["end"]

                if dayInfo.date() >= start and dayInfo.date() <= end:
                    changes_over_time_info = chunk
                    break

            changes_over_time_info["days"] += 1
            changes_over_time_info["cluster_breakdown"][cluster] += 1
            changes_over_time_info["cough_count_per_day"].append(
                dayInfo._total_cough_count)
            changes_over_time_info["avg_cough_activity_per_day"].append(
                np.mean(dayInfo._cough_activity_events))

            # Month info
            month_info = None
            for chunk in month_chunk_info:
                start = chunk["start"]
                end = chunk["end"]

                if dayInfo.date() >= start and dayInfo.date() <= end:
                    month_info = chunk
                    break

            month_info["cluster_breakdown"][cluster] += 1

            for hour in range(24):
                cough_count_per_hour[cluster][hour].append(dayInfo.coughCount()[
                    hour])
                cough_activity_per_hour[cluster][hour].append(dayInfo.coughActivity()[
                    hour])
                valid_events_per_hour[cluster][hour].append(
                    dayInfo.validEvents()[hour])

                month_info["cough_count_per_hour"][cluster][hour].append(dayInfo.coughCount()[
                    hour])
                month_info["cough_activity_per_hour"][cluster][hour].append(dayInfo.coughActivity()[
                    hour])

            num_of_valid_events = len(dayInfo._valid_events_distributions)
            num_of_cough_count = len(dayInfo._cough_count_distributions)
            num_of_cough_activity = len(dayInfo._cough_activity_distributions)
            num_of_activity = len(dayInfo._activity_distributions)

            valid_events_distributions.extend(
                dayInfo._valid_events_distributions)
            valid_events_cluster_labels.extend(
                [cluster for _ in range(num_of_valid_events)])

            cough_count_distributions.extend(
                dayInfo._cough_count_distributions)
            cough_count_cluster_labels.extend(
                [cluster for _ in range(num_of_cough_count)])
            cough_count_distributions_cmb[cluster].extend(
                dayInfo._cough_count_distributions)

            cough_activity_distributions.extend(
                dayInfo._cough_activity_distributions)
            cough_activity_cluster_labels.extend(
                [cluster for _ in range(num_of_cough_activity)])

            activity_distributions.extend(dayInfo._activity_distributions)
            activity_cluster_labels.extend(
                [cluster for _ in range(num_of_activity)])

        # graphDistributions(
        #     valid_events_distributions,
        #     valid_events_cluster_labels,
        #     f'{RESULTS_DIR}{subject.get_id()}_valid_events_{n_clusters}_clusters.jpg'
        # )

        # graphDistributions(
        #     cough_count_distributions,
        #     cough_count_cluster_labels,
        #     f'{RESULTS_DIR}{subject.get_id()}_cough_count_{n_clusters}_clusters.jpg'
        # )

        # graphDistributions(
        #     cough_activity_distributions,
        #     cough_activity_cluster_labels,
        #     f'{RESULTS_DIR}{subject.get_id()}_cough_activity_{n_clusters}_clusters.jpg'
        # )

        # graphDistributions(
        #     activity_distributions,
        #     activity_cluster_labels,
        #     f'{RESULTS_DIR}{subject.get_id()}_activity_{n_clusters}_clusters.jpg'
        # )

        # Get cluster info
        cluster_infos = []
        for cluster in range(n_clusters):
            combined = list(zip(cough_count_distributions,
                                cough_count_cluster_labels))
            cluster_cough_count_distribution = list(
                map(lambda x: x[0],
                    filter(lambda x: x[1] == cluster,
                           combined
                           )
                    )
            )
            cluster_cough_count_distribution = np.array(
                cluster_cough_count_distribution)

            median, lower, upper = calculateDistributionInfo(
                cluster_cough_count_distribution)

            cluster_infos.append({
                "cluster": cluster,
                "median": median,
                "lower": lower,
                "upper": upper
            })

        # Sort clusters based on median
        cluster_infos = sorted(cluster_infos, key=lambda x: x["median"])
        cluster_map = {}
        for cluster in range(n_clusters):
            cluster_info = cluster_infos[cluster]
            cluster_map[cluster_info["cluster"]] = cluster

        # Find days that are outliers
        outliers = np.zeros(len(dates))
        for i in range(numOfDays):
            dayInfo = dates[i]
            cluster = clusters[i]

            day_cough_count = dayInfo.coughCount()
            day_cough_count_dist = cough_count_per_hour[cluster]

            outlier_count = 0
            for hour in range(24):
                sample = day_cough_count[hour]
                median, lower, upper = calculateDistributionInfo(
                    np.array(day_cough_count_dist[hour]))

                if (sample > upper):
                    outlier_count += 1

            if (outlier_count > 3):
                outliers[i] = 1.0

        # Mann-Whitney U Test
        print("Running Mann-Whitney U Test...")

        def getGreaterThan(cluster, threshold=0):
            cluster_hours_greater_than = 0
            for hour in range(24):
                dist = np.array(
                    cluster[hour])
                _, _, upper = calculateDistributionInfo(dist)
                if (upper > threshold):
                    cluster_hours_greater_than += 1
            return cluster_hours_greater_than

        mannwhitneyuData = {
            "Cluster i": [],
            "Cluster j": [],
            "p-value": [],
            "Cluster i Hours > 0": [],
            "Cluster j Hours > 0": [],
            "Total p-values < 0.05": [],
        }

        for i in range(24):
            mannwhitneyuData[f'p-value < 0.05 Hour {i}'] = []

        for i in range(24):
            mannwhitneyuData[f'p-value Hour {i}'] = []

        for i in range(n_clusters-1):
            cluster_i_id = cluster_infos[i]["cluster"]
            cluster_i = np.transpose(
                np.array(cough_count_per_hour[cluster_i_id]))
            cluster_i_dist = np.array(
                cough_count_distributions_cmb[cluster_i_id])

            cluster_i_hours_greater_than_0 = getGreaterThan(
                cough_count_per_hour[cluster_i_id], 0)

            for j in range(i+1, n_clusters):
                cluster_j_id = cluster_infos[j]["cluster"]
                cluster_j = np.transpose(
                    np.array(cough_count_per_hour[cluster_j_id]))
                cluster_j_dist = np.array(
                    cough_count_distributions_cmb[cluster_j_id])

                cluster_j_hours_greater_than_0 = getGreaterThan(
                    cough_count_per_hour[cluster_j_id], 0)

                _, pvalue = ttest_ind(cluster_i_dist, cluster_j_dist)
                _, phour = mannwhitneyu(cluster_i, cluster_j, method="exact")

                mannwhitneyuData["Cluster i"].append(i+1)
                mannwhitneyuData["Cluster j"].append(j+1)

                mannwhitneyuData["p-value"].append(pvalue)

                mannwhitneyuData["Cluster i Hours > 0"].append(
                    cluster_i_hours_greater_than_0)
                mannwhitneyuData["Cluster j Hours > 0"].append(
                    cluster_j_hours_greater_than_0)

                totalPValuesLessThan = 0
                for k in range(24):
                    difference = 1 if phour[k] < 0.05 else 0
                    totalPValuesLessThan += difference
                    mannwhitneyuData[f'p-value < 0.05 Hour {k}'].append(
                        difference)
                    mannwhitneyuData[f'p-value Hour {k}'].append(phour[k])

                mannwhitneyuData["Total p-values < 0.05"].append(
                    totalPValuesLessThan)

        mannwhitneydf = pd.DataFrame(data=mannwhitneyuData)
        mannwhitneydf.to_csv(
            f'{RESULTS_DIR}{subject.get_id()}_{n_clusters}_mannwhitneyu.csv')

        # Stats for 3 month chunks
        month_chunks_data = {
            "start date": [],
            "end date": [],
            "days": [],
            "days with data": [],
            "days without data": []
        }

        for i in range(n_clusters):
            month_chunks_data[f"cluster {i+1}"] = []

        for chunk in month_chunk_info:
            start = chunk["start"]
            end = chunk["end"]
            days = (end - start).days
            cluster_breakdown = chunk["cluster_breakdown"]
            days_with_data = sum(cluster_breakdown)
            days_without_data = days - days_with_data

            month_chunks_data["start date"].append(start)
            month_chunks_data["end date"].append(end)
            month_chunks_data["days"].append(days)
            month_chunks_data["days with data"].append(days_with_data)
            month_chunks_data["days without data"].append(days_without_data)

            for i in range(n_clusters):
                clusterId = cluster_infos[i]["cluster"]
                month_chunks_data[f"cluster {i+1}"].append(
                    cluster_breakdown[clusterId])

        month_chunks_df = pd.DataFrame(data=month_chunks_data)
        month_chunks_df.to_csv(
            f'{RESULTS_DIR}{subject.get_id()}_{n_clusters}_3_month_info.csv')

        # Changes over time
        max_avg_cough_count = 0
        max_avg_cough_activity = 0
        max_std_cough_count = 0
        max_std_cough_activity = 0
        changes_over_time_stats = []
        for chunk in changes_over_time_chunk_info:
            stats = {}

            start = chunk["start"]
            end = chunk["end"]
            days = chunk["days"]

            stats["start"] = start
            stats["end"] = end
            stats["days"] = days / 14

            if days <= 0:
                stats["avg_cough_count"] = 0
                stats["avg_cough_activity"] = 0
                stats["std_cough_count"] = 0
                stats["std_cough_activity"] = 0
                stats["cluster_breakdown"] = chunk["cluster_breakdown"]
                continue

            stats["cluster_breakdown"] = chunk["cluster_breakdown"] / days

            avg_cough_count = np.mean(chunk["cough_count_per_day"])
            avg_cough_activity = np.mean(chunk["avg_cough_activity_per_day"])

            std_cough_count = np.std(chunk["cough_count_per_day"])
            std_cough_activity = np.std(chunk["avg_cough_activity_per_day"])

            if avg_cough_count > max_avg_cough_count:
                max_avg_cough_count = avg_cough_count

            if avg_cough_activity > max_avg_cough_activity:
                max_avg_cough_activity = avg_cough_activity

            if std_cough_count > max_std_cough_count:
                max_std_cough_count = std_cough_count

            if std_cough_activity > max_std_cough_activity:
                max_std_cough_activity = std_cough_activity

            stats["avg_cough_count"] = avg_cough_count
            stats["avg_cough_activity"] = avg_cough_activity

            stats["std_cough_count"] = std_cough_count
            stats["std_cough_activity"] = std_cough_activity

            changes_over_time_stats.append(stats)

        # Normailize stats
        for stats in changes_over_time_stats:
            stats["avg_cough_count"] /= max_avg_cough_count
            stats["avg_cough_activity"] /= max_avg_cough_activity

            stats["std_cough_count"] /= max_std_cough_count
            stats["std_cough_activity"] /= max_std_cough_activity

        # Plot
        print("Plotting...")
        # Plot changes over time
        changes_over_time_bar_plot_dates = []

        changes_usage_stats = {
            'Usage': [],
        }

        changes_stats = {
            'Cough Count': [],
            'Cough Count STD': [],
            'Cough Activity': [],
            'Cough Activity STD': [],
        }

        changes_cluster_stats = {}

        for i in range(n_clusters):
            changes_cluster_stats[f"C{i+1}"] = []

        for stats in changes_over_time_stats:
            changes_over_time_bar_plot_dates.append(stats["start"])

            changes_usage_stats["Usage"].append(stats["days"])

            changes_stats["Cough Count"].append(stats["avg_cough_count"])
            changes_stats["Cough Count STD"].append(stats["std_cough_count"])
            changes_stats["Cough Activity"].append(stats["avg_cough_activity"])
            changes_stats["Cough Activity STD"].append(
                stats["std_cough_activity"])

            for i in range(n_clusters):
                clusterId = cluster_infos[i]["cluster"]
                changes_cluster_stats[f"C{i+1}"].append(
                    stats["cluster_breakdown"][clusterId])

        # the label locations
        x = np.arange(len(changes_over_time_bar_plot_dates))
        width = 0.1  # the width of the bars

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=CHANGES_FIG_SIZE, layout='constrained')

        # Usage Stats
        multiplier = 0
        for attribute, measurement in changes_usage_stats.items():
            offset = width * multiplier
            rects = ax1.bar(x + offset, measurement, width, label=attribute)
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        # Stats
        multiplier = 0
        for attribute, measurement in changes_stats.items():
            offset = width * multiplier
            rects = ax2.bar(x + offset, measurement, width, label=attribute)
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        # Clusters
        multiplier = 0
        for attribute, measurement in changes_cluster_stats.items():
            offset = width * multiplier
            rects = ax3.bar(x + offset, measurement, width, label=attribute)
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel('Magnitu')
        ax1.set_title('Usage Over Time')
        ax1.set_xticks(x + width, changes_over_time_bar_plot_dates)
        ax1.tick_params(axis='x', labelrotation=45)
        ax1.legend(loc='upper left', ncols=3)
        ax1.set_ylim(0, 1.5)

        ax2.set_title('Changes Over Time')
        ax2.set_xticks(x + width, changes_over_time_bar_plot_dates)
        ax2.tick_params(axis='x', labelrotation=45)
        ax2.legend(loc='upper left', ncols=3)
        ax2.set_ylim(0, 1.5)

        ax3.set_title('Changes Over Time of Clusters')
        ax3.set_xticks(x + width, changes_over_time_bar_plot_dates)
        ax3.tick_params(axis='x', labelrotation=45)
        ax3.legend(loc='upper left', ncols=3)
        ax3.set_ylim(0, 1.5)

        plt.savefig(
            f'{RESULTS_DIR}{subject.get_id()}_{n_clusters}_changes_over_time.jpg', bbox_inches="tight")
        plt.close()

        # Plot clusters
        fig = plt.figure(figsize=FIG_SIZE)
        height_ratios = (1.0 - 1/(n_clusters+1),)
        for i in range(n_clusters):
            height_ratios += (1,)
        gs = fig.add_gridspec(n_clusters+1, 3,  width_ratios=(15, 15, 1), height_ratios=height_ratios,
                              left=0.15, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.15, hspace=n_clusters*0.15)

        ax1 = plt.subplot(gs[0, :2])

        totalNumDays = (subject.get_last_day() - subject.get_first_day()).days
        clusterLineData = np.zeros((2, totalNumDays))
        clusterLineLabels = []

        for days in range(totalNumDays):
            date = subject.get_first_day() + datetime.timedelta(days=days)
            clusterLineLabels.append(date)

        for i in range(numOfDays):
            dayInfo = dates[i]
            cluster = cluster_map[clusters[i]]
            outlier = outliers[i]

            dayIndex = (dayInfo.date() - subject.get_first_day()).days

            clusterLineData[0, dayIndex] = cluster + 1
            clusterLineData[1, dayIndex] = outlier * (n_clusters + 1)

        colors = [(1, 1, 1), (0.71, 0.84, 0.77), (0.69, 0.55, 0.73),
                  (0.93, 0.66, 0.41), (0.40, 0.42, 0.78), (0.80, 0.42, 0.78)]
        outlier_color = (0.15, 0.16, 0.34)
        colors = colors[:(n_clusters+1)]
        colors.append(outlier_color)
        cmap = LinearSegmentedColormap.from_list(
            "CustomPastel", colors, N=n_clusters+2)

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
        patches.append(mpatches.Patch(
            color=colors[-1], label="Outlier"))
        # put those patched as legend-handles into the legend
        ax1.legend(handles=patches, facecolor='white',
                   bbox_to_anchor=(1.05, 1), borderaxespad=0)

        labels = [i for i in range(24)]
        lastCoughCountAx = None
        lastCoughActivityAx = None
        for cluster in range(n_clusters):
            clusterId = cluster_infos[cluster]["cluster"]
            # Save time of day cough count box plot
            if (lastCoughCountAx == None):
                ax = plt.subplot(gs[cluster+1, 0])
                lastCoughCountAx = ax
            else:
                ax = plt.subplot(gs[cluster+1, 0],
                                 sharey=lastCoughCountAx)

            ax.boxplot(
                cough_count_per_hour[clusterId], labels=labels, showfliers=False)
            ax.set_ylabel('Cough Count')
            ax.tick_params(axis='x', labelrotation=45)
            ax.set_title(
                f'Cluster {cluster+1} - Time of Day Cough Count')
            ax.yaxis.grid(True)

            # Save time of day cough activity box plot
            if (lastCoughActivityAx == None):
                ax = plt.subplot(gs[cluster+1, 1])
                lastCoughActivityAx = ax
            else:
                ax = plt.subplot(gs[cluster+1, 1],
                                 sharey=lastCoughActivityAx)

            ax.boxplot(
                valid_events_per_hour[clusterId], labels=labels, showfliers=False)
            ax.set_ylabel('Valid Events')
            ax.tick_params(axis='x', labelrotation=45)
            ax.set_title(
                f'Cluster {cluster+1} - Time of Day Valid Events')
            ax.yaxis.grid(True)

            # Save cluster size
            ax = plt.subplot(gs[cluster+1, 2])

            ax.bar([f'Cluster {cluster+1}'],
                   [len(cough_count_per_hour[clusterId][0])])
            ax.set_ylabel('Days')
            ax.set_ylim([0, numOfDays])
            ax.yaxis.grid(True)
        # plt.subplots_adjust(right=0.7)
        # plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.savefig(
            f'{RESULTS_DIR}{subject.get_id()}_{n_clusters}_time_of_day_box_plot.jpg', bbox_inches="tight")
        plt.close()


def main():
    print("Loading Data...")
    df = pd.read_csv(CSV_PATH)

    subjects = {}
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        subject_id = row['subject_id']
        event_id = row['id']

        start_time = row['corrected_start_time']
        event_type = row['event type']

        if subject_id not in subjects:
            subjects[subject_id] = Subject(subject_id)

        subjects[subject_id].add_activity(
            start_time, event_id, event_type, row)

    # Subject Stats
    print("Caculating Stats for Subjects...")
    for subject_id in tqdm(subjects):
        # Get subject data
        subject = subjects[subject_id]

        # Calculating
        calculate(subject)


def test():
    x = np.random.rand(100)
    x = np.concatenate([x, 100*np.random.rand(5) + 50])
    print(x)

    print(calculateMax(x))


if __name__ == "__main__":
    main()
