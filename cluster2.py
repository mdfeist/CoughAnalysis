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
from day_info import DayInfo, FEATURE_SIZE
from cluster_info import ClusterInfo
from days import create_days

import graph
import utils
import time_utils


DATA_DIR = "data_oct_2022/"
CSV_PATH = DATA_DIR + "cleaned.csv"
RESULTS_DIR = "results/cluster2/"

makedirs(RESULTS_DIR, exist_ok=True)


def create_features_array(dates):
    numOfDays = len(dates)

    features = np.zeros((numOfDays, FEATURE_SIZE))

    for i in range(numOfDays):
        features[i] = dates[i].features()

    return features


def create_clusters(dates, features, n_clusters):
    # Clusters
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10)
    kmeans.fit(features)

    cluster_ids = kmeans.fit_predict(features)
    clusterSizes = np.unique(cluster_ids, return_counts=True)[1]

    print(f'Clusters: {n_clusters}')
    print(f'Cluster Sizes: {clusterSizes}')

    clustersToSmall = np.vectorize(
        lambda x: 1 if x <= 2 else 0)(clusterSizes).sum()

    if clustersToSmall > 0:
        raise Exception('Skipping because cluster is to small!')

    # Create Cluster Objects
    numOfDays = len(dates)
    clusters = [ClusterInfo(i) for i in range(n_clusters)]

    for i in range(numOfDays):
        dayInfo = dates[i]
        cluster_id = cluster_ids[i]
        cluster = clusters[cluster_id]

        cluster.add_date(dayInfo)

    # Sort by Size
    clusters.sort(key=lambda x: x.size(), reverse=True)

    # Sort Dates
    for cluster in clusters:
        cluster.sort_dates()

    return clusters


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
    for subject_id in tqdm(sorted(subjects.keys())):
        # Get subject data
        subject = subjects[subject_id]

        print(subject.get_id())

        # Formatting Dates
        dates = create_days(subject)

        # Graph Dates
        graph.graph_day_info(
            dates,
            f'{RESULTS_DIR}{subject.get_id()}_dates.jpg',
            title=subject.get_id(),
            normalize=False
        )

        graph.graph_day_info(
            dates,
            f'{RESULTS_DIR}{subject.get_id()}_dates_normalized.jpg',
            title=subject.get_id(),
            normalize=True
        )

        # Average Usage
        estimated_usage = []
        for dayInfo in dates:
            estimated_usage.append(dayInfo.estimated_usage().sum())

        print(f'Estimated Usage Mean: {np.mean(estimated_usage)}')
        print(f'Estimated Usage STD: {np.std(estimated_usage)}')

        # Create Features Array
        features = create_features_array(dates)

        # Create Clusters
        for n_clusters in range(2, 6):
            try:
                clusters = create_clusters(dates, features, n_clusters)

                for cluster in clusters:
                    print(cluster)
            except Exception as e:
                print(e)
                continue


if __name__ == "__main__":
    main()
