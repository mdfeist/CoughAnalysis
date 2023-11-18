from os import makedirs

import numpy as np
import scipy.stats
import pandas as pd

from tqdm import tqdm

from subject import load_from_csv
import cluster
import days
import date_utils
import graph


DATA_DIR = "results/clean/"
CSV_PATH = DATA_DIR + "cleaned.csv"
RESULTS_DIR = "results/main/"

makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("Loading Data...")
    # Load CSV and split data based on subject ID
    subjects = load_from_csv(CSV_PATH)

    correlations_with_time = {
        "Patient": [],
        "Cough Count Correlation": [],
        "Cough Count P-Value": [],
        "Cough Activity Correlation": [],
        "Cough Activity P-Value": [],
        "Activity Correlation": [],
        "Activity P-Value": [],
    }

    correlations = {
        "Patient": [],
        "Cough Count with Cough Activity": [],
        "Cough Count with Cough Activity P-Value": [],
        "Cough Count with Activity": [],
        "Cough Count with Activity P-Value": [],
    }

    # Subject stats
    print("Caculating Stats for Subjects...")
    for subject_id in tqdm(sorted(subjects.keys())):
        # Get subject
        subject = subjects[subject_id]
        print(subject.get_id())

        # Formatting dates for subject into days
        # Returns a list of DayInfo objects
        dates = days.create_days(subject)

        if len(dates) <= 0:
            continue

        # Get summary of each day in a table
        # Includes date, start time, hours used,
        # total cough count, average cough activity, and
        # average activity for each day in dates
        df = days.dates_to_table(dates)
        print(df)

        # Returns a dictionary object with averages,
        # totals, and distributions for each hour
        dates_per_hour = days.calculate_per_hour(dates)

        # Graph dates
        graph.graph_day_info(
            dates_per_hour,
            f'{RESULTS_DIR}{subject.get_id()}_dates.jpg',
            title=subject.get_id()
        )

        # graph.estimate_box_plot_day_info(
        #     dates_per_hour,
        #     f'{RESULTS_DIR}{subject.get_id()}_dates_estimated_box.jpg',
        #     title=subject.get_id()
        # )

        # Each hour is first is normalized by the average for that hour.
        # This gives us a percentage for each hour in each day that the
        # hour differs from the norm.
        #
        # Then the average, minimum, and maximum for each day is returned.
        #
        # This method attempts to find changes while ignoring changes in
        # usage.
        #
        # For example:
        #
        # If the maximum cough count for a day is 2. Then there were twice
        # as many coughs than normal during one of the hours in the day.
        dates_summary = days.calculate_per_day_summary(
            dates, subject.get_first_day())

        if dates_summary is not None:
            graph.graph_day_summary(
                dates_summary,
                f'{RESULTS_DIR}{subject.get_id()}_dates_plot.jpg',
                title=subject.get_id()
            )

        day_v_cough_count_corr = scipy.stats.spearmanr(
            dates_summary["days_from_start"],
            dates_summary["cough_count_avg_per_day"])

        day_v_cough_activity_corr = scipy.stats.spearmanr(
            dates_summary["days_from_start"],
            dates_summary["cough_activity_avg_per_day"])

        day_v_activity_corr = scipy.stats.spearmanr(
            dates_summary["days_from_start"],
            dates_summary["activity_avg_per_day"])

        correlations_with_time["Patient"].append(subject_id)
        correlations_with_time["Cough Count Correlation"].append(
            day_v_cough_count_corr[0])
        correlations_with_time["Cough Count P-Value"].append(
            day_v_cough_count_corr[1])
        correlations_with_time["Cough Activity Correlation"].append(
            day_v_cough_activity_corr[0])
        correlations_with_time["Cough Activity P-Value"].append(
            day_v_cough_activity_corr[1])
        correlations_with_time["Activity Correlation"].append(
            day_v_activity_corr[0])
        correlations_with_time["Activity P-Value"].append(
            day_v_activity_corr[1])

        cough_count_v_cough_activity_corr = scipy.stats.spearmanr(
            dates_summary["cough_count_avg_per_day"],
            dates_summary["cough_count_avg_per_day"])

        cough_count_v_activity_corr = scipy.stats.spearmanr(
            dates_summary["cough_count_avg_per_day"],
            dates_summary["activity_avg_per_day"])

        correlations["Patient"].append(subject_id)
        correlations["Cough Count with Cough Activity"].append(
            cough_count_v_cough_activity_corr[0])
        correlations["Cough Count with Cough Activity P-Value"].append(
            cough_count_v_cough_activity_corr[1])
        correlations["Cough Count with Activity"].append(
            cough_count_v_activity_corr[0])
        correlations["Cough Count with Activity P-Value"].append(
            cough_count_v_activity_corr[1])

        # Average usage
        estimated_usage = []
        for dayInfo in dates:
            estimated_usage.append(dayInfo.estimated_usage().sum())

        print(f'Estimated Usage Mean: {np.mean(estimated_usage)}')
        print(f'Estimated Usage STD: {np.std(estimated_usage)}')

        # Usage clusters
        features = np.zeros((len(dates), 24))

        for i, date in enumerate(dates):
            features[i] = date.estimated_usage()

        try:
            clusters = cluster.create_clusters(dates, features, 3)

            graph.plot_usage_clusters(
                clusters, subject, f'{RESULTS_DIR}{subject.get_id()}_usage_clusters.jpg')
        except Exception as e:
            print(e)

        chunks_sizes = [14, 30, 60]

        for chunks_size in chunks_sizes:
            # Split dates into 30 day chunks
            time_chunks = date_utils.chunk_dates(dates,
                                                 subject.get_first_day(),
                                                 subject.get_last_day(),
                                                 chunks_size)

            for chunk in time_chunks:
                print(
                    f'{chunk["start"]} - {chunk["end"]} - {len(chunk["dates"])}')

            changes_between_chunks = days.calculate_changes_between_chunks(
                time_chunks,
                dates_per_hour
            )

            graph.changes_between_chunks_avg(
                changes_between_chunks,
                f'{RESULTS_DIR}{subject.get_id()}_dates_changes_{chunks_size}_days.jpg',
                title=subject.get_id()
            )

            graph.changes_between_chunks(
                changes_between_chunks,
                f'{RESULTS_DIR}{subject.get_id()}_dates_chunk_changes_{chunks_size}_days.jpg',
                title=subject.get_id()
            )

    # Save Correlation
    correlations = pd.DataFrame(data=correlations)
    correlations.to_csv(
        f'{RESULTS_DIR}correlations.csv')

    correlations_with_time = pd.DataFrame(data=correlations_with_time)
    correlations_with_time.to_csv(
        f'{RESULTS_DIR}time_correlations.csv')


if __name__ == "__main__":
    main()
