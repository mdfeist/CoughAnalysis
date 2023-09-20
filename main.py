from os import makedirs

import numpy as np

from tqdm import tqdm

from subject import load_from_csv
import days

import graph


DATA_DIR = "data_oct_2022/"
CSV_PATH = DATA_DIR + "cleaned.csv"
RESULTS_DIR = "results/main/"

makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("Loading Data...")
    # Load CSV and split data based on subject ID
    subjects = load_from_csv(CSV_PATH)

    # Subject stats
    print("Caculating Stats for Subjects...")
    for subject_id in tqdm(sorted(subjects.keys())):
        # Get subject
        subject = subjects[subject_id]
        print(subject.get_id())

        # Formatting dates for subject into days
        # Returns a list of DayInfo objects
        dates = days.create_days(subject)

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

        graph.estimate_box_plot_day_info(
            dates_per_hour,
            f'{RESULTS_DIR}{subject.get_id()}_dates_estimated_box.jpg',
            title=subject.get_id()
        )

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
        dates_changes = days.calculate_per_day_summary(dates, dates_per_hour)

        graph.changes_over_time_day_info(
            dates_changes,
            f'{RESULTS_DIR}{subject.get_id()}_dates_changes.jpg',
            title=subject.get_id()
        )

        # Average usage
        estimated_usage = []
        for dayInfo in dates:
            estimated_usage.append(dayInfo.estimated_usage().sum())

        print(f'Estimated Usage Mean: {np.mean(estimated_usage)}')
        print(f'Estimated Usage STD: {np.std(estimated_usage)}')


if __name__ == "__main__":
    main()