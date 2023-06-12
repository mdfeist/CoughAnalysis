import os.path
from os import makedirs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import datetime

from subject import Subject
import time_utils

EVENT_TYPES = [
    "activity",
    "asthma",
    "battery",
    "cough",
    "cough activity",
    "heartrate",
    "respiration",
    "temperature",
    "voice journal"
]

BATTERY_STATUS = [
    "battery",
    "charging"
]

DATA_DIR = "data_oct_2022/"
CSV_PATH = DATA_DIR + "cleaned.csv"
RESULTS_DIR = "results/check/"

makedirs(RESULTS_DIR, exist_ok=True)

FIG_SIZE = (40, 20)


def caculate_per_block(df, column, block_size=5, step_size=2.5):
    df = df.copy()
    results = []

    while df.shape[0] > 0:
        # Get min day start time
        first = df['corrected_start_time'].min()

        # Get x minute chunk
        per_block_df = df.loc[(df['corrected_start_time'] >= first) & (
            df['corrected_start_time'] < (first + block_size*60))]
        count = per_block_df[column].sum()

        if np.isnan(count):
            count = 0.0

        results.append(count)

        # Step x minutes
        df = df.loc[(df['corrected_start_time'] >= (first + step_size*60))]

    return results


def plot_cough_activity_per_day(subject):
    # Get subject data
    subject_df = subject.get_data()

    day_label = []

    # Sums
    day_cough_count_total = []
    day_activity_total = []

    # Distributions
    day_cough_count = []
    day_activity = []
    day_cough_activity = []
    day_cough_count_per_5 = []

    # Time of Day
    # Calculate bounds of when device was worn
    time_of_day_step_size = 60
    full_time_of_day_array_size = 1440 // time_of_day_step_size
    full_time_of_day_labels = []
    time_of_day_events = [[] for i in range(full_time_of_day_array_size)]

    for i in range(full_time_of_day_array_size):
        time = i * time_of_day_step_size
        h = int(time // 60)
        m = int(time % 60)
        full_time_of_day_labels.append(f'{h}:{m:02d}')

    for day_start in time_utils.daterange(subject.get_first_day(), subject.get_last_day()):
        day_end = day_start + datetime.timedelta(days=1)
        day_start_time = time_utils.convert_to_unix(day_start)
        day_end_time = time_utils.convert_to_unix(day_end)

        day_df = subject_df.loc[(subject_df['corrected_start_time'] >= day_start_time) & (
            subject_df['corrected_start_time'] < day_end_time)]
        cough_count_total = day_df['cough count'].sum()

        if np.isnan(cough_count_total):
            cough_count_total = 0.0

        if cough_count_total < 5:
            continue

        day_df = day_df.loc[
            (day_df['event type'] != 'cough') &
            (day_df['event type'] != 'cough activity') &
            (
                (day_df['event type'] != 'heartrate') |
                (
                    (day_df['event type'] == 'heartrate') & (
                        day_df['heart rate'] > 50)
                )
            ) &
            (
                (day_df['event type'] != 'temperature') |
                (
                    (day_df['event type'] == 'temperature') & (
                        day_df['skin temperature'] > 90)
                )
            )
        ]

        for i in range(full_time_of_day_array_size):
            time = (i * time_of_day_step_size) * 60
            start_time = day_start_time + time
            end_time = start_time + time_of_day_step_size * 60

            time_of_day_df = day_df.loc[(day_df['corrected_start_time'] >= start_time) & (
                day_df['corrected_start_time'] < end_time)]
            time_of_day_events[i].append(time_of_day_df.shape[0])

    # Find block and convert to minutes
    time_of_day_block = 1440

    # Create graph variables
    time_of_day_array_size = int(
        np.ceil(time_of_day_block / time_of_day_step_size))
    time_of_day_labels = []
    time_of_day_cough_count = [[] for i in range(time_of_day_array_size)]
    time_of_day_activity = [[] for i in range(time_of_day_array_size)]
    time_of_day_cough_activity = [[] for i in range(time_of_day_array_size)]

    for i in range(time_of_day_array_size):
        time = i * time_of_day_step_size  # + time_of_day_start
        h = int(time // 60)
        m = int(time % 60)
        time_of_day_labels.append(f'{h}:{m:02d}')

    for day_start in time_utils.daterange(subject.get_first_day(), subject.get_last_day()):
        day_end = day_start + datetime.timedelta(days=1)
        day_start_time = time_utils.convert_to_unix(day_start)
        day_end_time = time_utils.convert_to_unix(day_end)

        day_df = subject_df.loc[(subject_df['corrected_start_time'] >= day_start_time) & (
            subject_df['corrected_start_time'] < day_end_time)]
        cough_df = day_df.loc[(day_df['event type'] == 'cough')]
        activity_df = day_df.loc[(day_df['event type'] == 'activity')]
        cough_activity_df = day_df.loc[(
            day_df['event type'] == 'cough activity')]

        cough_count_total = cough_df['cough count'].sum()

        if np.isnan(cough_count_total):
            cough_count_total = 0.0

        day_str = day_start.strftime("%d/%m/%y")

        day_label.append(day_str)

        day_cough_count_total.append(cough_count_total)
        day_activity_total.append(sum(activity_df['activity']))

        day_cough_count.append(cough_df['cough count'].to_numpy())
        day_activity.append(activity_df['activity'].to_numpy())
        day_cough_activity.append(
            cough_activity_df['cough activity'].to_numpy())

        # Calcualte Cough per 5 minutes
        day_cough_count_per_5_day = caculate_per_block(
            cough_df, 'cough count', block_size=5, step_size=2.5)
        day_cough_count_per_5.append(day_cough_count_per_5_day)

        # Calculate Time of Day
        if cough_count_total > 5:
            for i in range(time_of_day_array_size):
                # time = (i * time_of_day_step_size + time_of_day_start) * 60
                time = (i * time_of_day_step_size) * 60
                start_time = day_start_time + time
                end_time = start_time + time_of_day_step_size * 60

                time_of_day_cough_df = cough_df.loc[(cough_df['corrected_start_time'] >= start_time) & (
                    cough_df['corrected_start_time'] < end_time)]
                time_of_day_activity_df = activity_df.loc[(activity_df['corrected_start_time'] >= start_time) & (
                    activity_df['corrected_start_time'] < end_time)]
                time_of_day_cough_activity_df = cough_activity_df.loc[(cough_activity_df['corrected_start_time'] >= start_time) & (
                    cough_activity_df['corrected_start_time'] < end_time)]

                time_of_day_cough_count[i].append(
                    sum(time_of_day_cough_df['cough count']))

                def get_mean(x):
                    if x.size > 0:
                        return np.mean(x)
                    return 0.0

                time_of_day_activity[i].append(
                    get_mean(time_of_day_activity_df['activity'].to_numpy()))
                time_of_day_cough_activity[i].append(
                    get_mean(time_of_day_cough_activity_df['cough activity'].to_numpy()))

    # Save cough count bar plot
    fig = plt.figure(figsize=FIG_SIZE)
    plt.bar(day_label, day_cough_count_total,
            align='center', alpha=0.5, capsize=10)
    ax = plt.gca()
    ax.set_ylabel('Cough Count')
    ax.set_title(f'{subject.get_id()} - Cough Count by Day')
    ax.tick_params(axis='x', labelrotation=45)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}{subject.get_id()}_cough_count_per_day.jpg')
    plt.close()

    # Save activity box plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.boxplot(day_activity, labels=day_label)
    ax.set_ylabel('Activity')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(f'{subject.get_id()} - Activity by Day')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(
        f'{RESULTS_DIR}{subject.get_id()}_activity_per_day_box_plot.jpg')
    plt.close()

    # Save activity count bar plot
    fig = plt.figure(figsize=FIG_SIZE)
    plt.bar(day_label, day_activity_total,
            align='center', alpha=0.5, capsize=10)
    ax = plt.gca()
    ax.set_ylabel('Activity Count')
    ax.set_title(f'{subject.get_id()} - Activity Count by Day')
    ax.tick_params(axis='x', labelrotation=45)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}{subject.get_id()}_activity_count_per_day.jpg')
    plt.close()

    # Save cough count and activity bar plot
    fig = plt.figure(figsize=FIG_SIZE)

    # Normalize
    day_cough_count_total_norm = day_cough_count_total / \
        np.max(day_cough_count_total)
    day_activity_total_norm = day_activity_total / np.max(day_activity_total)

    indices = np.arange(0, len(day_label))
    plt.bar(indices-0.2, day_cough_count_total_norm,
            0.4, label='Cough', capsize=10)
    plt.bar(indices+0.2, day_activity_total_norm,
            0.4, label='Activity', capsize=10)
    plt.xticks(indices, day_label)
    ax = plt.gca()
    ax.set_ylabel('Cough and Activity Count')
    ax.set_title(f'{subject.get_id()} - Cough and Activity Count by Day')
    ax.tick_params(axis='x', labelrotation=45)
    ax.yaxis.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f'{RESULTS_DIR}{subject.get_id()}_cough_and_activity_count_per_day.jpg')
    plt.close()

    # Save cough activity box plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.boxplot(day_cough_activity, labels=day_label)
    ax.set_ylabel('Cough Activity')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(f'{subject.get_id()} - Cough Activity by Day')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}{subject.get_id()}_cough_activity_per_day.jpg')
    plt.close()

    # Save cough count per 5 box plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.boxplot(day_cough_count_per_5, labels=day_label)
    ax.set_ylabel('Cough Count')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(f'{subject.get_id()} - Cough Count 5 Minute Blocks by Day')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(
        f'{RESULTS_DIR}{subject.get_id()}_cough_count_per_5_per_day.jpg')
    plt.close()

    # Save time of day cough count box plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.boxplot(time_of_day_cough_count, labels=time_of_day_labels)
    ax.set_ylabel('Cough Count')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(f'{subject.get_id()} - Time of Day Cough Count')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(
        f'{RESULTS_DIR}{subject.get_id()}_time_of_day_cough_count_box_plot.jpg')
    plt.close()

    # Save time of day activity box plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.boxplot(time_of_day_activity, labels=time_of_day_labels)
    ax.set_ylabel('Activity')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(f'{subject.get_id()} - Time of Day Activity')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(
        f'{RESULTS_DIR}{subject.get_id()}_time_of_day_activity_box_plot.jpg')
    plt.close()

    # Save time of day cough activity box plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.boxplot(time_of_day_cough_activity, labels=time_of_day_labels)
    ax.set_ylabel('Cough Activity')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(f'{subject.get_id()} - Time of Day Cough Activity')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(
        f'{RESULTS_DIR}{subject.get_id()}_time_of_day_cough_activity_box_plot.jpg')
    plt.close()

    # Save time of day events box plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.boxplot(time_of_day_events, labels=full_time_of_day_labels)
    ax.set_ylabel('Events')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(f'{subject.get_id()} - Time of Day Events')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(
        f'{RESULTS_DIR}{subject.get_id()}_time_of_day_events_box_plot.jpg')
    plt.close()


def plot_days_with_activity(subject):
    # Get days with activity
    subject_days = subject.get_days_with_activity()
    # Convert days to datetime
    subject_days = map(time_utils.convert_str_to_date, subject_days)
    # Sort days
    subject_days = sorted(subject_days)

    first_day = subject.get_first_day()
    last_day = subject.get_last_day()

    delta = last_day - first_day
    total_days = delta.days
    days_with_activity = len(subject_days)
    days_without_activity = total_days - days_with_activity

    first_day_str = first_day.strftime("%d/%m/%y")
    last_day_str = last_day.strftime("%d/%m/%y")

    title = f'{subject.get_id()} - Device usage from {first_day_str} to {last_day_str} ({total_days} Days in Total)'

    # Pie plot
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Days With Device', 'Days Without Device'
    sizes = [days_with_activity, days_without_activity]
    explode = (0.1, 0)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}{subject.get_id()}_days_with_device_pie.jpg')
    plt.close()


def plot_invalid_data(subject):
    valid_entries = subject.get_count()
    invalid_valid_entries = subject.get_invalid_count()
    # Pie plot
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Valid Entries', 'Invalid Entries'
    sizes = [valid_entries, invalid_valid_entries]
    explode = (0.1, 0)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    ax.set_title(f'{subject.get_id()} - Valid Dates')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}{subject.get_id()}_valid_dates_pie.jpg')
    plt.close()


def main():
    print("Loading Data...")
    df = pd.read_csv(CSV_PATH)

    subjects = {}
    missing_devices = set()
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        device_id = row['device_id']
        subject_id = row['subject_id']
        event_id = row['id']

        start_time = row['corrected_start_time']
        event_type = row['event type']

        if subject_id not in subjects:
            subjects[subject_id] = Subject(subject_id)

        subjects[subject_id].add_activity(
            start_time, event_id, event_type, row)

        if not os.path.exists(DATA_DIR + device_id):
            missing_devices.add(device_id)

    number_of_subjects = len(subjects)
    number_of_missing_devices = len(missing_devices)
    print(f'There were {number_of_subjects} subjects recorded.')
    print(
        f'There were {number_of_missing_devices} devices missing from raw data.')

    for device_id in sorted(missing_devices):
        print(f' - {device_id}')

    for key, subject in sorted(subjects.items()):
        print(f'{key} has {subject.get_count()} entries.')

    for key, subject in sorted(subjects.items()):
        print(f'{key} has {len(subject.get_days_with_activity())} days with activity.')
        print(
            f'{key} has {len(subject.get_weeks_with_activity())} weeks with activity.')

    # Subject Stats
    print("Caculating Stats for Subjects...")
    for subject_id in tqdm(subjects):
        # Get subject data
        subject = subjects[subject_id]

        # Calculating stats for coughing activity
        plot_cough_activity_per_day(subject)

        # Calcualte usage stats
        plot_days_with_activity(subject)


if __name__ == "__main__":
    main()
