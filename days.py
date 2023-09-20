import numpy as np
import pandas as pd
import datetime

from subject import SUBJECT_ACTIVE_TIME
from day_info import DayInfo

import utils
import time_utils


def create_days(subject):
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

    # Sort by date
    dates.sort(key=lambda x: x.date())

    max_valid_events = utils.calculateMax(valid_events_list)
    max_cough_count = utils.calculateMax(cough_count_list)
    max_cough_activity = utils.calculateMax(cough_activity_list)
    max_activity = utils.calculateMax(activity_list)

    for dayInfo in dates:
        dayInfo.calculateFeatures(
            max_valid_events, max_cough_count, max_cough_activity, max_activity)

    return dates


def dates_to_table(dates):
    date_label = []
    start_time = []
    usage_time = []
    total_cough = []
    avg_cough_activity = []
    avg_activity = []

    for dayInfo in dates:
        date_label.append(dayInfo.date())

        day_usage = dayInfo.estimated_usage()

        start_time.append(dayInfo.start_time())
        usage_time.append(day_usage.sum())
        total_cough.append(dayInfo.coughCount().sum())

        day_avg_cough_activity = dayInfo.coughActivity().sum() / day_usage.sum()
        day_avg_activity = dayInfo.activity().sum() / day_usage.sum()

        avg_cough_activity.append(day_avg_cough_activity)
        avg_activity.append(day_avg_activity)

    df = pd.DataFrame(data={
        "date": date_label,
        "start time": start_time,
        "hours used": usage_time,
        "total cough count": total_cough,
        "average cough activity": avg_cough_activity,
        "average activity": avg_activity
    })

    return df


def calculate_per_hour(dates, ignore_ends=False):
    # Calculate the totals
    total_cough_count_per_hour = np.zeros((24,))
    total_cough_activity_per_hour = np.zeros((24,))
    total_activity_per_hour = np.zeros((24,))
    total_usage_per_hour = np.zeros((24,))

    for dayInfo in dates:
        for hour in range(24):
            total_cough_count_per_hour[hour] += dayInfo.coughCount()[hour]
            total_cough_activity_per_hour[hour] += dayInfo.coughActivity()[
                hour]
            total_activity_per_hour[hour] += dayInfo.activity()[hour]
            total_usage_per_hour[hour] += dayInfo.estimated_usage(
                remove_ends=ignore_ends)[hour]

    # Normalize with usage
    # Find averages to fill in missing data
    avg_cough_count_per_hour = np.zeros((24,))
    avg_cough_activity_per_hour = np.zeros((24,))
    avg_activity_per_hour = np.zeros((24,))

    for hour in range(24):
        if total_usage_per_hour[hour] >= 2:
            avg_cough_count_per_hour[hour] = total_cough_count_per_hour[hour] / \
                total_usage_per_hour[hour]
            avg_cough_activity_per_hour[hour] = total_cough_activity_per_hour[hour] / \
                total_usage_per_hour[hour]
            avg_activity_per_hour[hour] = total_activity_per_hour[hour] / \
                total_usage_per_hour[hour]

    # Create distributions
    num_dates = len(dates)
    date_labels = []

    cough_count_per_hour = [[] for _ in range(24)]
    cough_activity_per_hour = [[] for _ in range(24)]
    activity_per_hour = [[] for _ in range(24)]

    for i in range(num_dates):
        dayInfo = dates[i]
        date_labels.append(dayInfo.date())
        for hour in range(24):
            usage = dayInfo.estimated_usage(
                remove_ends=ignore_ends)[hour]

            # If the device is used then use day info
            if usage:
                cough_count_per_hour[hour].append(dayInfo.coughCount()[hour])
                cough_activity_per_hour[hour].append(dayInfo.coughActivity()[
                    hour])
                activity_per_hour[hour].append(dayInfo.activity()[hour])

    results = {
        "dates": date_labels,
        "size": num_dates,
        "ignore_ends": ignore_ends,
        "total_cough_count_per_hour": total_cough_count_per_hour,
        "total_cough_activity_per_hour": total_cough_activity_per_hour,
        "total_activity_per_hour": total_activity_per_hour,
        "total_usage_per_hour": total_usage_per_hour,
        "avg_cough_count_per_hour": avg_cough_count_per_hour,
        "avg_cough_activity_per_hour": avg_cough_activity_per_hour,
        "avg_activity_per_hour": avg_activity_per_hour,
        "cough_count_per_hour": cough_count_per_hour,
        "cough_activity_per_hour": cough_activity_per_hour,
        "activity_per_hour": activity_per_hour,
    }

    return results


def calculate_per_day_summary(dates, dates_per_hour=None, min_hours=5):
    if dates_per_hour is None:
        dates_per_hour = calculate_per_hour(dates)

    num_days = len(dates)

    cough_count_avg_per_day = np.zeros((num_days,))
    cough_count_max_per_day = np.zeros((num_days,))
    cough_count_min_per_day = np.zeros((num_days,))

    cough_activity_avg_per_day = np.zeros((num_days,))
    cough_activity_max_per_day = np.zeros((num_days,))
    cough_activity_min_per_day = np.zeros((num_days,))

    activity_avg_per_day = np.zeros((num_days,))
    activity_max_per_day = np.zeros((num_days,))
    activity_min_per_day = np.zeros((num_days,))

    date_labels = []
    for i in range(num_days):
        dayInfo = dates[i]

        date_labels.append(dayInfo.date())

        usage = dayInfo.estimated_usage()

        # Cough count
        day_cough_count = dayInfo.coughCount()
        day_cough_count_avg = 0
        day_cough_count_max = 0
        day_cough_count_min = 1000

        for hour in range(24):
            if usage[hour] and dates_per_hour["total_usage_per_hour"][hour] >= min_hours:
                # Normalize hour
                hour_cough_count = 0

                if dates_per_hour["avg_cough_count_per_hour"][hour] > 0:
                    hour_cough_count = day_cough_count[hour]
                    hour_cough_count = hour_cough_count / \
                        dates_per_hour["avg_cough_count_per_hour"][hour]

                day_cough_count_avg += hour_cough_count
                day_cough_count_max = max(
                    day_cough_count_max, hour_cough_count)
                day_cough_count_min = min(
                    day_cough_count_min, hour_cough_count)

        day_cough_count_avg /= usage.sum()

        cough_count_avg_per_day[i] = day_cough_count_avg
        cough_count_max_per_day[i] = day_cough_count_max
        cough_count_min_per_day[i] = day_cough_count_min

        # Cough actitivty
        day_cough_activity = dayInfo.coughActivity()
        day_cough_activity_avg = 0
        day_cough_activity_max = 0
        day_cough_activity_min = 1000

        for hour in range(24):
            if usage[hour] and dates_per_hour["total_usage_per_hour"][hour] >= min_hours:
                # Normalize hour
                hour_cough_activity = 0

                if dates_per_hour["avg_cough_activity_per_hour"][hour] > 0:
                    hour_cough_activity = day_cough_activity[hour]
                    hour_cough_activity = hour_cough_activity / \
                        dates_per_hour["avg_cough_activity_per_hour"][hour]

                day_cough_activity_avg += hour_cough_activity
                day_cough_activity_max = max(
                    day_cough_activity_max, hour_cough_activity)
                day_cough_activity_min = min(
                    day_cough_activity_min, hour_cough_activity)

        day_cough_activity_avg /= usage.sum()

        cough_activity_avg_per_day[i] = day_cough_activity_avg
        cough_activity_max_per_day[i] = day_cough_activity_max
        cough_activity_min_per_day[i] = day_cough_activity_min

        # Actitivty
        day_activity = dayInfo.activity()
        day_activity_avg = 0
        day_activity_max = 0
        day_activity_min = 1000

        for hour in range(24):
            if usage[hour] and dates_per_hour["total_usage_per_hour"][hour] >= min_hours:
                # Normalize hour
                hour_activity = 0

                if dates_per_hour["avg_activity_per_hour"][hour] > 0:
                    hour_activity = day_activity[hour]
                    hour_activity = hour_activity / \
                        dates_per_hour["avg_activity_per_hour"][hour]

                day_activity_avg += hour_activity
                day_activity_max = max(
                    day_activity_max, hour_activity)
                day_activity_min = min(
                    day_activity_min, hour_activity)

        day_activity_avg /= usage.sum()

        activity_avg_per_day[i] = day_activity_avg
        activity_max_per_day[i] = day_activity_max
        activity_min_per_day[i] = day_activity_min

    results = {
        "dates": date_labels,
        "size": num_days,
        "cough_count_avg_per_day": cough_count_avg_per_day,
        "cough_count_max_per_day": cough_count_max_per_day,
        "cough_count_min_per_day": cough_count_min_per_day,
        "cough_activity_avg_per_day": cough_activity_avg_per_day,
        "cough_activity_max_per_day": cough_activity_max_per_day,
        "cough_activity_min_per_day": cough_activity_min_per_day,
        "activity_avg_per_day": activity_avg_per_day,
        "activity_max_per_day": activity_max_per_day,
        "activity_min_per_day": activity_min_per_day
    }

    return results