import numpy as np
import pandas as pd

import time_utils

FEATURE_SIZE = 24 * 2
USAGE_THRESHOLD = 6


class DayInfo:
    def __init__(self, date, day_df, valid_events_df) -> None:
        self._date = date
        self._day_df = day_df
        self._valid_events = np.zeros(24)
        self._cough_count = np.zeros(24)
        self._cough_activity = np.zeros(24)
        self._activity = np.zeros(24)
        self._estimated_usage = np.zeros(24)

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

    def setHour(self, hour, valid_events, cough_count, cough_activity, activity):
        self._valid_events[hour] = valid_events
        self._cough_count[hour] = cough_count
        self._cough_activity[hour] = cough_activity
        self._activity[hour] = activity

        if (valid_events + cough_count) >= USAGE_THRESHOLD:
            self._estimated_usage[hour] = 1

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

    def estimated_usage(self, remove_ends=False):
        if remove_ends:
            usage = np.zeros_like(self._estimated_usage)

            for i in range(1, usage.size - 1):
                if self._estimated_usage[i-1] == 1 and \
                        self._estimated_usage[i] == 1 and \
                        self._estimated_usage[i+1] == 1:
                    usage[i] = 1

            return usage

        return self._estimated_usage

    def start_time(self):
        found_start_time = -1
        for hour in range(24):
            if self._estimated_usage[hour]:
                found_start_time = hour
                break

        return found_start_time
