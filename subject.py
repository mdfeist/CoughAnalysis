import datetime
import pytz

import pandas as pd

import device_info

SUBJECT_ACTIVE_TIME = {
    'EDM-001': (2, 17),
    'EDM-002': (10, 16),
    'EDM-003': (8, 15),
    'EDM-004': (8, 15),
    'EDM-005': (0, 23),
    'EDM-006': (0, 23),
    'EDM-007': (0, 23),
    'EDM-008': (0, 23)
}


class Subject:
    def __init__(self, subject_id):
        self._id = subject_id
        self._activity_count = 0
        self._days_with_activity = set()
        self._weeks_with_activity = set()
        self._first_day = datetime.date.today()
        self._last_day = datetime.date(2021, 1, 1)
        self._data = {
            "subject_id": [],
            "id": [],
            "start_time": [],
            "corrected_start_time": [],
            "event type": [],
            "respiration count": [],
            "heart rate": [],
            "skin temperature": [],
            "activity": [],
            "cough count": [],
            "cough activity": [],
            "severity level": []
        }

        self._time_of_day_with_cough = {}
        self._time_of_day_with_cough_activity = {}
        self._time_of_day_with_activity = {}

        self._setup_time_of_day(self._time_of_day_with_cough)
        self._setup_time_of_day(self._time_of_day_with_cough_activity)
        self._setup_time_of_day(self._time_of_day_with_activity)

    def _setup_time_of_day(self, x, step_size=15):
        minutes_in_a_day = 1440
        for i in range(0, minutes_in_a_day, step_size):
            h = int(i // 60)
            m = int(i % 60)

            key = f'{h}:{m:02d}'

            x[key] = []

    def get_id(self):
        return self._id

    def _time_to_string():
        h = int(x // 60)
        m = int(x % 60)
        return f'{h}:{m:02d}'

    def add_activity(self, start_time, event_id, event_type, row):
        device_id = row['device_id']

        st = device_info.DEVICE_DATE_RANGE[device_id][0]
        et = device_info.DEVICE_DATE_RANGE[device_id][1]

        if self._first_day > st:
            self._first_day = st

        if self._last_day < et:
            self._last_day = et

        start_time = datetime.datetime.utcfromtimestamp(start_time)
        start_time = start_time.astimezone(pytz.timezone('US/Mountain'))

        self._activity_count += 1

        if event_type != "battery":
            for key in self._data:
                self._data[key].append(row[key])

            day = start_time.strftime("%d/%m/%Y %z")
            self._days_with_activity.add(day)

            # Sun = 0
            weekday = (start_time.weekday() + 1) % 7
            sun = start_time - datetime.timedelta(weekday)
            sun = sun.strftime("%d/%m/%Y %z")
            self._weeks_with_activity.add(sun)

    def get_count(self):
        return self._activity_count

    def get_days_with_activity(self):
        return self._days_with_activity

    def get_weeks_with_activity(self):
        return self._weeks_with_activity

    def get_first_day(self):
        return self._first_day

    def get_last_day(self):
        return self._last_day

    def get_data(self):
        return pd.DataFrame.from_dict(self._data)
