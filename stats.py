from os import makedirs

import pandas as pd
import numpy as np

from tqdm import tqdm

from subject import Subject, SUBJECT_ACTIVE_TIME

import utils
import time_utils


DATA_DIR = "data_oct_2022/"
CSV_PATH = DATA_DIR + "cleaned.csv"
RESULTS_DIR = "results/stats/"

makedirs(RESULTS_DIR, exist_ok=True)

FIG_SIZE = (40, 20)


def calculate_correlation_between_cough_and_activity(subject):
    # Get subject data
    subject_df = subject.get_data()

    # Time of Day
    step_size = 450  # 450
    window_size = 900  # 900
    number_of_steps = 86400 // step_size

    min_time = SUBJECT_ACTIVE_TIME[subject.get_id()][0] * 3600
    max_time = SUBJECT_ACTIVE_TIME[subject.get_id()][1] * 3600

    dates = {}

    for day_start in time_utils.daterange(subject.get_first_day(), subject.get_last_day()):
        day_start_time = time_utils.convert_to_unix(day_start)

        day_df = subject_df.loc[(subject_df['corrected_start_time'] >= (day_start_time + min_time)) & (
            subject_df['corrected_start_time'] < (day_start_time + max_time))]
        cough_df = day_df.loc[(day_df['event type'] == 'cough')]
        cough_activity_df = day_df.loc[(
            day_df['event type'] == 'cough activity')]
        activity_df = day_df.loc[(day_df['event type'] == 'activity')]

        cough_count_total = cough_df['cough count'].sum()

        if np.isnan(cough_count_total):
            cough_count_total = 0.0

        valid_events_df = utils.get_valid_events(day_df)
        valid_enents_count = valid_events_df.shape[0]

        # Calculate Time of Day
        if cough_count_total > 0 and valid_enents_count > 5:
            activity_total = activity_df['activity'].sum()

            date = day_start.month
            if date not in dates:
                dates[date] = {
                    'date': day_start.strftime('%m/%Y'),
                    'event_count': 0,
                    'min_event_count': valid_enents_count,
                    'max_event_count': 0,
                    'cough_count': 0,
                    'min_cough_count': cough_count_total,
                    'max_cough_count': 0,
                    'activity': 0,
                    'min_activity': activity_total,
                    'max_activity': 0,
                    'recorded_days': 0,
                    'window_data': ([], [], []),
                }

            dates[date]['recorded_days'] += 1

            dates[date]['event_count'] += valid_enents_count
            dates[date]['min_event_count'] = np.minimum(
                dates[date]['min_event_count'], valid_enents_count)
            dates[date]['max_event_count'] = np.maximum(
                dates[date]['max_event_count'], valid_enents_count)

            dates[date]['cough_count'] += cough_count_total

            dates[date]['min_cough_count'] = np.minimum(
                dates[date]['min_cough_count'], cough_count_total)
            dates[date]['max_cough_count'] = np.maximum(
                dates[date]['max_cough_count'], cough_count_total)

            dates[date]['activity'] += activity_total

            dates[date]['min_activity'] = np.minimum(
                dates[date]['min_activity'], activity_total)
            dates[date]['max_activity'] = np.maximum(
                dates[date]['max_activity'], activity_total)

            for i in range(number_of_steps):
                time = i * step_size

                if time < min_time:
                    continue
                if time > max_time:
                    break

                start_time = day_start_time + time
                end_time = start_time + window_size

                time_of_day_cough_df = cough_df.loc[(cough_df['corrected_start_time'] >= start_time) & (
                    cough_df['corrected_start_time'] < end_time)]
                time_of_day_cough_activity_df = cough_activity_df.loc[(cough_activity_df['corrected_start_time'] >= start_time) & (
                    cough_activity_df['corrected_start_time'] < end_time)]
                time_of_day_activity_df = activity_df.loc[(activity_df['corrected_start_time'] >= start_time) & (
                    activity_df['corrected_start_time'] < end_time)]

                cough_count = time_of_day_cough_df['cough count'].sum()
                cough_activity = utils.get_mean(
                    time_of_day_cough_activity_df['cough activity'])
                activity = time_of_day_activity_df['activity'].sum()

                dates[date]['window_data'][0].append(cough_count)
                dates[date]['window_data'][1].append(cough_activity)
                dates[date]['window_data'][2].append(activity)

    print(f'{subject.get_id()}')

    data_df = {
        'date': [],
        'days recorded count': [],
        'total event count': [],
        'min event count': [],
        'max event count': [],
        'total cough count': [],
        'min cough count': [],
        'max cough count': [],
        'total activity': [],
        'min activity': [],
        'max activity': [],
        '15m cough count median': [],
        '15m cough count mean': [],
        '15m cough count std': [],
        '15m cough activity median': [],
        '15m cough activity mean': [],
        '15m cough activity std': [],
        '15m activity median': [],
        '15m activity mean': [],
        '15m activity std': [],
        '15m correlation cough count vs cough activity': [],
        '15m correlation cough count vs activity': [],
        '15m correlation cough activity vs activity': [],
    }

    for date in dates:
        # Get date
        d = dates[date]

        # Get correlation
        wdata = d['window_data']
        cor_array = np.vstack((wdata[0], wdata[1], wdata[2]))
        cor = np.corrcoef(cor_array)

        recorded_days = d['recorded_days']

        event_count = d['event_count']
        min_event_count = d['min_event_count']
        max_event_count = d['max_event_count']

        cough_count = d['cough_count']
        min_cough_count = d['min_cough_count']
        max_cough_count = d['max_cough_count']

        activity = d['activity']
        min_activity = d['min_activity']
        max_activity = d['max_activity']

        data_df['date'].append(d['date'])

        data_df['days recorded count'].append(recorded_days)

        data_df['total event count'].append(event_count)
        data_df['min event count'].append(min_event_count)
        data_df['max event count'].append(max_event_count)

        data_df['total cough count'].append(cough_count)
        data_df['min cough count'].append(min_cough_count)
        data_df['max cough count'].append(max_cough_count)

        data_df['total activity'].append(activity)
        data_df['min activity'].append(min_activity)
        data_df['max activity'].append(max_activity)

        data_df['15m cough count median'].append(np.median(wdata[0]))
        data_df['15m cough count mean'].append(np.mean(wdata[0]))
        data_df['15m cough count std'].append(np.std(wdata[0]))

        data_df['15m cough activity median'].append(np.median(wdata[1]))
        data_df['15m cough activity mean'].append(np.mean(wdata[1]))
        data_df['15m cough activity std'].append(np.std(wdata[1]))

        data_df['15m activity median'].append(np.median(wdata[2]))
        data_df['15m activity mean'].append(np.mean(wdata[2]))
        data_df['15m activity std'].append(np.std(wdata[2]))

        data_df['15m correlation cough count vs cough activity'].append(
            cor[0, 1])
        data_df['15m correlation cough count vs activity'].append(cor[0, 2])
        data_df['15m correlation cough activity vs activity'].append(cor[1, 2])

    df = pd.DataFrame(data_df)
    df.to_csv(f'{RESULTS_DIR}{subject.get_id()}_stats.csv')


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

        # Calculating stats for coughing activity
        calculate_correlation_between_cough_and_activity(subject)


if __name__ == "__main__":
    main()
