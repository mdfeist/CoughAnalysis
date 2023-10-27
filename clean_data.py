from os import makedirs

import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

import time_utils
import device_info

DATA_DIR = 'data_oct_2022/'
CSV_PATH = DATA_DIR + 'rsm021_combined.csv'
RESULTS_DIR = "results/clean/"

makedirs(RESULTS_DIR, exist_ok=True)

MINUTE = 60
HOUR = 3600
DAY = 86400
WEEK = 604800
MONTH = 2629743
YEAR = 31556926


class Chunk:
    def __init__(self, st, et, ids, times, valid):
        self.st = st
        self.et = et
        self.ids = ids
        self.times = times
        self.valid = valid


def get_chunks(device):
    chunks = []

    ids = []
    times = []
    st = device['times'][0]
    et = device['times'][0]

    valid_st = device['st'] - WEEK
    valid_et = device['et'] + WEEK

    for i in range(device['times'].size):
        t0 = device['times'][i]

        ids.append(device['events'][i])
        times.append(t0)
        et = t0

        if i >= device['times'].size - 1:
            valid = st <= valid_et and et >= valid_st
            chunks.append(Chunk(st, et, ids, times, valid))
            break

        # New chunck
        t1 = device['times'][i+1]
        diff = t1 - t0
        # New Chunk if
        # time goes backwards by more than a day
        # or there is a two week diff between events
        if t0 > (t1 + 6*HOUR) or diff > 2*WEEK:
            valid = st <= valid_et and et >= valid_st
            chunks.append(Chunk(st, et, ids, times, valid))

            ids = []
            times = []
            st = t1
            et = t1

    return chunks


def correct_times(device_id, device):
    num_chunks = device['times'].size
    while True:
        chunks = get_chunks(device)
        new_num_chunks = len(chunks)

        print(f"{device_id} has {new_num_chunks} chunks")

        if new_num_chunks <= 1:
            return

        num_chunks = new_num_chunks

        # First valid chunks
        first_valid = -1

        for i in range(num_chunks):
            if chunks[i].valid:
                first_valid = i
                break

        print(f"first valid {first_valid}")

        chunks_updated = False

        # Is there a chunk before the first valid chunk
        if first_valid > 0:
            invalid = first_valid - 1

            # Is the chuck greater than a month and possibly has the date set incorrectly
            if (chunks[invalid].et + MONTH) < chunks[first_valid].st:
                # Calculate time shift
                diff = np.floor(
                    (chunks[first_valid].st - chunks[invalid].et) / DAY)
                diff = DAY*diff

                # Shift chunk
                chunks[invalid].st += diff
                chunks[invalid].et += diff
                for j in range(len(chunks[invalid].times)):
                    chunks[invalid].times[j] += diff

                chunks_updated = True

        # Get first invalid after first valid
        first_invalid = -1
        first_valid = first_valid if first_valid >= 0 else 0

        for i in range(first_valid+1, num_chunks):
            if not chunks[i].valid:
                first_invalid = i
                break

        # Is there a chunk after first valid chunk that needs to be shifted
        if first_invalid < num_chunks:
            # Is date set incorrectly
            if (chunks[first_invalid].st + DAY) < chunks[first_valid].et:
                # Calculate time shift
                diff = np.ceil(
                    (chunks[first_valid].et - chunks[first_invalid].st) / DAY)
                if diff > 1:
                    diff = DAY*(diff + 1)

                    print(
                        f"{(chunks[first_valid].et/DAY)} to {((chunks[first_invalid].st + diff)/DAY)}")

                    # Shift chunk
                    chunks[first_invalid].st += diff
                    chunks[first_invalid].et += diff
                    for j in range(len(chunks[first_invalid].times)):
                        chunks[first_invalid].times[j] += diff
                    chunks[first_invalid].valid = True

                    chunks_updated = True

        # Last valid chunks
        last_valid = -1

        for i in reversed(range(num_chunks)):
            if chunks[i].valid:
                last_valid = i
                break

        # Is there a chunk after the last valid chunk
        if last_valid > 0 and last_valid < (num_chunks - 1):
            invalid = last_valid + 1

            if (chunks[invalid].st + DAY) < chunks[last_valid].et:
                # Calculate time shift
                diff = np.ceil(
                    (chunks[last_valid].et - chunks[invalid].st) / DAY)
                if diff > 1:
                    diff = DAY*(diff + 1)

                    print(
                        f"{(chunks[last_valid].et/DAY)} to {((chunks[invalid].st + diff)/DAY)}")

                    # Shift chunk
                    chunks[invalid].st += diff
                    chunks[invalid].et += diff
                    for j in range(len(chunks[invalid].times)):
                        chunks[invalid].times[j] += diff

                    chunks_updated = True

        if not chunks_updated:
            return

        # Flatten chunks
        ii = 0
        for c in range(num_chunks):
            chunk = chunks[c]
            for i in range(len(chunk.times)):
                device['times'][ii] = chunk.times[i]
                ii += 1


def remove_invalid_times(device):
    # Remove invalid times
    valid_st = device['st']
    valid_et = device['et']

    # Count number of invalids
    st_valid = np.zeros_like(device['times'], dtype=np.dtype(bool))
    et_valid = np.zeros_like(device['times'], dtype=np.dtype(bool))

    st_valid[device['times'] >= valid_st] = True
    et_valid[device['times'] <= valid_et] = True

    valid = st_valid & et_valid

    new_size = sum(valid)

    new_events = np.zeros(new_size)
    new_times = np.zeros(new_size)
    new_original_times = np.zeros(new_size)

    ii = 0
    for i in range(device['times'].size):
        if valid[i]:
            new_events[ii] = device['events'][i]
            new_times[ii] = device['times'][i]
            new_original_times[ii] = device['original_times'][i]
            ii += 1

    device['events'] = new_events
    device['times'] = new_times
    device['original_times'] = new_original_times

    st = device['st']
    et = device['et']

    device['target'] = np.linspace(st, et, num=new_size, dtype=np.dtype(int))


def shift_times(device):
    for i in range(device['times'].size - 1):
        t0 = device['times'][i]
        t1 = device['times'][i+1]

        if (t1 + DAY) < t0:
            print("SHIFT")
            diff = np.floor((t0 - t1)/DAY)
            device['times'][i+1:] += DAY*(diff + 1)


def check_original_times(device):
    for i in range(device['times'].size):
        tc = device['times'][i]
        to = device['original_times'][i]
        tt = device['target'][i]
        et = device['et']

        diff = tc - to
        if abs(diff) < WEEK:
            device['times'][i:] -= DAY*np.floor(diff/DAY)

        if abs(tt - tc) > abs(tt - to) or tc > et:
            device['times'][i:] -= DAY*np.floor(diff/DAY)


def main():
    print('Loading Data...')
    df = pd.read_csv(CSV_PATH)

    # Get Data for each device
    devices = {}
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        device_id = row['device_id']

        if device_id not in device_info.DEVICE_DATE_RANGE:
            continue

        if device_id not in devices:
            devices[device_id] = {
                'events': [],
                'times': []
            }

        start_time = row['start_time']
        event_id = row['id']

        devices[device_id]['events'].append(event_id)

        if device_id == 'A020219902410000231':
            devices[device_id]['times'].append(start_time - 11*HOUR)
        else:
            devices[device_id]['times'].append(start_time)

    # Check valid times
    for device_id, device in sorted(devices.items()):
        device['events'] = np.array(device['events'], dtype=np.dtype(int))
        device['times'] = np.array(device['times'], dtype=np.dtype(int))
        device['original_times'] = np.copy(device['times'])

        st = time_utils.convert_to_unix(
            device_info.DEVICE_DATE_RANGE[device_id][0])
        et = time_utils.convert_to_unix(
            device_info.DEVICE_DATE_RANGE[device_id][1])

        device['target'] = np.linspace(st, et, num=len(
            device['times']), dtype=np.dtype(int))
        device['st'] = st
        device['et'] = et

    # Correct times
    for device_id, device in sorted(devices.items()):
        correct_times(device_id, device)
        remove_invalid_times(device)
        shift_times(device)
        # check_original_times(device)

    # Save clean csv
    dataframes = []
    for device_id, device in tqdm(sorted(devices.items())):
        if device_id not in device_info.SUBJECTS_ID:
            continue

        subject_id = device_info.SUBJECTS_ID[device_id]

        device_data = df.loc[(df['device_id'] == device_id)
                             & (df['id'].isin(device['events']))]
        device_data = device_data.drop(['emailid', 'firstname', 'lastname', 'isblocked',
                                       'isdeleted', 'isadmin', 'dob', 'gender', 'UTC time', 'local time'], axis=1)
        device_data['corrected_start_time'] = device['times']
        device_data['subject_id'] = subject_id

        dataframes.append(device_data)

    df_clean = pd.concat(dataframes)
    df_clean.to_csv(f'{RESULTS_DIR}cleaned.csv', index=False)

    # Graph
    for device_id, device in sorted(devices.items()):
        diff_form_original = abs(device['times'] - device['original_times'])
        was_shifted = np.zeros_like(device['times'], dtype=np.dtype(bool))
        was_shifted[diff_form_original > DAY] = True
        percent_shifted = 0.0
        if device['times'].size > 0:
            percent_shifted = 100. * sum(was_shifted) / device['times'].size
        print(f"Device {device_id} had {percent_shifted}% shifted.")
        # print(f"Device {device_id} had {device['invalid_time']} invalid times.")

        fig, ax = plt.subplots()
        ax.plot(device['events'], device['original_times'], color='red')
        ax.plot(device['events'], device['target'], color='gray')
        ax.set_title(f'{device_id}')
        plt.savefig(f'{RESULTS_DIR}{device_id}_time.jpg')
        plt.close()

        fig, ax = plt.subplots()
        ax.set_ylim([device['st'] - 2*MONTH, device['et'] + 2*MONTH])
        ax.plot(device['events'], device['original_times'], color='red')
        ax.plot(device['events'], device['times'], color='blue')
        ax.plot(device['events'], device['target'], color='gray')
        ax.set_title(f'{device_id}')
        plt.savefig(f'{RESULTS_DIR}{device_id}_corrected_time.jpg')
        plt.close()


if __name__ == '__main__':
    main()
