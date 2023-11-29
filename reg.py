from os import makedirs

import numpy as np
import scipy.stats
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
import sklearn.metrics
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime

from tqdm import tqdm

from subject import load_from_csv
import days
import date_utils
import utils
import graph

DATA_DIR = "data_oct_2022/"
CLINICAL_CSV = DATA_DIR + "clinical.csv"

CLEAN_DATA_DIR = "results/clean/"
DEVICE_CSV_PATH = CLEAN_DATA_DIR + "cleaned.csv"

RESULTS_DIR = "results/reg/"

makedirs(RESULTS_DIR, exist_ok=True)

LCQ_N_FEATURES = 3
KBILD_N_FEATURES = 3
CROSS_VAL_BATCH_SIZE = 1

y_labels_LCQ = ["LCQ_Physical", "LCQ_Psychological", "LCD_Social", "LCD_Total"]
y_labels_KBILD = ["KBILD_Breathlessness",
                  "KBILD_Psychological", "KBILD_Chest", "KBILD_Total"]


def calculate_LCQ_features(X, df, dates, subject_baseline):
    for index, row in df.iterrows():
        # date = datetime.strptime(row["Date"], '%Y-%m-%d')
        date = row["Date"].to_pydatetime()

        start_date = date_utils.get_date_with_offset(date, -7).date()
        end_date = date_utils.get_date_with_offset(date, 7).date()

        dates_with_range = date_utils.dates_with_range(
            dates, start_date, end_date)

        cc_baseline_mean = subject_baseline["cc_baseline_mean"]
        cc_baseline_std = subject_baseline["cc_baseline_std"]
        ca_baseline_mean = subject_baseline["ca_baseline_mean"]
        ca_baseline_std = subject_baseline["ca_baseline_std"]
        a_baseline_mean = subject_baseline["a_baseline_mean"]
        a_baseline_std = subject_baseline["a_baseline_std"]

        cc_hour_dist = []
        ca_hour_dist = []
        a_hour_dist = []

        num_hours = 0
        for day_info in dates_with_range:
            cc = day_info.coughCount()
            ca = day_info.coughActivity()
            a = day_info.activity()
            usage = day_info.estimated_usage()

            for hour in range(24):
                if usage[hour]:
                    cc_hour = cc[hour]
                    ca_hour = ca[hour]
                    a_hour = a[hour]

                    # Normalize
                    cc_hour = (cc_hour-cc_baseline_mean)
                    ca_hour = (ca_hour-ca_baseline_mean)
                    a_hour = (a_hour-a_baseline_mean)

                    cc_hour_dist.append(cc_hour)
                    ca_hour_dist.append(ca_hour)
                    a_hour_dist.append(a_hour)
                    num_hours += 1

        if num_hours < 2:
            # print(f"SKIP! - {index} - {num_hours} hours")
            # print(
            #     f"Num Dates: {len(dates)} in range {len(dates_with_range)}")
            continue

        # Calculate features
        cc_mean = np.mean(cc_hour_dist)
        cc_std = np.std(cc_hour_dist)

        ca_median = np.median(ca_hour_dist)
        ca_mean = np.mean(ca_hour_dist)
        ca_std = np.std(ca_hour_dist)

        a_mean = np.mean(a_hour_dist)
        a_std = np.std(a_hour_dist)

        # Outliers
        cc_lower_outliers, cc_upper_outliers = utils.number_of_outliers(
            cc_hour_dist,
            subject_baseline["cc_baseline_lower"],
            subject_baseline["cc_baseline_upper"])

        cc_lower_outliers /= num_hours
        cc_upper_outliers /= num_hours

        ca_lower_outliers, ca_upper_outliers = utils.number_of_outliers(
            ca_hour_dist,
            subject_baseline["ca_baseline_lower"],
            subject_baseline["ca_baseline_upper"])

        ca_lower_outliers /= num_hours
        ca_upper_outliers /= num_hours

        a_lower_outliers, a_upper_outliers = utils.number_of_outliers(
            a_hour_dist,
            subject_baseline["a_baseline_lower"],
            subject_baseline["a_baseline_upper"])

        a_lower_outliers /= num_hours
        a_upper_outliers /= num_hours

        # X_LCQ[index, 0] = c_a_std
        X[index, 0] = ca_std
        X[index, 1] = ca_median
        # X_LCQ[index, 2] = a_upper_outliers


def calculate_KBILD_features(X, df, dates, subject_baseline):
    for index, row in df.iterrows():
        # date = datetime.strptime(row["Date"], '%Y-%m-%d')
        date = row["Date"].to_pydatetime()

        start_date = date_utils.get_date_with_offset(date, -7).date()
        end_date = date_utils.get_date_with_offset(date, 7).date()

        dates_with_range = date_utils.dates_with_range(
            dates, start_date, end_date)

        cc_baseline_mean = subject_baseline["cc_baseline_mean"]
        cc_baseline_std = subject_baseline["cc_baseline_std"]
        ca_baseline_mean = subject_baseline["ca_baseline_mean"]
        ca_baseline_std = subject_baseline["ca_baseline_std"]
        a_baseline_mean = subject_baseline["a_baseline_mean"]
        a_baseline_std = subject_baseline["a_baseline_std"]

        cc_hour_dist = []
        ca_hour_dist = []
        a_hour_dist = []

        num_hours = 0
        for day_info in dates_with_range:
            cc = day_info.coughCount()
            ca = day_info.coughActivity()
            a = day_info.activity()
            usage = day_info.estimated_usage()

            for hour in range(24):
                if usage[hour]:
                    cc_hour = cc[hour]
                    ca_hour = ca[hour]
                    a_hour = a[hour]

                    # Normalize
                    # cc_hour = (cc_hour-cc_baseline_mean)
                    # ca_hour = (ca_hour-ca_baseline_mean)
                    # a_hour = (a_hour-a_baseline_mean)

                    cc_hour_dist.append(cc_hour)
                    ca_hour_dist.append(ca_hour)
                    a_hour_dist.append(a_hour)
                    num_hours += 1

        if num_hours < 2:
            # print(f"SKIP! - {index} - {num_hours} hours")
            # print(
            #     f"Num Dates: {len(dates)} in range {len(dates_with_range)}")
            continue

        # Calculate features
        cc_mean = np.mean(cc_hour_dist)
        cc_std = np.std(cc_hour_dist)

        ca_median = np.median(ca_hour_dist)
        ca_mean = np.mean(ca_hour_dist)
        ca_std = np.std(ca_hour_dist)

        a_mean = np.mean(a_hour_dist)
        a_std = np.std(a_hour_dist)

        # Outliers
        cc_lower_outliers, cc_upper_outliers = utils.number_of_outliers(
            cc_hour_dist,
            subject_baseline["cc_baseline_lower"],
            subject_baseline["cc_baseline_upper"])

        cc_lower_outliers /= num_hours
        cc_upper_outliers /= num_hours

        ca_lower_outliers, ca_upper_outliers = utils.number_of_outliers(
            ca_hour_dist,
            subject_baseline["ca_baseline_lower"],
            subject_baseline["ca_baseline_upper"])

        ca_lower_outliers /= num_hours
        ca_upper_outliers /= num_hours

        a_lower_outliers, a_upper_outliers = utils.number_of_outliers(
            a_hour_dist,
            subject_baseline["a_baseline_lower"],
            subject_baseline["a_baseline_upper"])

        a_lower_outliers /= num_hours
        a_upper_outliers /= num_hours

        X[index, 0] = ca_std
        # X[index, 1] = ca_std


def fit(X, y):
    # Cross Val
    y_pred = np.zeros_like(y)

    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    batches = list(range(X.shape[0]))
    batches = list(chunk(batches, CROSS_VAL_BATCH_SIZE))

    for batch in batches:
        test_X = X[batch]
        train_X = np.delete(X, batch, axis=0)
        train_y = np.delete(y, batch, axis=0)
        reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]
                      ).fit(train_X, train_y)
        y_pred[batch] = reg.predict(test_X)
    return y_pred


def main():
    print("Loading Data...")
    # Load CSV and split data based on subject ID
    subjects = load_from_csv(DEVICE_CSV_PATH)

    # clinical_df = None
    clinical_df = pd.read_csv(CLINICAL_CSV, parse_dates=['Date'])

    # Color Map for Labeling Patients
    unique_patients = clinical_df["Patient"].unique()
    patient_cmap = cm.get_cmap('Accent', len(unique_patients))

    def plot(x, y, y_pred, patients):
        p = x.argsort()
        x = x[p]
        y = y[p]
        y_pred = y_pred[p]

        for i in range(len(patients)):
            p = patients[i]
            plt.scatter(x[i],
                        y[i],
                        c=patient_cmap.colors[p],
                        label=unique_patients[p])

        plt.plot(x, y_pred, color="blue", linewidth=3)

    # Interpolate Data for Missing Months
    # clinical_df_subjects = []
    # for subject_id in tqdm(sorted(subjects.keys())):
    #     clinical_df_subject = clinical_df[clinical_df["Patient"] == subject_id]
    #     clinical_df_subject.set_index(clinical_df_subject.Date, inplace=True)
    #     clinical_df_subject = clinical_df_subject.resample('D').sum().fillna(0)
    #     clinical_df_subject.replace({
    #         "LCQ_Physical": 0,
    #         "LCQ_Psychological": 0,
    #         "LCD_Social": 0,
    #         "LCD_Total": 0,
    #         "KBILD_Breathlessness": 0,
    #         "KBILD_Psychological": 0,
    #         "KBILD_Chest": 0,
    #         "KBILD_Total": 0
    #     }, np.nan, inplace=True)
    #     clinical_df_subject.interpolate(inplace=True)
    #     clinical_df_subject['Date'] = clinical_df_subject.index
    #     clinical_df_subject.index = range(len(clinical_df_subject))
    #     clinical_df_subject["Patient"] = subject_id
    #     clinical_df_subjects.append(clinical_df_subject)

    # clinical_df = pd.concat(clinical_df_subjects)

    # LCQ
    clinical_df_for_y_LCQ = clinical_df.dropna(
        subset=["LCQ_Physical",
                "LCQ_Psychological",
                "LCD_Social",
                "LCD_Total",
                "Date",
                "Patient"])

    clinical_df_for_y_LCQ.reset_index(drop=True, inplace=True)

    y_LCQ_patient_labels = pd.Index(unique_patients).get_indexer(
        clinical_df_for_y_LCQ["Patient"])

    y_LCQ = clinical_df_for_y_LCQ[y_labels_LCQ].to_numpy()

    X_LCQ = np.zeros((y_LCQ.shape[0], LCQ_N_FEATURES))

    print(f'X_LCQ shape: {X_LCQ.shape}')
    print(f'y_LCQ shape: {y_LCQ.shape}')

    # KBILD
    clinical_df_for_y_KBILD = clinical_df.dropna(
        subset=["KBILD_Breathlessness",
                "KBILD_Psychological",
                "KBILD_Chest",
                "KBILD_Total",
                "Date",
                "Patient"])

    clinical_df_for_y_KBILD.reset_index(drop=True, inplace=True)

    y_KBILD_patient_labels = pd.Index(unique_patients).get_indexer(
        clinical_df_for_y_KBILD["Patient"])

    y_KBILD = clinical_df_for_y_KBILD[y_labels_KBILD].to_numpy()

    X_KBILD = np.zeros((y_KBILD.shape[0], KBILD_N_FEATURES))

    print(f'X_KBILD shape: {X_KBILD.shape}')
    print(f'y_KBILD shape: {y_KBILD.shape}')

    # Subject stats
    all_dates = {}
    subject_baselines = {}

    print("Caculating Stats for Subjects...")
    for subject_id in tqdm(sorted(subjects.keys())):
        # Get subject
        subject = subjects[subject_id]

        # Formatting dates for subject into days
        # Returns a list of DayInfo objects
        dates = days.create_days(subject)
        all_dates[subject_id] = dates

        if len(dates) <= 0:
            continue

        cc_hour_dist_all = []
        ca_hour_dist_all = []
        a_hour_dist_all = []

        for day_info in dates:
            cc = day_info.coughCount()
            ca = day_info.coughActivity()
            a = day_info.activity()
            usage = day_info.estimated_usage()

            for hour in range(24):
                if usage[hour]:
                    cc_hour_dist_all.append(cc[hour])
                    ca_hour_dist_all.append(ca[hour])
                    a_hour_dist_all.append(a[hour])

        cc_baseline_mean = np.mean(cc_hour_dist_all)
        cc_baseline_std = np.std(cc_hour_dist_all)

        ca_baseline_mean = np.mean(ca_hour_dist_all)
        ca_baseline_std = np.std(ca_hour_dist_all)

        a_baseline_mean = np.mean(a_hour_dist_all)
        a_baseline_std = np.std(a_hour_dist_all)

        _, cc_baseline_lower, cc_baseline_upper = utils.calculateDistributionInfo(
            cc_hour_dist_all, 1.0)

        _, ca_baseline_lower, ca_baseline_upper = utils.calculateDistributionInfo(
            ca_hour_dist_all, 1.0)

        _, a_baseline_lower, a_baseline_upper = utils.calculateDistributionInfo(
            a_hour_dist_all, 1.0)

        subject_baselines[subject_id] = {
            'cc_baseline_mean': cc_baseline_mean,
            'cc_baseline_std': cc_baseline_std,
            'ca_baseline_mean': ca_baseline_mean,
            'ca_baseline_std': ca_baseline_std,
            'a_baseline_mean': a_baseline_mean,
            'a_baseline_std': a_baseline_std,
            'cc_baseline_lower': cc_baseline_lower,
            'cc_baseline_upper': cc_baseline_upper,
            'ca_baseline_lower': ca_baseline_lower,
            'ca_baseline_upper': ca_baseline_upper,
            'a_baseline_lower': a_baseline_lower,
            'a_baseline_upper': a_baseline_upper,
        }

    # Calculate Features
    for subject_id in tqdm(sorted(subjects.keys())):
        dates = all_dates[subject_id]

        if len(dates) <= 0:
            continue

        subject_clinical_LCQ = clinical_df_for_y_LCQ[
            clinical_df_for_y_LCQ["Patient"] == subject_id
        ]
        calculate_LCQ_features(X_LCQ, subject_clinical_LCQ,
                               dates, subject_baselines[subject_id])

        subject_clinical_KBILD = clinical_df_for_y_KBILD[
            clinical_df_for_y_KBILD["Patient"] == subject_id
        ]
        calculate_KBILD_features(X_KBILD, subject_clinical_KBILD,
                                 dates, subject_baselines[subject_id])

    y_pred_LCQ = fit(X_LCQ, y_LCQ)
    y_pred_KBILD = fit(X_KBILD, y_KBILD)

    print("\nLCQ: ")
    print("Explained Variance Score: ")
    print(sklearn.metrics.explained_variance_score(
        y_LCQ, y_pred_LCQ, multioutput="raw_values"))
    print("Mean Squared Error: ")
    print(sklearn.metrics.mean_squared_error(
        y_LCQ, y_pred_LCQ, multioutput="raw_values"))
    print("Mean Absolute Error: ")
    print(sklearn.metrics.mean_absolute_error(
        y_LCQ, y_pred_LCQ, multioutput="raw_values"))

    print("\nKBILD: ")
    print("Explained Variance Score: ")
    print(sklearn.metrics.explained_variance_score(
        y_KBILD, y_pred_KBILD, multioutput="raw_values"))
    print("Mean Squared Error: ")
    print(sklearn.metrics.mean_squared_error(
        y_KBILD, y_pred_KBILD, multioutput="raw_values"))
    print("Mean Absolute Error: ")
    print(sklearn.metrics.mean_absolute_error(
        y_KBILD, y_pred_KBILD, multioutput="raw_values"))

    # Plot outputs
    plt.figure()
    plot(y_pred_LCQ[:, 3], y_LCQ[:, 3], y_pred_LCQ[:, 3], y_LCQ_patient_labels)
    plt.xlabel("Y Pred")
    plt.ylabel("Y True")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="lower right")
    plt.tight_layout()

    plt.figure()
    plot(y_pred_KBILD[:, 3], y_KBILD[:, 3],
         y_pred_KBILD[:, 3], y_KBILD_patient_labels)
    plt.xlabel("Y Pred")
    plt.ylabel("Y True")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="lower right")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
