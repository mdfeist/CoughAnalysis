import numpy as np


def get_mean(x):
    x = x.to_numpy()
    if x.size > 0:
        return np.mean(x)
    return 0.0


def get_valid_events(df):
    return df.loc[
        (df['event type'] != 'cough') &
        (df['event type'] != 'cough activity') &
        (df['event type'] != 'asthma') &
        (df['event type'] != 'respiration') &
        (df['event type'] != 'voice journal') &
        (
            (df['event type'] != 'heartrate') |
            (
                (df['event type'] == 'heartrate') & (
                    df['heart rate'] > 50)
            )
        ) &
        (
            (df['event type'] != 'temperature') |
            (
                (df['event type'] == 'temperature') & (
                    df['skin temperature'] >= 90)
            )
        )
    ]


def calculateMax(x):
    if type(x) == list:
        x = np.array(x)

    size = x.shape[0]
    n = size // 2

    x = np.sort(x)

    Q1 = np.median(x[:n])
    Q3 = np.median(x[n:])

    IQR = Q3 - Q1

    upper = Q3 + 1.5*IQR

    # Remove outliers
    x = x[x <= upper]

    return np.max(x)


def calculateDistributionInfo(x):
    if type(x) == list:
        x = np.array(x)

    size = x.shape[0]
    n = size // 2

    x = np.sort(x)

    median = np.median(x)

    Q1 = np.median(x[:n])
    Q3 = np.median(x[n:])

    IQR = Q3 - Q1

    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    return median, lower, upper


def mean_remove_outliers(x):
    if type(x) == list:
        x = np.array(x)

    size = x.shape[0]

    if size < 10:
        return np.mean(x)

    _, lower, upper = calculateDistributionInfo(x)
    x = x[(x > lower) & (x < upper)]
    return np.mean(x)
