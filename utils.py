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


def calculateMax(list):
    size = len(list)
    n = size // 2
    x = np.array(list)

    x = np.sort(x)

    Q1 = np.median(x[:n])
    Q3 = np.median(x[n:])

    IQR = Q3 - Q1

    upper = Q3 + 1.5*IQR

    # Remove outliers
    x = x[x <= upper]

    return np.max(x)


def calculateDistributionInfo(x):
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
