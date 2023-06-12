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
