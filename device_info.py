import datetime

EVENT_TYPES = [
    'activity',
    'asthma',
    'battery',
    'cough',
    'cough activity',
    'heartrate',
    'respiration',
    'temperature',
    'voice journal'
]

BATTERY_STATUS = [
    'battery',
    'charging'
]

SUBJECTS_ID = {
    'A020219902410000241': 'EDM-002',
    'A020219902410000181': 'EDM-001',
    'A020219902410000291': 'EDM-005',
    'A020219902410000151': 'EDM-008',
    'A020219902410000321': 'EDM-004',
    'A020219902410000141': 'EDM-002',
    'A020219902410000231': 'EDM-001',
    'A020219902410000121': 'EDM-003',
    'A020219902410000161': 'EDM-006',
    'A020219902410000251': 'EDM-007',
    'A020219902410000151': 'EDM-008',
}

REMOVED_DEVICES_AND_SUBJECTS_ID = {
    'A020219902410000251': 'EDM-007'
}

DEVICE_DATE_RANGE = {
    'A020219902410000161': (
        datetime.date(2021, 12, 1),
        datetime.date(2022, 5, 10)
    ),
    'A020219902410000241': (
        datetime.date(2021, 7, 26),
        datetime.date(2021, 8, 6)
    ),
    'A020219902410000181': (
        datetime.date(2021, 10, 13),
        datetime.date(2022, 1, 1)
    ),
    'A020219902410000291': (
        datetime.date(2021, 11, 1),
        datetime.date(2022, 5, 11)
    ),
    'A020219902410000151': (
        datetime.date(2022, 8, 25),
        datetime.date(2023, 3, 1)
    ),
    'A020219902410000251': (
        datetime.date(2021, 4, 20),
        datetime.date(2021, 10, 24)
    ),
    'A020219902410000321': (
        datetime.date(2021, 10, 4),
        datetime.date(2022, 4, 4)
    ),
    'A020219902410000141': (
        datetime.date(2021, 8, 6),
        datetime.date(2021, 10, 25)
    ),
    'A020219902410000231': (
        datetime.date(2021, 7, 9),
        datetime.date(2021, 10, 13)
    ),
    'A020219902410000121': (
        datetime.date(2021, 10, 8),
        datetime.date(2022, 4, 8)
    ),
}
