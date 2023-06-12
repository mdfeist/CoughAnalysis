import time
import datetime


def convert_to_unix(date):
    return time.mktime(date.timetuple())


def convert_str_to_date(str_date):
    return datetime.datetime.strptime(str_date, '%d/%m/%Y %z')


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def daterange_hours(start_date, end_date):
    seconds_in_day = 24 * 60 * 60
    difference = (end_date - start_date)
    hours = divmod(difference.days * seconds_in_day +
                   difference.seconds, 3600)[0]
    for n in range(int(hours)):
        yield datetime.datetime.combine(start_date, datetime.time(hour=n))
