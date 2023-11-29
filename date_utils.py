import datetime


def chunk_dates(dates, start, end, days_per_group=14):
    # Calculate 14 - day chuncks
    one_day = datetime.timedelta(1)
    time_start_dates = [start]
    time_end_dates = []
    today = start
    days_in_group = 0
    while today <= end:
        tomorrow = today + one_day
        days_in_group += 1

        if days_in_group >= days_per_group:
            time_start_dates.append(tomorrow)
            time_end_dates.append(today)
            days_in_group = 0
        today = tomorrow

    time_end_dates.append(end)
    time_chunk_dates = list(zip(time_start_dates, time_end_dates))

    time_chunks = [
        {
            "start": start,
            "end": end,
            "dates": []
        } for start, end in time_chunk_dates
    ]

    for dayInfo in dates:
        # Find chunk for day
        current_chunk = None
        for chunk in time_chunks:
            start = chunk["start"]
            end = chunk["end"]

            if dayInfo.date() >= start and dayInfo.date() <= end:
                current_chunk = chunk
                break

        if current_chunk is None:
            continue

        current_chunk["dates"].append(dayInfo)

    return time_chunks


def get_date_with_offset(date, offset):
    return date + datetime.timedelta(offset)


def dates_with_range(dates, start, end):
    in_range = []

    for day_info in dates:
        date = day_info.date()
        if date >= start and date <= end:
            in_range.append(day_info)

    return in_range
