import datetime
begin = '2018-02-15'
end = '2019-04-23'

dt_start = datetime.datetime.strptime(begin, '%Y-%m-%d')
dt_end = datetime.datetime.strptime(end, '%Y-%m-%d')

one_day = datetime.timedelta(1)
start_dates = [dt_start]
end_dates = []
today = dt_start
months_in_group = 0
month_group_size = 3
while today <= dt_end:
    # print(today)
    tomorrow = today + one_day
    if tomorrow.month != today.month:
        months_in_group += 1

    if months_in_group == month_group_size:
        start_dates.append(tomorrow)
        end_dates.append(today)

        months_in_group = 0
    today = tomorrow

end_dates.append(dt_end)

month_3_chunks = zip(start_dates, end_dates)


month_chunk_info = [
    {
        "start": start,
        "end": end
    } for start, end in month_3_chunks
]

out_fmt = '%d %B %Y'
for chunk in month_chunk_info:
    start = chunk["start"]
    end = chunk["end"]
    print('{} to {}'.format(start.strftime(out_fmt), end.strftime(out_fmt)))
