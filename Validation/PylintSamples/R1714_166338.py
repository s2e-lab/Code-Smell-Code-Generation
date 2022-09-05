def date_from_relative_week_year(base_date, time, dow, ordinal=1):
    """
    Converts relative day to time
    Eg. this tuesday, last tuesday
    """
    # If there is an ordinal (next 3 weeks) => return a start and end range
    # Reset date to start of the day
    relative_date = datetime(base_date.year, base_date.month, base_date.day)
    ord = convert_string_to_number(ordinal)
    if dow in year_variations:
        if time == 'this' or time == 'coming':
            return datetime(relative_date.year, 1, 1)
        elif time == 'last' or time == 'previous':
            return datetime(relative_date.year - 1, relative_date.month, 1)
        elif time == 'next' or time == 'following':
            return relative_date + timedelta(ord * 365)
        elif time == 'end of the':
            return datetime(relative_date.year, 12, 31)
    elif dow in month_variations:
        if time == 'this':
            return datetime(relative_date.year, relative_date.month, relative_date.day)
        elif time == 'last' or time == 'previous':
            return datetime(relative_date.year, relative_date.month - 1, relative_date.day)
        elif time == 'next' or time == 'following':
            if relative_date.month + ord >= 12:
                month = relative_date.month - 1 + ord
                year = relative_date.year + month // 12
                month = month % 12 + 1
                day = min(relative_date.day, calendar.monthrange(year, month)[1])
                return datetime(year, month, day)
            else:
                return datetime(relative_date.year, relative_date.month + ord, relative_date.day)
        elif time == 'end of the':
            return datetime(
                relative_date.year,
                relative_date.month,
                calendar.monthrange(relative_date.year, relative_date.month)[1]
            )
    elif dow in week_variations:
        if time == 'this':
            return relative_date - timedelta(days=relative_date.weekday())
        elif time == 'last' or time == 'previous':
            return relative_date - timedelta(weeks=1)
        elif time == 'next' or time == 'following':
            return relative_date + timedelta(weeks=ord)
        elif time == 'end of the':
            day_of_week = base_date.weekday()
            return day_of_week + timedelta(days=6 - relative_date.weekday())
    elif dow in day_variations:
        if time == 'this':
            return relative_date
        elif time == 'last' or time == 'previous':
            return relative_date - timedelta(days=1)
        elif time == 'next' or time == 'following':
            return relative_date + timedelta(days=ord)
        elif time == 'end of the':
            return datetime(relative_date.year, relative_date.month, relative_date.day, 23, 59, 59)