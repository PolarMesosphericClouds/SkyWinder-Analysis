def to_years(num_seconds):
    return to_days(num_seconds) / 365.25


def from_years(num_years):
    return from_days(num_years * 365.25)


def to_days(num_seconds):
    return to_hours(num_seconds) / 24.0


def from_days(num_days):
    return from_hours(num_days * 24.0)


def to_julday(num_seconds):
    return (num_seconds / 86400.0 + 2440587.5)


def from_julday(jd):
    return ((jd - 2440587.5) * 86400.0)


def to_hours(num_seconds):
    return to_minutes(num_seconds) / 60.0


def from_hours(num_hours):
    return from_minutes(num_hours * 60.0)


def to_minutes(num_seconds):
    return num_seconds / 60.0


def from_minutes(num_minutes):
    return num_minutes * 60.0


def to_seconds(num_seconds):
    return num_seconds


def from_seconds(num_seconds):
    return num_seconds


def to_milliseconds(num_seconds):
    return num_seconds * 1000.0


def from_milliseconds(num_milliseconds):
    return num_milliseconds / 1000.0


def time_string(num_seconds):
    if num_seconds > from_years(999.0):
        return "%.0e yr" % to_years(num_seconds)
    elif num_seconds > from_years(1.0):
        return "%.1f yr" % to_years(num_seconds)
    elif num_seconds > from_days(1.0):
        return "%.1f dy" % to_days(num_seconds)
    elif num_seconds > from_hours(1.0):
        return "%.1f hr" % to_hours(num_seconds)
    elif num_seconds > from_minutes(1.0):
        return "%.1f min" % to_minutes(num_seconds)
    elif num_seconds > from_seconds(10.0):
        return "%.1f s" % to_seconds(num_seconds)
    elif num_seconds > from_seconds(1.0):
        return "%.2f s" % to_seconds(num_seconds)
    else:
        return "%.0f ms" % to_milliseconds(num_seconds)
