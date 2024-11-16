"""Utils specific for EPIC data."""
import datetime


def timestamp_to_seconds(timestamp: str):
    # Parse the timestamp string into a datetime object
    time_obj = datetime.datetime.strptime(timestamp, '%H:%M:%S.%f')

    # Calculate the total number of seconds using the timedelta object
    total_seconds = time_obj.time().second \
        + time_obj.time().minute * 60 \
        + time_obj.time().hour * 3600 \
        + time_obj.time().microsecond / 1000000
    
    return total_seconds
