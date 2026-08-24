import datetime as dt

from beer_sentiment.io.csv_io import compute_window


def test_morning_window():
    start, end = compute_window("morning", dt.date(2026, 8, 20), {})
    assert start == dt.datetime(2026, 8, 19, 17, 30)
    assert end == dt.datetime(2026, 8, 20, 10, 0)


def test_monday_morning_window():
    start, end = compute_window("morning", dt.date(2026, 8, 24), {})
    assert start == dt.datetime(2026, 8, 21, 17, 30)
    assert end == dt.datetime(2026, 8, 24, 10, 0)


def test_afternoon_window():
    start, end = compute_window("afternoon", dt.date(2026, 8, 20), {})
    assert start == dt.datetime(2026, 8, 20, 10, 0)
    assert end == dt.datetime(2026, 8, 20, 17, 30)
