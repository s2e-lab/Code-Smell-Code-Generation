def compute_stop_time_series_base(
    stop_times_subset: DataFrame,
    trip_subset: DataFrame,
    freq: str = "5Min",
    date_label: str = "20010101",
    *,
    split_directions: bool = False,
) -> DataFrame:
    """
    Given a subset of a stop times DataFrame and a subset of a trips
    DataFrame, return a DataFrame that provides a summary time series
    about the stops in the inner join of the two DataFrames.

    Parameters
    ----------
    stop_times_subset : DataFrame
        A valid GTFS stop times table
    trip_subset : DataFrame
        A valid GTFS trips table
    split_directions : boolean
        If ``True``, then separate each stop's stats by trip direction;
        otherwise aggregate trips visiting from both directions
    freq : Pandas frequency string
        Specifices the frequency with which to resample the time series;
        max frequency is one minute ('Min')
    date_label : string
        YYYYMMDD date string used as the date in the time series index

    Returns
    -------
    DataFrame
        A time series with a timestamp index for a 24-hour period
        sampled at the given frequency.
        The only indicator variable for each stop is

        - ``num_trips``: the number of trips that visit the stop and
          have a nonnull departure time from the stop

        The maximum allowable frequency is 1 minute.

        The columns are hierarchical (multi-indexed) with

        - top level: name = 'indicator', values = ['num_trips']
        - middle level: name = 'stop_id', values = the active stop IDs
        - bottom level: name = 'direction_id', values = 0s and 1s

        If not ``split_directions``, then don't include the bottom level.

    Notes
    -----
    - The time series is computed at a one-minute frequency, then
      resampled at the end to the given frequency
    - Stop times with null departure times are ignored, so the aggregate
      of ``num_trips`` across the day could be less than the
      ``num_trips`` column in :func:`compute_stop_stats_base`
    - All trip departure times are taken modulo 24 hours,
      so routes with trips that end past 23:59:59 will have all
      their stats wrap around to the early morning of the time series.
    - 'num_trips' should be resampled with ``how=np.sum``
    - If ``trip_subset`` is empty, then return an empty DataFrame
    - Raise a ValueError if ``split_directions`` and no non-NaN
      direction ID values present

    """
    if trip_subset.empty:
        return pd.DataFrame()

    f = pd.merge(stop_times_subset, trip_subset)

    if split_directions:
        if "direction_id" not in f.columns:
            f["direction_id"] = np.nan
        f = f.loc[lambda x: x.direction_id.notnull()].assign(
            direction_id=lambda x: x.direction_id.astype(int)
        )
        if f.empty:
            raise ValueError(
                "At least one trip direction ID value " "must be non-NaN."
            )

        # Alter stop IDs to encode trip direction:
        # <stop ID>-0 and <stop ID>-1
        f["stop_id"] = f["stop_id"] + "-" + f["direction_id"].map(str)
    stops = f["stop_id"].unique()

    # Bin each stop departure time
    bins = [i for i in range(24 * 60)]  # One bin for each minute
    num_bins = len(bins)

    def F(x):
        return (hp.timestr_to_seconds(x) // 60) % (24 * 60)

    f["departure_index"] = f["departure_time"].map(F)

    # Create one time series for each stop
    series_by_stop = {stop: [0 for i in range(num_bins)] for stop in stops}

    for stop, group in f.groupby("stop_id"):
        counts = Counter((bin, 0) for bin in bins) + Counter(
            group["departure_index"].values
        )
        series_by_stop[stop] = [counts[bin] for bin in bins]

    # Combine lists into dictionary of form indicator -> time series.
    # Only one indicator in this case, but could add more
    # in the future as was done with route time series.
    rng = pd.date_range(date_label, periods=24 * 60, freq="Min")
    series_by_indicator = {
        "num_trips": pd.DataFrame(series_by_stop, index=rng).fillna(0)
    }

    # Combine all time series into one time series
    g = hp.combine_time_series(
        series_by_indicator, kind="stop", split_directions=split_directions
    )
    return hp.downsample(g, freq=freq)