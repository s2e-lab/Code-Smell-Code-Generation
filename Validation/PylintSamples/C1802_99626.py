def send_trending_data(events):
    """creates data point payloads for trending data to influxdb
    """
    bodies = {}

    # sort the values
    top_hits = sorted(
        [(key, count) for key, count in events.items()],
        key=lambda x: x[1],
        reverse=True
    )[:100]

    # build up points to be written
    for (site, content_id), count in top_hits:
        if not len(site) or not re.match(CONTENT_ID_REGEX, content_id):
            continue

        # add point
        bodies.setdefault(site, [])
        bodies[site].append([content_id, count])

    for site, points in bodies.items():
        # create name
        name = "{}_trending".format(site)
        # send payload to influxdb
        try:
            data = [{
                "name": name,
                "columns": ["content_id", "value"],
                "points": points,
            }]
            INFLUXDB_CLIENT.write_points(data)
        except Exception as e:
            LOGGER.exception(e)