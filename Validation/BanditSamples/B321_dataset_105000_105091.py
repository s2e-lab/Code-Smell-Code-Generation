def update_old_names():
    """Fetches the list of old tz names and returns a mapping"""

    url = urlparse(ZONEINFO_URL)
    log.info('Connecting to %s' % url.netloc)
    ftp = ftplib.FTP(url.netloc)
    ftp.login()
    gzfile = BytesIO()

    log.info('Fetching zoneinfo database')
    ftp.retrbinary('RETR ' + url.path, gzfile.write)
    gzfile.seek(0)

    log.info('Extracting backwards data')
    archive = tarfile.open(mode="r:gz", fileobj=gzfile)
    backward = {}
    for line in archive.extractfile('backward').readlines():
        if line[0] == '#':
            continue
        if len(line.strip()) == 0:
            continue
        parts = line.split()
        if parts[0] != b'Link':
            continue

        backward[parts[2].decode('ascii')] = parts[1].decode('ascii')

    return backward