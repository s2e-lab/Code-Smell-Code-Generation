def query_nexus(query_url, timeout_sec, basic_auth=None):
    """Queries Nexus for an artifact

    :param query_url: (str) Query URL
    :param timeout_sec: (int) query timeout
    :param basic_auth (HTTPBasicAuth) object or none
    :return: requests.Response object
    :raises: RuntimeError
    """
    log = logging.getLogger(mod_logger + '.query_nexus')

    # Attempt to query Nexus
    retry_sec = 5
    max_retries = 6
    try_num = 1
    query_success = False
    nexus_response = None
    while try_num <= max_retries:
        if query_success:
            break
        log.debug('Attempt # {n} of {m} to query the Nexus URL: {u}'.format(n=try_num, u=query_url, m=max_retries))
        try:
            nexus_response = requests.get(query_url, auth=basic_auth, stream=True, timeout=timeout_sec)
        except requests.exceptions.Timeout:
            _, ex, trace = sys.exc_info()
            msg = '{n}: Nexus initial query timed out after {t} seconds:\n{e}'.format(
                n=ex.__class__.__name__, t=timeout_sec, r=retry_sec, e=str(ex))
            log.warn(msg)
            if try_num < max_retries:
                log.info('Retrying query in {t} sec...'.format(t=retry_sec))
                time.sleep(retry_sec)
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
            _, ex, trace = sys.exc_info()
            msg = '{n}: Nexus initial query failed with the following exception:\n{e}'.format(
                n=ex.__class__.__name__, r=retry_sec, e=str(ex))
            log.warn(msg)
            if try_num < max_retries:
                log.info('Retrying query in {t} sec...'.format(t=retry_sec))
                time.sleep(retry_sec)
        else:
            query_success = True
        try_num += 1

    if not query_success:
        msg = 'Unable to query Nexus after {m} attempts using URL: {u}'.format(
            u=query_url, m=max_retries)
        log.error(msg)
        raise RuntimeError(msg)

    if nexus_response.status_code != 200:
        msg = 'Nexus request returned code {c}, unable to query Nexus using URL: {u}'.format(
            u=query_url, c=nexus_response.status_code)
        log.error(msg)
        raise RuntimeError(msg)
    return nexus_response