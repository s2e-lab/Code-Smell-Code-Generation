def retry(retries=KAFKA_WAIT_RETRIES, delay=KAFKA_WAIT_INTERVAL,
          check_exceptions=()):
    """Retry decorator."""
    def decorator(func):
        """Decorator."""
        def f_retry(*args, **kwargs):
            """Retry running function on exception after delay."""
            for i in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                # pylint: disable=W0703
                # We want to catch all exceptions here to retry.
                except check_exceptions + (Exception,) as exc:
                    if i < retries:
                        logger.info('Connection attempt %d of %d failed',
                                    i, retries)
                        if isinstance(exc, check_exceptions):
                            logger.debug('Caught known exception, retrying...',
                                         exc_info=True)
                        else:
                            logger.warn(
                                'Caught unknown exception, retrying...',
                                exc_info=True)
                    else:
                        logger.exception('Failed after %d attempts', retries)

                        raise

                # No exception so wait before retrying
                time.sleep(delay)

        return f_retry
    return decorator