def _redis_process_checkpoint(host, port):
    '''this helper method checks if
    redis server is available in the sys
    if not fires up one
    '''
    try:
        subprocess.check_output("pgrep redis", shell=True)
    except Exception:
        logger.warning(
            'Your redis server is offline, fake2db will try to launch it now!',
            extra=extra_information)
        # close_fds = True argument is the flag that is responsible
        # for Popen to launch the process completely independent
        subprocess.Popen("redis-server --bind %s --port %s" % (host, port),
                         close_fds=True,
                         shell=True)
        time.sleep(3)