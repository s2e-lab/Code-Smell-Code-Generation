def cleanup_environment():
    """
    Shutdown the ZEO server process running in another thread and cleanup the
    temporary directory.
    """
    SERV.terminate()
    shutil.rmtree(TMP_PATH)
    if os.path.exists(TMP_PATH):
        os.rmdir(TMP_PATH)

    global TMP_PATH
    TMP_PATH = None