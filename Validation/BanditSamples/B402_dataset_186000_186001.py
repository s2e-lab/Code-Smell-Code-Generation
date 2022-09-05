def download(date_array, tag, sat_id, data_path, user=None, password=None):
    """Routine to download Kp index data

    Parameters
    -----------
    tag : (string or NoneType)
        Denotes type of file to load.  Accepted types are '1min' and '5min'.
        (default=None)
    sat_id : (string or NoneType)
        Specifies the satellite ID for a constellation.  Not used.
        (default=None)
    data_path : (string or NoneType)
        Path to data directory.  If None is specified, the value previously
        set in Instrument.files.data_path is used.  (default=None)

    Returns
    --------
    Void : (NoneType)
        data downloaded to disk, if available.
    
    Notes
    -----
    Called by pysat. Not intended for direct use by user.

    """

    import ftplib
    from ftplib import FTP
    import sys
    ftp = FTP('ftp.gfz-potsdam.de')   # connect to host, default port
    ftp.login()               # user anonymous, passwd anonymous@
    ftp.cwd('/pub/home/obs/kp-ap/tab')

    for date in date_array:
        fname = 'kp{year:02d}{month:02d}.tab'
        fname = fname.format(year=(date.year - date.year//100*100), month=date.month)
        local_fname = fname
        saved_fname = os.path.join(data_path,local_fname) 
        try:
            print('Downloading file for '+date.strftime('%D'))
            sys.stdout.flush()
            ftp.retrbinary('RETR '+fname, open(saved_fname,'wb').write)
        except ftplib.error_perm as exception:
            # if exception[0][0:3] != '550':
            if str(exception.args[0]).split(" ", 1)[0] != '550':
                raise
            else:
                os.remove(saved_fname)
                print('File not available for '+date.strftime('%D'))
    
    ftp.close()
    return