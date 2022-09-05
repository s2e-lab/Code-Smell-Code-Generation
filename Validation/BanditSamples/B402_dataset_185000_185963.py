def download(supported_tags, date_array, tag, sat_id, 
             ftp_site='cdaweb.gsfc.nasa.gov', 
             data_path=None, user=None, password=None,
             fake_daily_files_from_monthly=False):
    """Routine to download NASA CDAWeb CDF data.
    
    This routine is intended to be used by pysat instrument modules supporting
    a particular NASA CDAWeb dataset.

    Parameters
    -----------
    supported_tags : dict
        dict of dicts. Keys are supported tag names for download. Value is 
        a dict with 'dir', 'remote_fname', 'local_fname'. Inteded to be
        pre-set with functools.partial then assigned to new instrument code.
    date_array : array_like
        Array of datetimes to download data for. Provided by pysat.
    tag : (str or NoneType)
        tag or None (default=None)
    sat_id : (str or NoneType)
        satellite id or None (default=None)
    data_path : (string or NoneType)
        Path to data directory.  If None is specified, the value previously
        set in Instrument.files.data_path is used.  (default=None)
    user : (string or NoneType)
        Username to be passed along to resource with relevant data.  
        (default=None)
    password : (string or NoneType)
        User password to be passed along to resource with relevant data.  
        (default=None)
    fake_daily_files_from_monthly : bool
        Some CDAWeb instrument data files are stored by month.This flag, 
        when true, accomodates this reality with user feedback on a monthly
        time frame. 

    Returns
    --------
    Void : (NoneType)
        Downloads data to disk.

    Examples
    --------
    :: 
    
        # download support added to cnofs_vefi.py using code below
        rn = '{year:4d}/cnofs_vefi_bfield_1sec_{year:4d}{month:02d}{day:02d}_v05.cdf'
        ln = 'cnofs_vefi_bfield_1sec_{year:4d}{month:02d}{day:02d}_v05.cdf'
        dc_b_tag = {'dir':'/pub/data/cnofs/vefi/bfield_1sec',
                    'remote_fname':rn,
                    'local_fname':ln}
        supported_tags = {'dc_b':dc_b_tag}
    
        download = functools.partial(nasa_cdaweb_methods.download, 
                                     supported_tags=supported_tags)
    
    """

    import os
    import ftplib

    # connect to CDAWeb default port
    ftp = ftplib.FTP(ftp_site) 
    # user anonymous, passwd anonymous@
    ftp.login()               
    
    try:
        ftp_dict = supported_tags[tag]
    except KeyError:
        raise ValueError('Tag name unknown.')
        
    # path to relevant file on CDAWeb
    ftp.cwd(ftp_dict['dir'])
    
    # naming scheme for files on the CDAWeb server
    remote_fname = ftp_dict['remote_fname']
    
    # naming scheme for local files, should be closely related
    # to CDAWeb scheme, though directory structures may be reduced
    # if desired
    local_fname = ftp_dict['local_fname']
    
    for date in date_array:            
        # format files for specific dates and download location
        formatted_remote_fname = remote_fname.format(year=date.year, 
                        month=date.month, day=date.day)
        formatted_local_fname = local_fname.format(year=date.year, 
                        month=date.month, day=date.day)
        saved_local_fname = os.path.join(data_path,formatted_local_fname) 

        # perform download                  
        try:
            print('Attempting to download file for '+date.strftime('%x'))
            sys.stdout.flush()
            ftp.retrbinary('RETR '+formatted_remote_fname, open(saved_local_fname,'wb').write)
            print('Finished.')
        except ftplib.error_perm as exception:
            # if exception[0][0:3] != '550':
            if str(exception.args[0]).split(" ", 1)[0] != '550':
                raise
            else:
                os.remove(saved_local_fname)
                print('File not available for '+ date.strftime('%x'))
    ftp.close()