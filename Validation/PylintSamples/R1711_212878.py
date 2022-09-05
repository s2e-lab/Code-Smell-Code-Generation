def extract_bzip2 (archive, compression, cmd, verbosity, interactive, outdir):
    """Extract a BZIP2 archive with the bz2 Python module."""
    targetname = util.get_single_outfile(outdir, archive)
    try:
        with bz2.BZ2File(archive) as bz2file:
            with open(targetname, 'wb') as targetfile:
                data = bz2file.read(READ_SIZE_BYTES)
                while data:
                    targetfile.write(data)
                    data = bz2file.read(READ_SIZE_BYTES)
    except Exception as err:
        msg = "error extracting %s to %s: %s" % (archive, targetname, err)
        raise util.PatoolError(msg)
    return None