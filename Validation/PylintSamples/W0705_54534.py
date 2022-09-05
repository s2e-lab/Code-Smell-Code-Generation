def extract_tar(url, target_dir, additional_compression="", remove_common_prefix=False, overwrite=False):
    """ extract a targz and install to the target directory """
    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        tf = tarfile.TarFile.open(fileobj=download_to_bytesio(url))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        common_prefix = os.path.commonprefix(tf.getnames())
        if not common_prefix.endswith('/'):
            common_prefix += "/"
        for tfile in tf.getmembers():
            if remove_common_prefix:
                tfile.name = tfile.name.replace(common_prefix, "", 1)
            if tfile.name != "":
                target_path = os.path.join(target_dir, tfile.name)
                if target_path != target_dir and os.path.exists(target_path):
                    if overwrite:
                        remove_path(target_path)
                    else:
                        continue
                tf.extract(tfile, target_dir)
    except OSError:
        e = sys.exc_info()[1]
        raise ExtractException(str(e))
    except IOError:
        e = sys.exc_info()[1]
        raise ExtractException(str(e))