def _get_info_dir():
  """Get path to directory in which to store info files.

  The directory returned by this function is "owned" by this module. If
  the contents of the directory are modified other than via the public
  functions of this module, subsequent behavior is undefined.

  The directory will be created if it does not exist.
  """
  path = os.path.join(tempfile.gettempdir(), ".tensorboard-info")
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise
  else:
    os.chmod(path, 0o777)
  return path