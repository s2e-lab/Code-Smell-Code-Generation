def InternalPlatformSwitch(funcname, *args, **kwargs):
  """Determine, on a platform-specific basis, which module to use."""
  # pylint: disable=g-import-not-at-top
  clz = None
  if sys.platform.startswith('linux'):
    from pyu2f.hid import linux
    clz = linux.LinuxHidDevice
  elif sys.platform.startswith('win32'):
    from pyu2f.hid import windows
    clz = windows.WindowsHidDevice
  elif sys.platform.startswith('darwin'):
    from pyu2f.hid import macos
    clz = macos.MacOsHidDevice

  if not clz:
    raise Exception('Unsupported platform: ' + sys.platform)

  if funcname == '__init__':
    return clz(*args, **kwargs)
  return getattr(clz, funcname)(*args, **kwargs)