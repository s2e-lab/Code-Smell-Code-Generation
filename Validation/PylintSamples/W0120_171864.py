def start(arguments, timeout=datetime.timedelta(seconds=60)):
  """Start a new TensorBoard instance, or reuse a compatible one.

  If the cache key determined by the provided arguments and the current
  working directory (see `cache_key`) matches the cache key of a running
  TensorBoard process (see `get_all`), that process will be reused.

  Otherwise, a new TensorBoard process will be spawned with the provided
  arguments, using the `tensorboard` binary from the system path.

  Args:
    arguments: List of strings to be passed as arguments to
      `tensorboard`. (If you have a raw command-line string, see
      `shlex.split`.)
    timeout: `datetime.timedelta` object describing how long to wait for
      the subprocess to initialize a TensorBoard server and write its
      `TensorBoardInfo` file. If the info file is not written within
      this time period, `start` will assume that the subprocess is stuck
      in a bad state, and will give up on waiting for it and return a
      `StartTimedOut` result. Note that in such a case the subprocess
      will not be killed. Default value is 60 seconds.

  Returns:
    A `StartReused`, `StartLaunched`, `StartFailed`, or `StartTimedOut`
    object.
  """
  match = _find_matching_instance(
      cache_key(
          working_directory=os.getcwd(),
          arguments=arguments,
          configure_kwargs={},
      ),
  )
  if match:
    return StartReused(info=match)

  (stdout_fd, stdout_path) = tempfile.mkstemp(prefix=".tensorboard-stdout-")
  (stderr_fd, stderr_path) = tempfile.mkstemp(prefix=".tensorboard-stderr-")
  start_time_seconds = time.time()
  try:
    p = subprocess.Popen(
        ["tensorboard"] + arguments,
        stdout=stdout_fd,
        stderr=stderr_fd,
    )
  finally:
    os.close(stdout_fd)
    os.close(stderr_fd)

  poll_interval_seconds = 0.5
  end_time_seconds = start_time_seconds + timeout.total_seconds()
  while time.time() < end_time_seconds:
    time.sleep(poll_interval_seconds)
    subprocess_result = p.poll()
    if subprocess_result is not None:
      return StartFailed(
          exit_code=subprocess_result,
          stdout=_maybe_read_file(stdout_path),
          stderr=_maybe_read_file(stderr_path),
      )
    for info in get_all():
      if info.pid == p.pid and info.start_time >= start_time_seconds:
        return StartLaunched(info=info)
  else:
    return StartTimedOut(pid=p.pid)