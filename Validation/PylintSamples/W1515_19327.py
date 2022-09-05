def install_pdb_handler():
  """Make CTRL+\ break into gdb."""

  import signal
  import pdb

  def handler(_signum, _frame):
    pdb.set_trace()

  signal.signal(signal.SIGQUIT, handler)