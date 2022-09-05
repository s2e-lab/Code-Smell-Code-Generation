def _symlink_bcbio(args, script="bcbio_nextgen.py", env_name=None, prefix=None):
    """Ensure a bcbio-nextgen script symlink in final tool directory.
    """
    if env_name:
        bcbio_anaconda = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(sys.executable))),
                                      "envs", env_name, "bin", script)
    else:
        bcbio_anaconda = os.path.join(os.path.dirname(os.path.realpath(sys.executable)), script)
    bindir = os.path.join(args.tooldir, "bin")
    if not os.path.exists(bindir):
        os.makedirs(bindir)
    if prefix:
        script = "%s_%s" % (prefix, script)
    bcbio_final = os.path.join(bindir, script)
    if not os.path.exists(bcbio_final):
        if os.path.lexists(bcbio_final):
            subprocess.check_call(["rm", "-f", bcbio_final])
        subprocess.check_call(["ln", "-s", bcbio_anaconda, bcbio_final])