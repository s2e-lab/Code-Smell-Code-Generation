def main(argv=None):
    """ Script execution.

    The project repo will be cloned to a temporary directory, and the desired
    branch, tag, or commit will be checked out. Then, the application will be
    installed into a self-contained virtualenv environment.

    """
    @contextmanager
    def tmpdir():
        """ Create a self-deleting temporary directory. """
        path = mkdtemp()
        try:
            yield path
        finally:
            rmtree(path)
        return

    def test():
        """ Execute the test suite. """
        install = "{:s} install -r requirements-test.txt"
        check_call(install.format(pip).split())
        pytest = join(path, "bin", "py.test")
        test = "{:s} test/".format(pytest)
        check_call(test.split())
        uninstall = "{:s} uninstall -y -r requirements-test.txt"
        check_call(uninstall.format(pip).split())
        return

    args = _cmdline(argv)
    path = join(abspath(args.root), args.name)
    with tmpdir() as tmp:
        clone = "git clone {:s} {:s}".format(args.repo, tmp)
        check_call(clone.split())
        chdir(tmp)
        checkout = "git checkout {:s}".format(args.checkout)
        check_call(checkout.split())
        virtualenv = "virtualenv {:s}".format(path)
        check_call(virtualenv.split())
        pip = join(path, "bin", "pip")
        install = "{:s} install -U -r requirements.txt .".format(pip)
        check_call(install.split())
        if args.test:
            test()
    return 0