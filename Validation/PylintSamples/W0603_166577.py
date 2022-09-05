def main() -> None:
    """Command-line wrapper to re-run a script whenever its source changes.

    Scripts may be specified by filename or module name::

        python -m tornado.autoreload -m tornado.test.runtests
        python -m tornado.autoreload tornado/test/runtests.py

    Running a script with this wrapper is similar to calling
    `tornado.autoreload.wait` at the end of the script, but this wrapper
    can catch import-time problems like syntax errors that would otherwise
    prevent the script from reaching its call to `wait`.
    """
    # Remember that we were launched with autoreload as main.
    # The main module can be tricky; set the variables both in our globals
    # (which may be __main__) and the real importable version.
    import tornado.autoreload

    global _autoreload_is_main
    global _original_argv, _original_spec
    tornado.autoreload._autoreload_is_main = _autoreload_is_main = True
    original_argv = sys.argv
    tornado.autoreload._original_argv = _original_argv = original_argv
    original_spec = getattr(sys.modules["__main__"], "__spec__", None)
    tornado.autoreload._original_spec = _original_spec = original_spec
    sys.argv = sys.argv[:]
    if len(sys.argv) >= 3 and sys.argv[1] == "-m":
        mode = "module"
        module = sys.argv[2]
        del sys.argv[1:3]
    elif len(sys.argv) >= 2:
        mode = "script"
        script = sys.argv[1]
        sys.argv = sys.argv[1:]
    else:
        print(_USAGE, file=sys.stderr)
        sys.exit(1)

    try:
        if mode == "module":
            import runpy

            runpy.run_module(module, run_name="__main__", alter_sys=True)
        elif mode == "script":
            with open(script) as f:
                # Execute the script in our namespace instead of creating
                # a new one so that something that tries to import __main__
                # (e.g. the unittest module) will see names defined in the
                # script instead of just those defined in this module.
                global __file__
                __file__ = script
                # If __package__ is defined, imports may be incorrectly
                # interpreted as relative to this module.
                global __package__
                del __package__
                exec_in(f.read(), globals(), globals())
    except SystemExit as e:
        logging.basicConfig()
        gen_log.info("Script exited with status %s", e.code)
    except Exception as e:
        logging.basicConfig()
        gen_log.warning("Script exited with uncaught exception", exc_info=True)
        # If an exception occurred at import time, the file with the error
        # never made it into sys.modules and so we won't know to watch it.
        # Just to make sure we've covered everything, walk the stack trace
        # from the exception and watch every file.
        for (filename, lineno, name, line) in traceback.extract_tb(sys.exc_info()[2]):
            watch(filename)
        if isinstance(e, SyntaxError):
            # SyntaxErrors are special:  their innermost stack frame is fake
            # so extract_tb won't see it and we have to get the filename
            # from the exception object.
            watch(e.filename)
    else:
        logging.basicConfig()
        gen_log.info("Script exited normally")
    # restore sys.argv so subsequent executions will include autoreload
    sys.argv = original_argv

    if mode == "module":
        # runpy did a fake import of the module as __main__, but now it's
        # no longer in sys.modules.  Figure out where it is and watch it.
        loader = pkgutil.get_loader(module)
        if loader is not None:
            watch(loader.get_filename())  # type: ignore

    wait()