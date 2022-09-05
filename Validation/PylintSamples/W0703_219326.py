def run_commandline(self, argv=None):
        """
        Run the command-line interface of this experiment.

        If ``argv`` is omitted it defaults to ``sys.argv``.

        Parameters
        ----------
        argv : list[str] or str, optional
            Command-line as string or list of strings like ``sys.argv``.

        Returns
        -------
        sacred.run.Run
            The Run object corresponding to the finished run.

        """
        argv = ensure_wellformed_argv(argv)
        short_usage, usage, internal_usage = self.get_usage()
        args = docopt(internal_usage, [str(a) for a in argv[1:]], help=False)

        cmd_name = args.get('COMMAND') or self.default_command
        config_updates, named_configs = get_config_updates(args['UPDATE'])

        err = self._check_command(cmd_name)
        if not args['help'] and err:
            print(short_usage)
            print(err)
            exit(1)

        if self._handle_help(args, usage):
            exit()

        try:
            return self.run(cmd_name, config_updates, named_configs, {}, args)
        except Exception as e:
            if self.current_run:
                debug = self.current_run.debug
            else:
                # The usual command line options are applied after the run
                # object is built completely. Some exceptions (e.g.
                # ConfigAddedError) are raised before this. In these cases,
                # the debug flag must be checked manually.
                debug = args.get('--debug', False)

            if debug:
                # Debug: Don't change behaviour, just re-raise exception
                raise
            elif self.current_run and self.current_run.pdb:
                # Print exception and attach pdb debugger
                import traceback
                import pdb
                traceback.print_exception(*sys.exc_info())
                pdb.post_mortem()
            else:
                # Handle pretty printing of exceptions. This includes
                # filtering the stacktrace and printing the usage, as
                # specified by the exceptions attributes
                if isinstance(e, SacredError):
                    print(format_sacred_error(e, short_usage), file=sys.stderr)
                else:
                    print_filtered_stacktrace()
                exit(1)