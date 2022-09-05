def _complete_exit(self, cmd, args, text):
        """Find candidates for the 'exit' command."""
        if args:
            return
        return [ x for x in { 'root', 'all', } \
                if x.startswith(text) ]