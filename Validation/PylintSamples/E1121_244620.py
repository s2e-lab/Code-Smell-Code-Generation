def verify_arguments(self, args=None, kwargs=None):
        """Ensures that the arguments specified match the signature of the real method.

        :raise: ``VerifyingDoubleError`` if the arguments do not match.
        """

        args = self.args if args is None else args
        kwargs = self.kwargs if kwargs is None else kwargs

        try:
            verify_arguments(self._target, self._method_name, args, kwargs)
        except VerifyingBuiltinDoubleArgumentError:
            if doubles.lifecycle.ignore_builtin_verification():
                raise