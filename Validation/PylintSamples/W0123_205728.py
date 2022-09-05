def _evaluate(self, message):
        """
        Evaluate the expression with the given Python object in its locals.

        @param message: A decoded JSON input.

        @return: The resulting object.
        """
        return eval(
            self.code,
            globals(), {
                "J": message,
                "timedelta": timedelta,
                "datetime": datetime,
                "SKIP": self._SKIP})