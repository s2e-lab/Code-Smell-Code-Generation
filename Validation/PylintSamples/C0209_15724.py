def _check_holiday_structure(self, times):
        """ To check the structure of the HolidayClass

        :param list times: years or months or days or number week
        :rtype: None or Exception
        :return: in the case of exception returns the exception
        """

        if not isinstance(times, list):
            raise TypeError("an list is required")

        for time in times:
            if not isinstance(time, tuple):
                raise TypeError("a tuple is required")
            if len(time) > 5:
                raise TypeError("Target time takes at most 5 arguments"
                                " ('%d' given)" % len(time))
            if len(time) < 5:
                raise TypeError("Required argument '%s' (pos '%d')"
                                " not found" % (TIME_LABEL[len(time)], len(time)))

            self._check_time_format(TIME_LABEL, time)(base)