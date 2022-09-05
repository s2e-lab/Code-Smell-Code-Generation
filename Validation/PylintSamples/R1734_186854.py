def check(cls, dap, network=False, yamls=True, raises=False, logger=logger):
        '''Checks if the dap is valid, reports problems

        Parameters:
            network -- whether to run checks that requires network connection
            output -- where to write() problems, might be None
            raises -- whether to raise an exception immediately after problem is detected'''
        dap._check_raises = raises
        dap._problematic = False
        dap._logger = logger
        problems = list()

        problems += cls.check_meta(dap)
        problems += cls.check_no_self_dependency(dap)
        problems += cls.check_topdir(dap)
        problems += cls.check_files(dap)

        if yamls:
            problems += cls.check_yamls(dap)

        if network:
            problems += cls.check_name_not_on_dapi(dap)

        for problem in problems:
            dap._report_problem(problem.message, problem.level)

        del dap._check_raises
        return not dap._problematic