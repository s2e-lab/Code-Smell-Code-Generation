def start(self, host, nornir):
        """
        Run the task for the given host.

        Arguments:
            host (:obj:`nornir.core.inventory.Host`): Host we are operating with. Populated right
              before calling the ``task``
            nornir(:obj:`nornir.core.Nornir`): Populated right before calling
              the ``task``

        Returns:
            host (:obj:`nornir.core.task.MultiResult`): Results of the task and its subtasks
        """
        self.host = host
        self.nornir = nornir

        try:
            logger.debug("Host %r: running task %r", self.host.name, self.name)
            r = self.task(self, **self.params)
            if not isinstance(r, Result):
                r = Result(host=host, result=r)

        except NornirSubTaskError as e:
            tb = traceback.format_exc()
            logger.error(
                "Host %r: task %r failed with traceback:\n%s",
                self.host.name,
                self.name,
                tb,
            )
            r = Result(host, exception=e, result=str(e), failed=True)

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(
                "Host %r: task %r failed with traceback:\n%s",
                self.host.name,
                self.name,
                tb,
            )
            r = Result(host, exception=e, result=tb, failed=True)

        r.name = self.name
        r.severity_level = logging.ERROR if r.failed else self.severity_level

        self.results.insert(0, r)
        return self.results