def run(self):
        """
        the main loop
        """
        try:
            master_process = BackgroundProcess.objects.filter(pk=self.process_id).first()
            if master_process:
                master_process.last_update = now()
                master_process.message = 'init child processes'
                master_process.save()
            else:
                self.delete_pid(force_del=True)
                self.stderr.write("no such process in BackgroundProcesses")
                sys.exit(0)

            self.manage_processes()
            while True:
                # handle signals
                sig = self.SIG_QUEUE.pop(0) if len(self.SIG_QUEUE) else None

                # check the DB connection
                check_db_connection()

                # update the P
                BackgroundProcess.objects.filter(pk=self.process_id).update(
                    last_update=now(),
                    message='running..')
                if sig is None:
                    self.manage_processes()
                elif sig not in self.SIGNALS:
                    logger.error('%s, unhandled signal %d' % (self.label, sig))
                    continue
                elif sig == signal.SIGTERM:
                    logger.debug('%s, termination signal' % self.label)
                    raise StopIteration
                elif sig == signal.SIGHUP:
                    # todo handle sighup
                    pass
                elif sig == signal.SIGUSR1:
                    # restart all child processes
                    logger.debug('PID %d, processed SIGUSR1 (%d) signal' % (self.pid, sig))
                    self.restart()
                elif sig == signal.SIGUSR2:
                    # write the process status to stdout
                    self.status()
                    pass
                sleep(5)
        except StopIteration:
            self.stop()
            self.delete_pid()
            sys.exit(0)
        except SystemExit:
            raise
        except:
            logger.error('%s(%d), unhandled exception\n%s' % (self.label, getpid(), traceback.format_exc()))