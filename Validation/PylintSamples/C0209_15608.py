def _update_status(self, sub_job_num=None):
        """Gets the job status.

        Return:
            str: The current status of the job

        """
        job_id = '%s.%s' % (self.cluster_id, sub_job_num) if sub_job_num else str(self.cluster_id)
        format = ['-format', '"%d"', 'JobStatus']
        cmd = 'condor_q {0} {1} && condor_history {0} {1}'.format(job_id, ' '.join(format))
        args = [cmd]
        out, err = self._execute(args, shell=True, run_in_job_dir=False)
        if err:
            log.error('Error while updating status for job %s: %s', job_id, err)
            raise HTCondorError(err)
        if not out:
            log.error('Error while updating status for job %s: Job not found.', job_id)
            raise HTCondorError('Job not found.')

        out = out.replace('\"', '')
        log.info('Job %s status: %s', job_id, out)

        if not sub_job_num:
            if len(out) >= self.num_jobs:
                out = out[:self.num_jobs]
            else:
                msg = 'There are {0} sub-jobs, but {1} status(es).'.format(self.num_jobs, len(out))
                log.error(msg)
                raise HTCondorError(msg)

        #initialize status dictionary
        status_dict = dict()
        for val in CONDOR_JOB_STATUSES.values():
            status_dict[val] = 0

        for status_code_str in out:
            status_code = 0
            try:
                status_code = int(status_code_str)
            except ValueError:
                pass
            key = CONDOR_JOB_STATUSES[status_code]
            status_dict[key] += 1

        return status_dict