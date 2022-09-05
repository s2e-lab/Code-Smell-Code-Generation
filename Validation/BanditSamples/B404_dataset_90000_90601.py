def _check_virtualenv():
    """Makes sure that the virtualenv specified in the global settings file
    actually exists.
    """
    from os import waitpid
    from subprocess import Popen, PIPE
    penvs = Popen("source /usr/local/bin/virtualenvwrapper.sh; workon",
                 shell=True, executable="/bin/bash", stdout=PIPE, stderr=PIPE)
    waitpid(penvs.pid, 0)
    envs = penvs.stdout.readlines()
    enverr = penvs.stderr.readlines()
    result = (settings.venv + '\n') in envs and len(enverr) == 0

    vms("Find virtualenv: {}".format(' '.join(envs).replace('\n', '')))
    vms("Find virtualenv | stderr: {}".format(' '.join(enverr)))
    
    if not result:
        info(envs)
        err("The virtualenv '{}' does not exist; can't use CI server.".format(settings.venv))
        if len(enverr) > 0:
            map(err, enverr)
    return result