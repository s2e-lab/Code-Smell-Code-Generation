def _cmd_run(cmd, as_json=False):
    '''
    Ensure that the Pki module is loaded, and convert to and extract data from
    Json as needed.
    '''
    cmd_full = ['Import-Module -Name PKI; ']

    if as_json:
        cmd_full.append(r'ConvertTo-Json -Compress -Depth 4 -InputObject '
                        r'@({0})'.format(cmd))
    else:
        cmd_full.append(cmd)
    cmd_ret = __salt__['cmd.run_all'](
        six.text_type().join(cmd_full), shell='powershell', python_shell=True)

    if cmd_ret['retcode'] != 0:
        _LOG.error('Unable to execute command: %s\nError: %s', cmd,
                   cmd_ret['stderr'])

    if as_json:
        try:
            items = salt.utils.json.loads(cmd_ret['stdout'], strict=False)
            return items
        except ValueError:
            _LOG.error('Unable to parse return data as Json.')

    return cmd_ret['stdout']