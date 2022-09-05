def load(path=None, **kwargs):
    '''
    Loads the configuration from the file provided onto the device.

    path (required)
        Path where the configuration/template file is present. If the file has
        a ``.conf`` extension, the content is treated as text format. If the
        file has a ``.xml`` extension, the content is treated as XML format. If
        the file has a ``.set`` extension, the content is treated as Junos OS
        ``set`` commands.

    overwrite : False
        Set to ``True`` if you want this file is to completely replace the
        configuration file.

    replace : False
        Specify whether the configuration file uses ``replace:`` statements. If
        ``True``, only those statements under the ``replace`` tag will be
        changed.

    format
        Determines the format of the contents

    update : False
        Compare a complete loaded configuration against the candidate
        configuration. For each hierarchy level or configuration object that is
        different in the two configurations, the version in the loaded
        configuration replaces the version in the candidate configuration. When
        the configuration is later committed, only system processes that are
        affected by the changed configuration elements parse the new
        configuration. This action is supported from PyEZ 2.1.

    template_vars
      Variables to be passed into the template processing engine in addition to
      those present in pillar, the minion configuration, grains, etc.  You may
      reference these variables in your template like so:

      .. code-block:: jinja

          {{ template_vars["var_name"] }}

    CLI Examples:

    .. code-block:: bash

        salt 'device_name' junos.load 'salt://production/network/routers/config.set'

        salt 'device_name' junos.load 'salt://templates/replace_config.conf' replace=True

        salt 'device_name' junos.load 'salt://my_new_configuration.conf' overwrite=True

        salt 'device_name' junos.load 'salt://syslog_template.conf' template_vars='{"syslog_host": "10.180.222.7"}'
    '''
    conn = __proxy__['junos.conn']()
    ret = {}
    ret['out'] = True

    if path is None:
        ret['message'] = \
            'Please provide the salt path where the configuration is present'
        ret['out'] = False
        return ret

    op = {}
    if '__pub_arg' in kwargs:
        if kwargs['__pub_arg']:
            if isinstance(kwargs['__pub_arg'][-1], dict):
                op.update(kwargs['__pub_arg'][-1])
    else:
        op.update(kwargs)

    template_vars = {}
    if "template_vars" in op:
        template_vars = op["template_vars"]

    template_cached_path = salt.utils.files.mkstemp()
    __salt__['cp.get_template'](
        path,
        template_cached_path,
        template_vars=template_vars)

    if not os.path.isfile(template_cached_path):
        ret['message'] = 'Invalid file path.'
        ret['out'] = False
        return ret

    if os.path.getsize(template_cached_path) == 0:
        ret['message'] = 'Template failed to render'
        ret['out'] = False
        return ret

    op['path'] = template_cached_path

    if 'format' not in op:
        if path.endswith('set'):
            template_format = 'set'
        elif path.endswith('xml'):
            template_format = 'xml'
        else:
            template_format = 'text'

        op['format'] = template_format

    if 'replace' in op and op['replace']:
        op['merge'] = False
        del op['replace']
    elif 'overwrite' in op and op['overwrite']:
        op['overwrite'] = True
    elif 'overwrite' in op and not op['overwrite']:
        op['merge'] = True
        del op['overwrite']

    try:
        conn.cu.load(**op)
        ret['message'] = "Successfully loaded the configuration."
    except Exception as exception:
        ret['message'] = 'Could not load configuration due to : "{0}"'.format(
            exception)
        ret['format'] = op['format']
        ret['out'] = False
        return ret
    finally:
        salt.utils.files.safe_rm(template_cached_path)

    return ret