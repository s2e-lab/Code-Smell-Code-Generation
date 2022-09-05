def list_absent(name, value, delimiter=DEFAULT_TARGET_DELIM):
    '''
    Delete a value from a grain formed as a list.

    .. versionadded:: 2014.1.0

    name
        The grain name.

    value
       The value to delete from the grain list.

    delimiter
        A delimiter different from the default ``:`` can be provided.

        .. versionadded:: v2015.8.2

    The grain should be `list type <http://docs.python.org/2/tutorial/datastructures.html#data-structures>`_

    .. code-block:: yaml

        roles:
          grains.list_absent:
            - value: db

    For multiple grains, the syntax looks like:

    .. code-block:: yaml

        roles:
          grains.list_absent:
            - value:
              - web
              - dev
    '''

    name = re.sub(delimiter, DEFAULT_TARGET_DELIM, name)
    ret = {'name': name,
           'changes': {},
           'result': True,
           'comment': ''}
    comments = []
    grain = __salt__['grains.get'](name, None)
    if grain:
        if isinstance(grain, list):
            if not isinstance(value, list):
                value = [value]
            for val in value:
                if val not in grain:
                    comments.append('Value {1} is absent from '
                                      'grain {0}'.format(name, val))
                elif __opts__['test']:
                    ret['result'] = None
                    comments.append('Value {1} in grain {0} is set '
                                     'to be deleted'.format(name, val))
                    if 'deleted' not in ret['changes'].keys():
                        ret['changes'] = {'deleted': []}
                    ret['changes']['deleted'].append(val)
                elif val in grain:
                    __salt__['grains.remove'](name, val)
                    comments.append('Value {1} was deleted from '
                                     'grain {0}'.format(name, val))
                    if 'deleted' not in ret['changes'].keys():
                        ret['changes'] = {'deleted': []}
                    ret['changes']['deleted'].append(val)
            ret['comment'] = '\n'.join(comments)
            return ret
        else:
            ret['result'] = False
            ret['comment'] = 'Grain {0} is not a valid list'\
                             .format(name)
    else:
        ret['comment'] = 'Grain {0} does not exist'.format(name)
    return ret