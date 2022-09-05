def user_list(user=None, password=None, host=None, port=None, database='admin', authdb=None):
    '''
    List users of a MongoDB database

    CLI Example:

    .. code-block:: bash

        salt '*' mongodb.user_list <user> <password> <host> <port> <database>
    '''
    conn = _connect(user, password, host, port, authdb=authdb)
    if not conn:
        return 'Failed to connect to mongo database'

    try:
        log.info('Listing users')
        mdb = pymongo.database.Database(conn, database)

        output = []
        mongodb_version = _version(mdb)

        if _LooseVersion(mongodb_version) >= _LooseVersion('2.6'):
            for user in mdb.command('usersInfo')['users']:
                output.append(
                    {'user': user['user'],
                     'roles': user['roles']}
                )
        else:
            for user in mdb.system.users.find():
                output.append(
                    {'user': user['user'],
                     'readOnly': user.get('readOnly', 'None')}
                )
        return output

    except pymongo.errors.PyMongoError as err:
        log.error('Listing users failed with error: %s', err)
        return six.text_type(err)