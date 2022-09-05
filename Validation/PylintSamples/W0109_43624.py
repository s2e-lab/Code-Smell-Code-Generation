def _get_udev_rules(self, channel_read, channel_write, channel_data):
        """construct udev rules info."""
        sub_str = '%(read)s %%k %(read)s %(write)s %(data)s qeth' % {
                                      'read': channel_read,
                                       'read': channel_read,
                                       'write': channel_write,
                                       'data': channel_data}
        rules_str = '# Configure qeth device at'
        rules_str += ' %(read)s/%(write)s/%(data)s\n' % {
                             'read': channel_read,
                             'write': channel_write,
                             'data': channel_data}
        rules_str += ('ACTION==\"add\", SUBSYSTEM==\"drivers\", KERNEL=='
               '\"qeth\", IMPORT{program}=\"collect %s\"\n') % sub_str
        rules_str += ('ACTION==\"add\", SUBSYSTEM==\"ccw\", KERNEL==\"'
           '%(read)s\", IMPORT{program}="collect %(channel)s\"\n') % {
                                 'read': channel_read, 'channel': sub_str}
        rules_str += ('ACTION==\"add\", SUBSYSTEM==\"ccw\", KERNEL==\"'
           '%(write)s\", IMPORT{program}=\"collect %(channel)s\"\n') % {
                            'write': channel_write, 'channel': sub_str}
        rules_str += ('ACTION==\"add\", SUBSYSTEM==\"ccw\", KERNEL==\"'
           '%(data)s\", IMPORT{program}=\"collect %(channel)s\"\n') % {
                                'data': channel_data, 'channel': sub_str}
        rules_str += ('ACTION==\"remove\", SUBSYSTEM==\"drivers\", KERNEL==\"'
           'qeth\", IMPORT{program}=\"collect --remove %s\"\n') % sub_str
        rules_str += ('ACTION==\"remove\", SUBSYSTEM==\"ccw\", KERNEL==\"'
           '%(read)s\", IMPORT{program}=\"collect --remove %(channel)s\"\n'
                   ) % {'read': channel_read, 'channel': sub_str}
        rules_str += ('ACTION==\"remove\", SUBSYSTEM==\"ccw\", KERNEL==\"'
           '%(write)s\", IMPORT{program}=\"collect --remove %(channel)s\"\n'
                   ) % {'write': channel_write, 'channel': sub_str}
        rules_str += ('ACTION==\"remove\", SUBSYSTEM==\"ccw\", KERNEL==\"'
           '%(data)s\", IMPORT{program}=\"collect --remove %(channel)s\"\n'
                   ) % {'data': channel_data, 'channel': sub_str}
        rules_str += ('TEST==\"[ccwgroup/%(read)s]\", GOTO=\"qeth-%(read)s'
           '-end\"\n') % {'read': channel_read, 'read': channel_read}
        rules_str += ('ACTION==\"add\", SUBSYSTEM==\"ccw\", ENV{COLLECT_'
           '%(read)s}==\"0\", ATTR{[drivers/ccwgroup:qeth]group}=\"'
           '%(read)s,%(write)s,%(data)s\"\n') % {
                    'read': channel_read, 'read': channel_read,
                    'write': channel_write, 'data': channel_data}
        rules_str += ('ACTION==\"add\", SUBSYSTEM==\"drivers\", KERNEL==\"qeth'
           '\", ENV{COLLECT_%(read)s}==\"0\", ATTR{[drivers/'
           'ccwgroup:qeth]group}=\"%(read)s,%(write)s,%(data)s\"\n'
           'LABEL=\"qeth-%(read)s-end\"\n') % {
            'read': channel_read, 'read': channel_read, 'write': channel_write,
            'data': channel_data, 'read': channel_read}
        rules_str += ('ACTION==\"add\", SUBSYSTEM==\"ccwgroup\", KERNEL=='
           '\"%s\", ATTR{layer2}=\"1\"\n') % channel_read
        rules_str += ('ACTION==\"add\", SUBSYSTEM==\"ccwgroup\", KERNEL=='
           '\"%s\", ATTR{online}=\"1\"\n') % channel_read
        return rules_str