def __valid_on_demand_ext_pillar(self, opts):
        '''
        Check to see if the on demand external pillar is allowed
        '''
        if not isinstance(self.ext, dict):
            log.error(
                'On-demand pillar %s is not formatted as a dictionary',
                self.ext
            )
            return False

        on_demand = opts.get('on_demand_ext_pillar', [])
        try:
            invalid_on_demand = set([x for x in self.ext if x not in on_demand])
        except TypeError:
            # Prevent traceback when on_demand_ext_pillar option is malformed
            log.error(
                'The \'on_demand_ext_pillar\' configuration option is '
                'malformed, it should be a list of ext_pillar module names'
            )
            return False

        if invalid_on_demand:
            log.error(
                'The following ext_pillar modules are not allowed for '
                'on-demand pillar data: %s. Valid on-demand ext_pillar '
                'modules are: %s. The valid modules can be adjusted by '
                'setting the \'on_demand_ext_pillar\' config option.',
                ', '.join(sorted(invalid_on_demand)),
                ', '.join(on_demand),
            )
            return False
        return True