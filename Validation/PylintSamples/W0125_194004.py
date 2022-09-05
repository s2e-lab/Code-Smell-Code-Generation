def update_components(self, forcereload=False, forcerebuild=False,
                          forcecopy=True, install=False):
        """Check all known entry points for components. If necessary,
        manage configuration updates"""

        # TODO: See if we can pull out major parts of the component handling.
        # They are also used in the manage tool to instantiate the
        # component frontend bits.

        self.log("Updating components")
        components = {}

        if True:  # try:

            from pkg_resources import iter_entry_points

            entry_point_tuple = (
                iter_entry_points(group='hfos.base', name=None),
                iter_entry_points(group='hfos.sails', name=None),
                iter_entry_points(group='hfos.components', name=None)
            )

            for iterator in entry_point_tuple:
                for entry_point in iterator:
                    try:
                        name = entry_point.name
                        location = entry_point.dist.location
                        loaded = entry_point.load()

                        self.log("Entry point: ", entry_point,
                                 name,
                                 entry_point.resolve(), lvl=verbose)

                        self.log("Loaded: ", loaded, lvl=verbose)
                        comp = {
                            'package': entry_point.dist.project_name,
                            'location': location,
                            'version': str(entry_point.dist.parsed_version),
                            'description': loaded.__doc__
                        }

                        components[name] = comp
                        self.loadable_components[name] = loaded

                        self.log("Loaded component:", comp, lvl=verbose)

                    except Exception as e:
                        self.log("Could not inspect entrypoint: ", e,
                                 type(e), entry_point, iterator, lvl=error,
                                 exc=True)

                        # for name in components.keys():
                        #     try:
                        #         self.log(self.loadable_components[name])
                        #         configobject = {
                        #             'type': 'object',
                        #             'properties':
                        # self.loadable_components[name].configprops
                        #         }
                        #         ComponentBaseConfigSchema['schema'][
                        # 'properties'][
                        #             'settings'][
                        #             'oneOf'].append(configobject)
                        #     except (KeyError, AttributeError) as e:
                        #         self.log('Problematic configuration
                        # properties in '
                        #                  'component ', name, exc=True)
                        #
                        # schemastore['component'] = ComponentBaseConfigSchema

        # except Exception as e:
        #    self.log("Error: ", e, type(e), lvl=error, exc=True)
        #    return

        self.log("Checking component frontend bits in ", self.frontendroot,
                 lvl=verbose)

        # pprint(self.config._fields)
        diff = set(components) ^ set(self.config.components)
        if diff or forcecopy and self.config.frontendenabled:
            self.log("Old component configuration differs:", diff, lvl=debug)
            self.log(self.config.components, components, lvl=verbose)
            self.config.components = components
        else:
            self.log("No component configuration change. Proceeding.")

        if forcereload:
            self.log("Restarting all components.", lvl=warn)
            self._instantiate_components(clear=True)