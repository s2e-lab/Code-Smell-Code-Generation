def _parse_data(self, data, charset):
        """ Parse the xml data into dictionary. """

        builder = TreeBuilder(numbermode=self._numbermode)
        if isinstance(data,basestring):
            xml.sax.parseString(data, builder)
        else:
            xml.sax.parse(data, builder)
        return builder.root[self._root_element_name()]