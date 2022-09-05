def parse_from_xml(self, xml_spec):
        '''Parse a string or file containing an XML specification.

        Example:
        >>> s = RtsProfile()
        >>> s.parse_from_xml(open('test/rtsystem.xml'))
        >>> len(s.components)
        3

        Load of invalid data should throw exception:
        >>> s.parse_from_xml('non-XML string')
        Traceback (most recent call last):
        ...
        ExpatError: syntax error: line 1, column 0
        '''
        if type(xml_spec) in string_types():
            dom = xml.dom.minidom.parseString(xml_spec)
        else:
            dom = xml.dom.minidom.parse(xml_spec)
        self._parse_xml(dom)
        dom.unlink()