def _prepare_event(eventinstances):
    """Converts event instances to a relevant dictionary."""
    import xml.etree.ElementTree as ET

    def parse_event(events):
        """Find all events inside of an topicset list.

        MessageInstance signals that subsequent children will
        contain source and data descriptions.
        """

        def clean_attrib(attrib={}):
            """Clean up child attributes by removing XML namespace."""
            attributes = {}
            for key, value in attrib.items():
                attributes[key.split('}')[-1]] = value
            return attributes

        description = {}
        for child in events:
            child_tag = child.tag.split('}')[-1]
            child_attrib = clean_attrib(child.attrib)
            if child_tag != 'MessageInstance':
                description[child_tag] = {
                    **child_attrib, **parse_event(child)}
            elif child_tag == 'MessageInstance':
                description = {}
                for item in child:
                    tag = item.tag.split('}')[-1]
                    description[tag] = clean_attrib(item[0].attrib)
        return description

    root = ET.fromstring(eventinstances)
    return parse_event(root[0][0][0])