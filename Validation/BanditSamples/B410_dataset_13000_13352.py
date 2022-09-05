def dict_to_xml(xml_dict):
    """
    Converts a dictionary to an XML ElementTree Element
    """
    import lxml.etree as etree
    root_tag = list(xml_dict.keys())[0]
    root = etree.Element(root_tag)
    _dict_to_xml_recurse(root, xml_dict[root_tag])
    return root