def prettify(unicode_text):
    """Return a pretty-printed version of a unicode XML string.

    Useful for debugging.

    Args:
        unicode_text (str): A text representation of XML (unicode,
            *not* utf-8).

    Returns:
        str: A pretty-printed version of the input.

    """
    import xml.dom.minidom
    reparsed = xml.dom.minidom.parseString(unicode_text.encode('utf-8'))
    return reparsed.toprettyxml(indent="  ", newl="\n")