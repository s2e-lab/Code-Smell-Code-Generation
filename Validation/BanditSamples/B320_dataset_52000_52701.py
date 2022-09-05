def pformat_xml(xml):
    """Return pretty formatted XML."""
    try:
        from lxml import etree  # delayed import
        if not isinstance(xml, bytes):
            xml = xml.encode('utf-8')
        xml = etree.parse(io.BytesIO(xml))
        xml = etree.tostring(xml, pretty_print=True, xml_declaration=True,
                             encoding=xml.docinfo.encoding)
        xml = bytes2str(xml)
    except Exception:
        if isinstance(xml, bytes):
            xml = bytes2str(xml)
        xml = xml.replace('><', '>\n<')
    return xml.replace('  ', ' ').replace('\t', ' ')