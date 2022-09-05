def render_xml(result, cfg, **kwargs):
    """
    Render to output a result in XML format
    """
    # Raw mode
    if cfg.dis == 'raw':
        return {'data': {'text/plain': result.decode('utf-8')},
                'metadata': {}}
    # Table
    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET
    root = ET.fromstring(result)
    try:
        ns = {'ns': re.match(r'\{([^}]+)\}', root.tag).group(1)}
    except Exception:
        raise KrnlException('Invalid XML data: cannot get namespace')
    columns = [c.attrib['name'] for c in root.find('ns:head', ns)]
    results = root.find('ns:results', ns)
    nrow = len(results)
    j = xml_iterator(columns, results, set(cfg.lan), add_vtype=cfg.typ)
    n, data = html_table(j, limit=cfg.lmt, withtype=cfg.typ)
    data += div('Total: {}, Shown: {}', nrow, n, css="tinfo")
    return {'data': {'text/html': div(data)},
            'metadata': {}}