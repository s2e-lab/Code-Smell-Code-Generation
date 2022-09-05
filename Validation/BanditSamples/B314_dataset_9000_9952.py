def show_ahrs_calibration(clb_upi, version='3'):
    """Show AHRS calibration data for given `clb_upi`."""
    db = DBManager()
    ahrs_upi = clbupi2ahrsupi(clb_upi)
    print("AHRS UPI: {}".format(ahrs_upi))
    content = db._get_content("show_product_test.htm?upi={0}&"
                              "testtype=AHRS-CALIBRATION-v{1}&n=1&out=xml"
                              .format(ahrs_upi, version)) \
        .replace('\n', '')

    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(io.StringIO(content)).getroot()
    except ET.ParseError:
        print("No calibration data found")
    else:
        for child in root:
            print("{}: {}".format(child.tag, child.text))
        names = [c.text for c in root.findall(".//Name")]
        values = [[i.text for i in c] for c in root.findall(".//Values")]
        for name, value in zip(names, values):
            print("{}: {}".format(name, value))