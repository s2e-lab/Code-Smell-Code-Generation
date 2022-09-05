def find_gsc_offset(image, input_catalog='GSC1', output_catalog='GAIA'):
    """Find the GSC to GAIA offset based on guide star coordinates

    Parameters
    ----------
    image : str
        Filename of image to be processed.

    Returns
    -------
    delta_ra, delta_dec : tuple of floats
        Offset in decimal degrees of image based on correction to guide star
        coordinates relative to GAIA.
    """
    serviceType = "GSCConvert/GSCconvert.aspx"
    spec_str = "TRANSFORM={}-{}&IPPPSSOOT={}"

    if 'rootname' in pf.getheader(image):
        ippssoot = pf.getval(image, 'rootname').upper()
    else:
        ippssoot = fu.buildNewRootname(image).upper()

    spec = spec_str.format(input_catalog, output_catalog, ippssoot)
    serviceUrl = "{}/{}?{}".format(SERVICELOCATION, serviceType, spec)
    rawcat = requests.get(serviceUrl)
    if not rawcat.ok:
        log.info("Problem accessing service with:\n{{}".format(serviceUrl))
        raise ValueError

    delta_ra = delta_dec = None
    tree = BytesIO(rawcat.content)
    for _, element in etree.iterparse(tree):
        if element.tag == 'deltaRA':
            delta_ra = float(element.text)
        elif element.tag == 'deltaDEC':
            delta_dec = float(element.text)

    return delta_ra, delta_dec