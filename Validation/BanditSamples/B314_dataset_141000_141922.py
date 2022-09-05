def read_sgems_variogram_xml(xml_file,return_type=GeoStruct):
    """ function to read an SGEMS-type variogram XML file into
    a GeoStruct

    Parameters
    ----------
    xml_file : (str)
        SGEMS variogram XML file
    return_type :  (object)
        the instance type to return.  Default is GeoStruct

    Returns
    -------
    GeoStruct : GeoStruct


    Example
    -------
    ``>>>import pyemu``

    ``>>>gs = pyemu.utils.geostats.read_sgems_variogram_xml("sgems.xml")``

    """
    try:
        import xml.etree.ElementTree as ET

    except Exception as e:
        print("error import elementtree, skipping...")
    VARTYPE = {1: SphVario, 2: ExpVario, 3: GauVario, 4: None}
    assert os.path.exists(xml_file)
    tree = ET.parse(xml_file)
    gs_model = tree.getroot()
    structures = []
    variograms = []
    nugget = 0.0
    num_struct = 0
    for key,val in gs_model.items():
        #print(key,val)
        if str(key).lower() == "nugget":
            if len(val) > 0:
                nugget = float(val)
        if str(key).lower() == "structures_count":
            num_struct = int(val)
    if num_struct == 0:
        raise Exception("no structures found")
    if num_struct != 1:
        raise NotImplementedError()
    for structure in gs_model:
        vtype, contribution = None, None
        mx_range,mn_range = None, None
        x_angle,y_angle = None,None
        #struct_name = structure.tag
        for key,val in structure.items():
            key = str(key).lower()
            if key == "type":
                vtype = str(val).lower()
                if vtype.startswith("sph"):
                    vtype = SphVario
                elif vtype.startswith("exp"):
                    vtype = ExpVario
                elif vtype.startswith("gau"):
                    vtype = GauVario
                else:
                    raise Exception("unrecognized variogram type:{0}".format(vtype))

            elif key == "contribution":
                contribution = float(val)
            for item in structure:
                if item.tag.lower() == "ranges":
                    mx_range = float(item.attrib["max"])
                    mn_range = float(item.attrib["min"])
                elif item.tag.lower() == "angles":
                    x_angle = float(item.attrib["x"])
                    y_angle = float(item.attrib["y"])

        assert contribution is not None
        assert mn_range is not None
        assert mx_range is not None
        assert x_angle is not None
        assert y_angle is not None
        assert vtype is not None
        v = vtype(contribution=contribution,a=mx_range,
                  anisotropy=mx_range/mn_range,bearing=(180.0/np.pi)*np.arctan2(x_angle,y_angle),
                  name=structure.tag)
        return GeoStruct(nugget=nugget,variograms=[v])