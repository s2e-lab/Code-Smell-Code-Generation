def load_sgems_exp_var(filename):
    """ read an SGEM experimental variogram into a sequence of
    pandas.DataFrames

    Parameters
    ----------
    filename : (str)
        an SGEMS experimental variogram XML file

    Returns
    -------
    dfs : list
        a list of pandas.DataFrames of x, y, pairs for each
        division in the experimental variogram

    """

    assert os.path.exists(filename)
    import xml.etree.ElementTree as etree
    tree = etree.parse(filename)
    root = tree.getroot()
    dfs = {}
    for variogram in root:
        #print(variogram.tag)
        for attrib in variogram:

            #print(attrib.tag,attrib.text)
            if attrib.tag == "title":
                title = attrib.text.split(',')[0].split('=')[-1]
            elif attrib.tag == "x":
                x = [float(i) for i in attrib.text.split()]
            elif attrib.tag == "y":
                y = [float(i) for i in attrib.text.split()]
            elif attrib.tag == "pairs":
                pairs = [int(i) for i in attrib.text.split()]

            for item in attrib:
                print(item,item.tag)
        df = pd.DataFrame({"x":x,"y":y,"pairs":pairs})
        df.loc[df.y<0.0,"y"] = np.NaN
        dfs[title] = df
    return dfs