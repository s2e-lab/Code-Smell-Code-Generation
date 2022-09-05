def load_ui_type(uifile):
    """Pyside equivalent for the loadUiType function in PyQt.

    From the PyQt4 documentation:
        Load a Qt Designer .ui file and return a tuple of the generated form
        class and the Qt base class. These can then be used to create any
        number of instances of the user interface without having to parse the
        .ui file more than once.

    Note:
        Pyside lacks the "loadUiType" command, so we have to convert the ui
        file to py code in-memory first and then execute it in a special frame
        to retrieve the form_class.

    Args:
        uifile (str): Absolute path to .ui file


    Returns:
        tuple: the generated form class, the Qt base class
    """
    import pysideuic
    import xml.etree.ElementTree as ElementTree
    from cStringIO import StringIO

    parsed = ElementTree.parse(uifile)
    widget_class = parsed.find('widget').get('class')
    form_class = parsed.find('class').text

    with open(uifile, 'r') as f:
        o = StringIO()
        frame = {}

        pysideuic.compileUi(f, o, indent=0)
        pyc = compile(o.getvalue(), '<string>', 'exec')
        exec(pyc) in frame

        # Fetch the base_class and form class based on their type in
        # the xml from designer
        form_class = frame['Ui_%s' % form_class]
        base_class = eval('QtWidgets.%s' % widget_class)
    return form_class, base_class