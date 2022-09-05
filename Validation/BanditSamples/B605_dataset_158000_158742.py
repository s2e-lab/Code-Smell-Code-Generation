def display_svg(kw, fname):  # pragma: nocover
    """Try to display the svg file on this platform.
    """
    if kw['display'] is None:
        cli.verbose("Displaying:", fname)
        if sys.platform == 'win32':
            os.startfile(fname)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, fname])
    else:
        cli.verbose(kw['display'] + " " + fname)
        os.system(kw['display'] + " " + fname)