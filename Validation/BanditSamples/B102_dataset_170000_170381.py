def bokeh_palette(name, rawtext, text, lineno, inliner, options=None, content=None):
    ''' Generate an inline visual representations of a single color palette.

    This function evaluates the expression ``"palette = %s" % text``, in the
    context of a ``globals`` namespace that has previously imported all of
    ``bokeh.plotting``. The resulting value for ``palette`` is used to
    construct a sequence of HTML ``<span>`` elements for each color.

    If evaluating the palette expression fails or does not produce a list or
    tuple of all strings, then a SphinxError is raised to terminate the build.

    For details on the arguments to this function, consult the Docutils docs:

    http://docutils.sourceforge.net/docs/howto/rst-roles.html#define-the-role-function

    '''
    try:
        exec("palette = %s" % text, _globals)
    except Exception as e:
        raise SphinxError("cannot evaluate palette expression '%r', reason: %s" % (text, e))
    p = _globals.get('palette', None)
    if not isinstance(p, (list, tuple)) or not all(isinstance(x, str) for x in p):
        raise SphinxError("palette expression '%r' generated invalid or no output: %s" % (text, p))
    w = 20 if len(p) < 15 else 10 if len(p) < 32 else 5 if len(p) < 64 else 2 if len(p) < 128 else 1
    html = PALETTE_DETAIL.render(palette=p, width=w)
    node = nodes.raw('', html, format="html")
    return [node], []