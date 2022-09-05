def const_factory(value):
    """return an astroid node for a python value"""
    # XXX we should probably be stricter here and only consider stuff in
    # CONST_CLS or do better treatment: in case where value is not in CONST_CLS,
    # we should rather recall the builder on this value than returning an empty
    # node (another option being that const_factory shouldn't be called with something
    # not in CONST_CLS)
    assert not isinstance(value, NodeNG)

    # Hack for ignoring elements of a sequence
    # or a mapping, in order to avoid transforming
    # each element to an AST. This is fixed in 2.0
    # and this approach is a temporary hack.
    if isinstance(value, (list, set, tuple, dict)):
        elts = []
    else:
        elts = value

    try:
        initializer_cls = CONST_CLS[value.__class__]
        initializer = _CONST_CLS_CONSTRUCTORS[initializer_cls]
        return initializer(initializer_cls, elts)
    except (KeyError, AttributeError):
        node = EmptyNode()
        node.object = value
        return node