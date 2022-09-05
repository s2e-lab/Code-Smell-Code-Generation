def _draw_mark(mark_type, options={}, axes_options={}, **kwargs):
    """Draw the mark of specified mark type.

    Parameters
    ----------
    mark_type: type
        The type of mark to be drawn
    options: dict (default: {})
        Options for the scales to be created. If a scale labeled 'x' is
        required for that mark, options['x'] contains optional keyword
        arguments for the constructor of the corresponding scale type.
    axes_options: dict (default: {})
        Options for the axes to be created. If an axis labeled 'x' is required
        for that mark, axes_options['x'] contains optional keyword arguments
        for the constructor of the corresponding axis type.
    figure: Figure or None
        The figure to which the mark is to be added.
        If the value is None, the current figure is used.
    cmap: list or string
        List of css colors, or name of bqplot color scheme
    """
    fig = kwargs.pop('figure', current_figure())
    scales = kwargs.pop('scales', {})
    update_context = kwargs.pop('update_context', True)

    # Set the color map of the color scale
    cmap = kwargs.pop('cmap', None)
    if cmap is not None:
        # Add the colors or scheme to the color scale options
        options['color'] = dict(options.get('color', {}),
                                **_process_cmap(cmap))

    # Going through the list of data attributes
    for name in mark_type.class_trait_names(scaled=True):
        dimension = _get_attribute_dimension(name, mark_type)
        # TODO: the following should also happen if name in kwargs and
        # scales[name] is incompatible.
        if name not in kwargs:
            # The scaled attribute is not being passed to the mark. So no need
            # create a scale for this.
            continue
        elif name in scales:
            if update_context:
                _context['scales'][dimension] = scales[name]
        # Scale has to be fetched from the context or created as it has not
        # been passed.
        elif dimension not in _context['scales']:
            # Creating a scale for the dimension if a matching scale is not
            # present in _context['scales']
            traitlet = mark_type.class_traits()[name]
            rtype = traitlet.get_metadata('rtype')
            dtype = traitlet.validate(None, kwargs[name]).dtype
            # Fetching the first matching scale for the rtype and dtype of the
            # scaled attributes of the mark.
            compat_scale_types = [
                    Scale.scale_types[key]
                    for key in Scale.scale_types
                    if Scale.scale_types[key].rtype == rtype and
                    issubdtype(dtype, Scale.scale_types[key].dtype)
                ]
            sorted_scales = sorted(compat_scale_types,
                                   key=lambda x: x.precedence)
            scales[name] = sorted_scales[-1](**options.get(name, {}))
            # Adding the scale to the context scales
            if update_context:
                _context['scales'][dimension] = scales[name]
        else:
            scales[name] = _context['scales'][dimension]

    mark = mark_type(scales=scales, **kwargs)
    _context['last_mark'] = mark
    fig.marks = [m for m in fig.marks] + [mark]
    if kwargs.get('axes', True):
        axes(mark, options=axes_options)
    return mark