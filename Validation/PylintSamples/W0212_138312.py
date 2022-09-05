def add_resolved_links(store, drop_defaults):
    """Adds the state of any link models between two models in store"""
    for widget_id, widget in Widget.widgets.items(): # go over all widgets
        if isinstance(widget, Link) and widget_id not in store:
            if widget.source[0].model_id in store and widget.target[0].model_id in store:
                store[widget.model_id] = widget._get_embed_state(drop_defaults=drop_defaults)