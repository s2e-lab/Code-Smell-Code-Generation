def iter_options(grouped_choices, cutoff=None, cutoff_text=None):
    """
    Helper function for options and option groups in templates.
    """
    class StartOptionGroup(object):
        start_option_group = True
        end_option_group = False

        def __init__(self, label):
            self.label = label

    class EndOptionGroup(object):
        start_option_group = False
        end_option_group = True

    class Option(object):
        start_option_group = False
        end_option_group = False

        def __init__(self, value, display_text, disabled=False):
            self.value = value
            self.display_text = display_text
            self.disabled = disabled

    count = 0

    for key, value in grouped_choices.items():
        if cutoff and count >= cutoff:
            break

        if isinstance(value, dict):
            yield StartOptionGroup(label=key)
            for sub_key, sub_value in value.items():
                if cutoff and count >= cutoff:
                    break
                yield Option(value=sub_key, display_text=sub_value)
                count += 1
            yield EndOptionGroup()
        else:
            yield Option(value=key, display_text=value)
            count += 1

    if cutoff and count >= cutoff and cutoff_text:
        cutoff_text = cutoff_text.format(count=cutoff)
        yield Option(value='n/a', display_text=cutoff_text, disabled=True)