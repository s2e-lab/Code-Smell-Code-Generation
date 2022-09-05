def format_decimal(interval, value):
    """Return formatted decimal according to interval decimal place

    For example:
    interval = 0.33 (two decimal places)
    my_float = 1.1215454
    Return 1.12 (return only two decimal places as string)
    If interval is an integer return integer part of my_number
    If my_number is an integer return as is
    """
    interval = get_significant_decimal(interval)
    if isinstance(interval, Integral) or isinstance(value, Integral):
        return add_separators(int(value))
    if interval != interval:
        # nan
        return str(value)
    if value != value:
        # nan
        return str(value)
    decimal_places = len(str(interval).split('.')[1])
    my_number_int = str(value).split('.')[0]
    my_number_decimal = str(value).split('.')[1][:decimal_places]
    if len(set(my_number_decimal)) == 1 and my_number_decimal[-1] == '0':
        return my_number_int
    formatted_decimal = (add_separators(int(my_number_int))
                         + decimal_separator()
                         + my_number_decimal)
    return formatted_decimal