def if_else(condition, when_true, otherwise):
    """
    Wraps creation of a series based on if-else conditional logic into a function
    call.

    Provide a boolean vector condition, value(s) when true, and value(s)
    when false, and a vector will be returned the same length as the conditional
    vector according to the logical statement.

    Args:
        condition: A boolean vector representing the condition. This is often
            a logical statement with a symbolic series.
        when_true: A vector the same length as the condition vector or a single
            value to apply when the condition is `True`.
        otherwise: A vector the same length as the condition vector or a single
            value to apply when the condition is `False`.

    Example:
    df = pd.DataFrame
    """

    if not isinstance(when_true, collections.Iterable) or isinstance(when_true, str):
        when_true = np.repeat(when_true, len(condition))
    if not isinstance(otherwise, collections.Iterable) or isinstance(otherwise, str):
        otherwise = np.repeat(otherwise, len(condition))
    assert (len(condition) == len(when_true)) and (len(condition) == len(otherwise))

    if isinstance(when_true, pd.Series):
        when_true = when_true.values
    if isinstance(otherwise, pd.Series):
        otherwise = otherwise.values

    output = np.array([when_true[i] if c else otherwise[i]
                       for i, c in enumerate(condition)])
    return output