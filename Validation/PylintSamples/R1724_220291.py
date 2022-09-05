def _compute_gas_price(probabilities, desired_probability):
    """
    Given a sorted range of ``Probability`` named-tuples returns a gas price
    computed based on where the ``desired_probability`` would fall within the
    range.

    :param probabilities: An iterable of `Probability` named-tuples sorted in reverse order.
    :param desired_probability: An floating point representation of the desired
        probability. (e.g. ``85% -> 0.85``)
    """
    first = probabilities[0]
    last = probabilities[-1]

    if desired_probability >= first.prob:
        return int(first.gas_price)
    elif desired_probability <= last.prob:
        return int(last.gas_price)

    for left, right in sliding_window(2, probabilities):
        if desired_probability < right.prob:
            continue
        elif desired_probability > left.prob:
            # This code block should never be reachable as it would indicate
            # that we already passed by the probability window in which our
            # `desired_probability` is located.
            raise Exception('Invariant')

        adj_prob = desired_probability - right.prob
        window_size = left.prob - right.prob
        position = adj_prob / window_size
        gas_window_size = left.gas_price - right.gas_price
        gas_price = int(math.ceil(right.gas_price + gas_window_size * position))
        return gas_price
    else:
        # The initial `if/else` clause in this function handles the case where
        # the `desired_probability` is either above or below the min/max
        # probability found in the `probabilities`.
        #
        # With these two cases handled, the only way this code block should be
        # reachable would be if the `probabilities` were not sorted correctly.
        # Otherwise, the `desired_probability` **must** fall between two of the
        # values in the `probabilities``.
        raise Exception('Invariant')