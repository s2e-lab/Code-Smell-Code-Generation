def augment_or_diminish_until_the_interval_is_right(note1, note2, interval):
    """A helper function for the minor and major functions.

    You should probably not use this directly.
    """
    cur = measure(note1, note2)
    while cur != interval:
        if cur > interval:
            note2 = notes.diminish(note2)
        elif cur < interval:
            note2 = notes.augment(note2)
        cur = measure(note1, note2)

    # We are practically done right now, but we need to be able to create the
    # minor seventh of Cb and get Bbb instead of B######### as the result
    val = 0
    for token in note2[1:]:
        if token == '#':
            val += 1
        elif token == 'b':
            val -= 1

    # These are some checks to see if we have generated too much #'s or too much
    # b's. In these cases we need to convert #'s to b's and vice versa.
    if val > 6:
        val = val % 12
        val = -12 + val
    elif val < -6:
        val = val % -12
        val = 12 + val

    # Rebuild the note
    result = note2[0]
    while val > 0:
        result = notes.augment(result)
        val -= 1
    while val < 0:
        result = notes.diminish(result)
        val += 1
    return result