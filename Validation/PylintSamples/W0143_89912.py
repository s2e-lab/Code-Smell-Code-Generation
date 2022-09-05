def _is_correct(text):
    """
    Check if the specified text has a correct spin syntax

    @type text: str
    @param text: Text written used spin syntax

    @rtype: tuple
    @return: A tuple: (is_correct, error). First position contains the result, and second one the error if not correct.
    """
    error = ''
    stack = []
    for i, c in enumerate(text):
        if c == char_opening:
            stack.append(c)
        elif c == char_closing:
            if stack.count == 0:
                error = 'Syntax incorrect. Found "}" before "{"'
                break
            last_char = stack.pop()
            if last_char != char_opening:
                error = 'Syntax incorrect. Found "}" before "{"'
                break
    if len(stack) > 0:
        error = 'Syntax incorrect. Some "{" were not closed'
    return not error, error