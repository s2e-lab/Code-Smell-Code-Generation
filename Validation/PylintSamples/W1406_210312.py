def convert(text, code=True, tabsize=4):
    """
    Convert text with ANSI escape sequences to HTML.

    :param text: The text with ANSI escape sequences (a string).
    :param code: Whether to wrap the returned HTML fragment in a
                 ``<code>...</code>`` element (a boolean, defaults
                 to :data:`True`).
    :param tabsize: Refer to :func:`str.expandtabs()` for details.
    :returns: The text converted to HTML (a string).
    """
    output = []
    in_span = False
    compatible_text_styles = {
        # The following ANSI text styles have an obvious mapping to CSS.
        ANSI_TEXT_STYLES['bold']: {'font-weight': 'bold'},
        ANSI_TEXT_STYLES['strike_through']: {'text-decoration': 'line-through'},
        ANSI_TEXT_STYLES['underline']: {'text-decoration': 'underline'},
    }
    for token in TOKEN_PATTERN.split(text):
        if token.startswith(('http://', 'https://', 'www.')):
            url = token if '://' in token else ('http://' + token)
            token = u'<a href="%s" style="color:inherit">%s</a>' % (html_encode(url), html_encode(token))
        elif token.startswith(ANSI_CSI):
            ansi_codes = token[len(ANSI_CSI):-1].split(';')
            if all(c.isdigit() for c in ansi_codes):
                ansi_codes = list(map(int, ansi_codes))
            # First we check for a reset code to close the previous <span>
            # element. As explained on Wikipedia [1] an absence of codes
            # implies a reset code as well: "No parameters at all in ESC[m acts
            # like a 0 reset code".
            # [1] https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_sequences
            if in_span and (0 in ansi_codes or not ansi_codes):
                output.append('</span>')
                in_span = False
            # Now we're ready to generate the next <span> element (if any) in
            # the knowledge that we're emitting opening <span> and closing
            # </span> tags in the correct order.
            styles = {}
            is_faint = (ANSI_TEXT_STYLES['faint'] in ansi_codes)
            is_inverse = (ANSI_TEXT_STYLES['inverse'] in ansi_codes)
            while ansi_codes:
                number = ansi_codes.pop(0)
                # Try to match a compatible text style.
                if number in compatible_text_styles:
                    styles.update(compatible_text_styles[number])
                    continue
                # Try to extract a text and/or background color.
                text_color = None
                background_color = None
                if 30 <= number <= 37:
                    # 30-37 sets the text color from the eight color palette.
                    text_color = EIGHT_COLOR_PALETTE[number - 30]
                elif 40 <= number <= 47:
                    # 40-47 sets the background color from the eight color palette.
                    background_color = EIGHT_COLOR_PALETTE[number - 40]
                elif 90 <= number <= 97:
                    # 90-97 sets the text color from the high-intensity eight color palette.
                    text_color = BRIGHT_COLOR_PALETTE[number - 90]
                elif 100 <= number <= 107:
                    # 100-107 sets the background color from the high-intensity eight color palette.
                    background_color = BRIGHT_COLOR_PALETTE[number - 100]
                elif number in (38, 39) and len(ansi_codes) >= 2 and ansi_codes[0] == 5:
                    # 38;5;N is a text color in the 256 color mode palette,
                    # 39;5;N is a background color in the 256 color mode palette.
                    try:
                        # Consume the 5 following 38 or 39.
                        ansi_codes.pop(0)
                        # Consume the 256 color mode color index.
                        color_index = ansi_codes.pop(0)
                        # Set the variable to the corresponding HTML/CSS color.
                        if number == 38:
                            text_color = EXTENDED_COLOR_PALETTE[color_index]
                        elif number == 39:
                            background_color = EXTENDED_COLOR_PALETTE[color_index]
                    except (ValueError, IndexError):
                        pass
                # Apply the 'faint' or 'inverse' text style
                # by manipulating the selected color(s).
                if text_color and is_inverse:
                    # Use the text color as the background color and pick a
                    # text color that will be visible on the resulting
                    # background color.
                    background_color = text_color
                    text_color = select_text_color(*parse_hex_color(text_color))
                if text_color and is_faint:
                    # Because I wasn't sure how to implement faint colors
                    # based on normal colors I looked at how gnome-terminal
                    # (my terminal of choice) handles this and it appears
                    # to just pick a somewhat darker color.
                    text_color = '#%02X%02X%02X' % tuple(
                        max(0, n - 40) for n in parse_hex_color(text_color)
                    )
                if text_color:
                    styles['color'] = text_color
                if background_color:
                    styles['background-color'] = background_color
            if styles:
                token = '<span style="%s">' % ';'.join(k + ':' + v for k, v in sorted(styles.items()))
                in_span = True
            else:
                token = ''
        else:
            token = html_encode(token)
        output.append(token)
    html = ''.join(output)
    html = encode_whitespace(html, tabsize)
    if code:
        html = '<code>%s</code>' % html
    return html