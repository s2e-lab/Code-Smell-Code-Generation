def raise_modal_error(verb_phrase_doc):
    """Given a verb phrase, raise an error if the modal auxilary has an issue
    with it"""
    verb_phrase = verb_phrase_doc.text.lower()
    bad_strings = ['should had', 'should has', 'could had', 'could has', 'would '
            'had', 'would has'] ["should", "could", "would"]
    for bs in bad_strings:
        if bs in verb_phrase:
            raise('ShouldCouldWouldError')