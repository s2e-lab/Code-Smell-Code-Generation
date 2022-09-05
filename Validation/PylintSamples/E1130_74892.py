def log_loss(oracle, test_seq, ab=[], m_order=None, verbose=False):
    """ Evaluate the average log-loss of a sequence given an oracle """

    if not ab:
        ab = oracle.get_alphabet()
    if verbose:
        print(' ')

    logP = 0.0
    context = []
    increment = np.floor((len(test_seq) - 1) / 100)
    bar_count = -1
    maxContextLength = 0
    avgContext = 0
    for i, t in enumerate(test_seq):

        p, c = predict(oracle, context, ab, verbose=False)
        if len(c) < len(context):
            context = context[-len(c):]
        logP -= np.log2(p[ab[t]])
        context.append(t)

        if m_order is not None:
            if len(context) > m_order:
                context = context[-m_order:]
        avgContext += float(len(context)) / len(test_seq)

        if verbose:
            percentage = np.mod(i, increment)
            if percentage == 0:
                bar_count += 1
            if len(context) > maxContextLength:
                maxContextLength = len(context)
            sys.stdout.write('\r')
            sys.stdout.write("\r[" + "=" * bar_count +
                             " " * (100 - bar_count) + "] " +
                             str(bar_count) + "% " +
                             str(i) + "/" + str(len(test_seq) - 1) + " Current max length: " + str(
                maxContextLength))
            sys.stdout.flush()
    return logP / len(test_seq), avgContext