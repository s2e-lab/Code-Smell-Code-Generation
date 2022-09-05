def train_punkt(ctx, input, output, abbr, colloc):
    """Train Punkt sentence splitter using sentences in input."""
    click.echo('chemdataextractor.tokenize.train_punkt')
    import pickle
    from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
    punkt = PunktTrainer()
    # Set these to true to include collocations more leniently, then increase MIN_COLLOC_FREQ to restrict again
    # punkt.INCLUDE_ALL_COLLOCS = False
    # punkt.INCLUDE_ABBREV_COLLOCS = False
    # punkt.MIN_COLLOC_FREQ = 1
    # Don't train on titles. They may contain abbreviations, but basically never have actual sentence boundaries.
    for fin in input:
        click.echo('Training on %s' % fin.name)
        sentences = fin.read()  #.replace('.\n', '. \n\n')
        punkt.train(sentences, finalize=False, verbose=True)
    punkt.finalize_training(verbose=True)
    if abbr:
        abbreviations = abbr.read().strip().split('\n')
        click.echo('Manually adding abbreviations: %s' % abbreviations)
        punkt._params.abbrev_types.update(abbreviations)
    if colloc:
        collocations = [tuple(l.split('. ', 1)) for l in colloc.read().strip().split('\n')]
        click.echo('Manually adding collocs: %s' % collocations)
        punkt._params.collocations.update(collocations)
    model = PunktSentenceTokenizer(punkt.get_params())
    pickle.dump(model, output, protocol=pickle.HIGHEST_PROTOCOL)