def plsda(df, a, b, n_components=2, mean_center=False, scale=True, **kwargs):
    """
    Partial Least Squares Discriminant Analysis, based on `sklearn.cross_decomposition.PLSRegression`

    Performs a binary group partial least squares discriminant analysis (PLS-DA) on the supplied
    dataframe, selecting the first ``n_components``.

    Sample groups are defined by the selectors ``a`` and ``b`` which are used to select columns
    from the supplied dataframe. The result model is applied to the entire dataset,
    projecting non-selected samples into the same space.

    For more information on PLS regression and the algorithm used, see the `scikit-learn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html>`_.

    :param df: Pandas ``DataFrame`` to perform the analysis on
    :param a: Column selector for group a
    :param b: Column selector for group b
    :param n_components: ``int`` number of components to select
    :param mean_center: ``bool`` mean center the data before performing PLS regression
    :param kwargs: additional keyword arguments to `sklearn.cross_decomposition.PLSRegression`
    :return: scores ``DataFrame`` of PLSDA scores n_components x n_samples
             weights ``DataFrame`` of PLSDA weights n_variables x n_components
    """

    if not sklearn:
        assert('This library depends on scikit-learn (sklearn) to perform PLS-DA')

    from sklearn.cross_decomposition import PLSRegression

    df = df.copy()

    # We have to zero fill, nan errors in PLSRegression
    df[ np.isnan(df) ] = 0

    if mean_center:
        mean = np.mean(df.values, axis=0)
        df = df - mean

    sxa, _ = df.columns.get_loc_level(a)
    sxb, _ = df.columns.get_loc_level(b)

    dfa = df.iloc[:, sxa]
    dfb = df.iloc[:, sxb]

    dff = pd.concat([dfa, dfb], axis=1)
    y = np.ones(dff.shape[1])
    y[np.arange(dfa.shape[1])] = 0

    plsr = PLSRegression(n_components=n_components, scale=scale, **kwargs)
    plsr.fit(dff.values.T, y)

    # Apply the generated model to the original data
    x_scores = plsr.transform(df.values.T)

    scores = pd.DataFrame(x_scores.T)
    scores.index = ['Latent Variable %d' % (n+1) for n in range(0, scores.shape[0])]
    scores.columns = df.columns

    weights = pd.DataFrame(plsr.x_weights_)
    weights.index = df.index
    weights.columns = ['Weights on Latent Variable %d' % (n+1) for n in range(0, weights.shape[1])]

    loadings = pd.DataFrame(plsr.x_loadings_)
    loadings.index = df.index
    loadings.columns = ['Loadings on Latent Variable %d' % (n+1) for n in range(0, loadings.shape[1])]

    return scores, weights, loadings