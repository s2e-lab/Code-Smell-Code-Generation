def pca(df, n_components=2, mean_center=False, **kwargs):
    """
    Principal Component Analysis, based on `sklearn.decomposition.PCA`

    Performs a principal component analysis (PCA) on the supplied dataframe, selecting the first ``n_components`` components
    in the resulting model. The model scores and weights are returned.

    For more information on PCA and the algorithm used, see the `scikit-learn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_.

    :param df: Pandas ``DataFrame`` to perform the analysis on
    :param n_components: ``int`` number of components to select
    :param mean_center: ``bool`` mean center the data before performing PCA
    :param kwargs: additional keyword arguments to `sklearn.decomposition.PCA`
    :return: scores ``DataFrame`` of PCA scores n_components x n_samples
             weights ``DataFrame`` of PCA weights n_variables x n_components
    """

    if not sklearn:
        assert('This library depends on scikit-learn (sklearn) to perform PCA analysis')
        
    from sklearn.decomposition import PCA

    df = df.copy()
    
    # We have to zero fill, nan errors in PCA
    df[ np.isnan(df) ] = 0

    if mean_center:
        mean = np.mean(df.values, axis=0)
        df = df - mean

    pca = PCA(n_components=n_components, **kwargs)
    pca.fit(df.values.T)

    scores = pd.DataFrame(pca.transform(df.values.T)).T
    scores.index = ['Principal Component %d (%.2f%%)' % ( (n+1), pca.explained_variance_ratio_[n]*100 ) for n in range(0, scores.shape[0])]
    scores.columns = df.columns

    weights = pd.DataFrame(pca.components_).T
    weights.index = df.index
    weights.columns = ['Weights on Principal Component %d' % (n+1) for n in range(0, weights.shape[1])]
       
    return scores, weights