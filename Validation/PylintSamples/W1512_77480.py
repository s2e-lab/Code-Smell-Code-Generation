def check_compatibility(datasets, reqd_num_features=None):
    """
    Checks whether the given MLdataset instances are compatible

    i.e. with same set of subjects, each beloning to the same class in all instances.

    Checks the first dataset in the list against the rest, and returns a boolean array.

    Parameters
    ----------
    datasets : Iterable
        A list of n datasets

    reqd_num_features : int
        The required number of features in each dataset.
        Helpful to ensure test sets are compatible with training set,
            as well as within themselves.

    Returns
    -------
    all_are_compatible : bool
        Boolean flag indicating whether all datasets are compatible or not

    compatibility : list
        List indicating whether first dataset is compatible with the rest individually.
        This could be useful to select a subset of mutually compatible datasets.
        Length : n-1

    dim_mismatch : bool
        Boolean flag indicating mismatch in dimensionality from that specified

    size_descriptor : tuple
        A tuple with values for (num_samples, reqd_num_features)
        - num_samples must be common for all datasets that are evaluated for compatibility
        - reqd_num_features is None (when no check on dimensionality is perfomed), or
            list of corresponding dimensionalities for each input dataset

    """

    from collections import Iterable
    if not isinstance(datasets, Iterable):
        raise TypeError('Input must be an iterable '
                        'i.e. (list/tuple) of MLdataset/similar instances')

    datasets = list(datasets)  # to make it indexable if coming from a set
    num_datasets = len(datasets)

    check_dimensionality = False
    dim_mismatch = False
    if reqd_num_features is not None:
        if isinstance(reqd_num_features, Iterable):
            if len(reqd_num_features) != num_datasets:
                raise ValueError('Specify dimensionality for exactly {} datasets.'
                                 ' Given for a different number {}'
                                 ''.format(num_datasets, len(reqd_num_features)))
            reqd_num_features = list(map(int, reqd_num_features))
        else:  # same dimensionality for all
            reqd_num_features = [int(reqd_num_features)] * num_datasets

        check_dimensionality = True
    else:
        # to enable iteration
        reqd_num_features = [None,] * num_datasets

    pivot = datasets[0]
    if not isinstance(pivot, MLDataset):
        pivot = MLDataset(pivot)

    if check_dimensionality and pivot.num_features != reqd_num_features[0]:
        warnings.warn('Dimensionality mismatch! Expected {} whereas current {}.'
                      ''.format(reqd_num_features[0], pivot.num_features))
        dim_mismatch = True

    compatible = list()
    for ds, reqd_dim in zip(datasets[1:], reqd_num_features[1:]):
        if not isinstance(ds, MLDataset):
            ds = MLDataset(ds)

        is_compatible = True
        # compound bool will short-circuit, not optim required
        if pivot.num_samples != ds.num_samples \
                or pivot.keys != ds.keys \
                or pivot.classes != ds.classes:
            is_compatible = False

        if check_dimensionality and reqd_dim != ds.num_features:
            warnings.warn('Dimensionality mismatch! Expected {} whereas current {}.'
                          ''.format(reqd_dim, ds.num_features))
            dim_mismatch = True

        compatible.append(is_compatible)

    return all(compatible), compatible, dim_mismatch, \
           (pivot.num_samples, reqd_num_features)