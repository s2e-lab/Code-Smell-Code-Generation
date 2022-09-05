def __score_method(X, y, fcounts, model_generator, score_function, method_name, nreps=10, test_size=100, cache_dir="/tmp"):
    """ Test an explanation method.
    """

    old_seed = np.random.seed()
    np.random.seed(3293)

    # average the method scores over several train/test splits
    method_reps = []

    data_hash = hashlib.sha256(__toarray(X).flatten()).hexdigest() + hashlib.sha256(__toarray(y)).hexdigest()
    for i in range(nreps):
        X_train, X_test, y_train, y_test = train_test_split(__toarray(X), y, test_size=test_size, random_state=i)

        # define the model we are going to explain, caching so we onlu build it once
        model_id = "model_cache__v" + "__".join([__version__, data_hash, model_generator.__name__])+".pickle"
        cache_file = os.path.join(cache_dir, model_id + ".pickle")
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                model = pickle.load(f)
        else:
            model = model_generator()
            model.fit(X_train, y_train)
            with open(cache_file, "wb") as f:
                pickle.dump(model, f)

        attr_key = "_".join([model_generator.__name__, method_name, str(test_size), str(nreps), str(i), data_hash])
        def score(attr_function):
            def cached_attr_function(X_inner):
                if attr_key not in _attribution_cache:
                    _attribution_cache[attr_key] = attr_function(X_inner)
                return _attribution_cache[attr_key]

            #cached_attr_function = lambda X: __check_cache(attr_function, X)
            if fcounts is None:
                return score_function(X_train, X_test, y_train, y_test, cached_attr_function, model, i)
            else:
                scores = []
                for f in fcounts:
                    scores.append(score_function(f, X_train, X_test, y_train, y_test, cached_attr_function, model, i))
                return np.array(scores)

        # evaluate the method (only building the attribution function if we need to)
        if attr_key not in _attribution_cache:
            method_reps.append(score(getattr(methods, method_name)(model, X_train)))
        else:
            method_reps.append(score(None))

    np.random.seed(old_seed)
    return np.array(method_reps).mean(0)