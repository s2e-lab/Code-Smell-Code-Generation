def load_mnist(training_num=50000):
    """Load mnist dataset"""
    data_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'mnist.npz')
    if not os.path.isfile(data_path):
        from six.moves import urllib
        origin = (
            'https://github.com/sxjscience/mxnet/raw/master/example/bayesian-methods/mnist.npz'
        )
        print('Downloading data from %s to %s' % (origin, data_path))
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(origin, context=ctx) as u, open(data_path, 'wb') as f:
            f.write(u.read())
        print('Done!')
    dat = numpy.load(data_path)
    X = (dat['X'][:training_num] / 126.0).astype('float32')
    Y = dat['Y'][:training_num]
    X_test = (dat['X_test'] / 126.0).astype('float32')
    Y_test = dat['Y_test']
    Y = Y.reshape((Y.shape[0],))
    Y_test = Y_test.reshape((Y_test.shape[0],))
    return X, Y, X_test, Y_test