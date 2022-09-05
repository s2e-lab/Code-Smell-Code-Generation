def call(poly, args):
    """
    Evaluate a polynomial along specified axes.

    Args:
        poly (Poly):
            Input polynomial.
        args (numpy.ndarray):
            Argument to be evaluated. Masked values keeps the variable intact.

    Returns:
        (Poly, numpy.ndarray):
            If masked values are used the Poly is returned. Else an numpy array
            matching the polynomial's shape is returned.
    """
    args = list(args)

    # expand args to match dim
    if len(args) < poly.dim:
        args = args + [np.nan]*(poly.dim-len(args))

    elif len(args) > poly.dim:
        raise ValueError("too many arguments")

    # Find and perform substitutions, if any
    x0, x1 = [], []
    for idx, arg in enumerate(args):

        if isinstance(arg, Poly):
            poly_ = Poly({
                tuple(np.eye(poly.dim)[idx]): np.array(1)
            })
            x0.append(poly_)
            x1.append(arg)
            args[idx] = np.nan
    if x0:
        poly = call(poly, args)
        return substitute(poly, x0, x1)

    # Create masks
    masks = np.zeros(len(args), dtype=bool)
    for idx, arg in enumerate(args):
        if np.ma.is_masked(arg) or np.any(np.isnan(arg)):
            masks[idx] = True
            args[idx] = 0

    shape = np.array(
        args[
            np.argmax(
                [np.prod(np.array(arg).shape) for arg in args]
            )
        ]
    ).shape
    args = np.array([np.ones(shape, dtype=int)*arg for arg in args])

    A = {}
    for key in poly.keys:

        key_ = np.array(key)*(1-masks)
        val = np.outer(poly.A[key], np.prod((args.T**key_).T, \
                axis=0))
        val = np.reshape(val, poly.shape + tuple(shape))
        val = np.where(val != val, 0, val)

        mkey = tuple(np.array(key)*(masks))
        if not mkey in A:
            A[mkey] = val
        else:
            A[mkey] = A[mkey] + val

    out = Poly(A, poly.dim, None, None)
    if out.keys and not np.sum(out.keys):
        out = out.A[out.keys[0]]
    elif not out.keys:
        out = np.zeros(out.shape, dtype=out.dtype)
    return out