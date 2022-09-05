def _compute_faulting_style_term(Frss, pR, Fnss, pN, rake):
    """
    Compute SHARE faulting style adjustment term.
    """
    if rake > 30.0 and rake <= 150.0:
        return np.power(Frss, 1 - pR) * np.power(Fnss, -pN)
    elif rake > -120.0 and rake <= -60.0:
        return np.power(Frss, - pR) * np.power(Fnss, 1 - pN)
    else:
        return np.power(Frss, - pR) * np.power(Fnss, - pN)