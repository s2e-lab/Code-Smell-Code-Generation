def reflections(depth, e_zH, Gam, lrec, lsrc, use_ne_eval):
    r"""Calculate Rp, Rm.

    .. math:: R^\pm_n, \bar{R}^\pm_n

    This function corresponds to equations 64/65 and A-11/A-12 in
    [HuTS15]_, and loosely to the corresponding files ``Rmin.F90`` and
    ``Rplus.F90``.

    This function is called from the function :mod:`kernel.greenfct`.

    """

    # Loop over Rp, Rm
    for plus in [True, False]:

        # Switches depending if plus or minus
        if plus:
            pm = 1
            layer_count = np.arange(depth.size-2, min(lrec, lsrc)-1, -1)
            izout = abs(lsrc-lrec)
            minmax = max(lrec, lsrc)
        else:
            pm = -1
            layer_count = np.arange(1, max(lrec, lsrc)+1, 1)
            izout = 0
            minmax = -min(lrec, lsrc)

        # If rec in last  and rec below src (plus) or
        # if rec in first and rec above src (minus), shift izout
        shiftplus = lrec < lsrc and lrec == 0 and not plus
        shiftminus = lrec > lsrc and lrec == depth.size-1 and plus
        if shiftplus or shiftminus:
            izout -= pm

        # Pre-allocate Ref
        Ref = np.zeros((Gam.shape[0], Gam.shape[1], abs(lsrc-lrec)+1,
                        Gam.shape[3]), dtype=complex)

        # Calculate the reflection
        for iz in layer_count:

            # Eqs 65, A-12
            e_zHa = e_zH[:, None, iz+pm, None]
            Gama = Gam[:, :, iz, :]
            e_zHb = e_zH[:, None, iz, None]
            Gamb = Gam[:, :, iz+pm, :]
            if use_ne_eval:
                rlocstr = "(e_zHa*Gama - e_zHb*Gamb)/(e_zHa*Gama + e_zHb*Gamb)"
                rloc = use_ne_eval(rlocstr)
            else:
                rloca = e_zHa*Gama
                rlocb = e_zHb*Gamb
                rloc = (rloca - rlocb)/(rloca + rlocb)

            # In first layer tRef = rloc
            if iz == layer_count[0]:
                tRef = rloc.copy()
            else:
                ddepth = depth[iz+1+pm]-depth[iz+pm]

                # Eqs 64, A-11
                if use_ne_eval:
                    term = use_ne_eval("tRef*exp(-2*Gamb*ddepth)")
                    tRef = use_ne_eval("(rloc + term)/(1 + rloc*term)")
                else:
                    term = tRef*np.exp(-2*Gamb*ddepth)  # NOQA
                    tRef = (rloc + term)/(1 + rloc*term)

            # The global reflection coefficient is given back for all layers
            # between and including src- and rec-layer
            if lrec != lsrc and pm*iz <= minmax:
                Ref[:, :, izout, :] = tRef[:]
                izout -= pm

        # If lsrc = lrec, we just store the last values
        if lsrc == lrec and layer_count.size > 0:
            Ref = tRef

        # Store Ref in Rm/Rp
        if plus:
            Rm = Ref
        else:
            Rp = Ref

    # Return reflections (minus and plus)
    return Rm, Rp