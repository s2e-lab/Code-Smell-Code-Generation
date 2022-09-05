def AstroDrizzle(input=None, mdriztab=False, editpars=False, configobj=None,
                 wcsmap=None, **input_dict):
    """ AstroDrizzle command-line interface """
    # Support input of filenames from command-line without a parameter name
    # then copy this into input_dict for merging with TEAL ConfigObj
    # parameters.

    # Load any user-specified configobj
    if isinstance(configobj, (str, bytes)):
        if configobj == 'defaults':
            # load "TEAL"-defaults (from ~/.teal/):
            configobj = teal.load(__taskname__)
        else:
            if not os.path.exists(configobj):
                raise RuntimeError('Cannot find .cfg file: '+configobj)
            configobj = teal.load(configobj, strict=False)
    elif configobj is None:
        # load 'astrodrizzle' parameter defaults as described in the docs:
        configobj = teal.load(__taskname__, defaults=True)

    if input and not util.is_blank(input):
        input_dict['input'] = input
    elif configobj is None:
        raise TypeError("AstroDrizzle() needs either 'input' or "
                        "'configobj' arguments")

    if 'updatewcs' in input_dict: # user trying to explicitly turn on updatewcs
        configobj['updatewcs'] = input_dict['updatewcs']
        del input_dict['updatewcs']

    # If called from interactive user-interface, configObj will not be
    # defined yet, so get defaults using EPAR/TEAL.
    #
    # Also insure that the input_dict (user-specified values) are folded in
    # with a fully populated configObj instance.
    try:
        configObj = util.getDefaultConfigObj(__taskname__, configobj,
                                             input_dict,
                                             loadOnly=(not editpars))
        log.debug('')
        log.debug("INPUT_DICT:")
        util.print_cfg(input_dict, log.debug)
        log.debug('')
        # If user specifies optional parameter for final_wcs specification in input_dict,
        #    insure that the final_wcs step gets turned on
        util.applyUserPars_steps(configObj, input_dict, step='3a')
        util.applyUserPars_steps(configObj, input_dict, step='7a')

    except ValueError:
        print("Problem with input parameters. Quitting...", file=sys.stderr)
        return

    if not configObj:
        return

    configObj['mdriztab'] = mdriztab
    # If 'editpars' was set to True, util.getDefaultConfigObj() will have
    # already called 'run()'.
    if not editpars:
        run(configObj, wcsmap=wcsmap)