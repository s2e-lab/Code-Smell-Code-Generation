def createMask(input=None, static_sig=4.0, group=None, editpars=False, configObj=None, **inputDict):
    """ The user can input a list of images if they like to create static masks
        as well as optional values for static_sig and inputDict.

        The configObj.cfg file will set the defaults and then override them
        with the user options.
    """

    if input is not None:
        inputDict["static_sig"]=static_sig
        inputDict["group"]=group
        inputDict["updatewcs"]=False
        inputDict["input"]=input
    else:
        print >> sys.stderr, "Please supply an input image\n"
        raise ValueError

    #this accounts for a user-called init where config is not defined yet
    configObj = util.getDefaultConfigObj(__taskname__,configObj,inputDict,loadOnly=(not editpars))
    if configObj is None:
        return

    if not editpars:
        run(configObj)