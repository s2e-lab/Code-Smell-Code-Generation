def getgrouploansurl(idgroup, *args, **kwargs):
    """Request Group loans URL.

    How to use it? By default MambuLoan uses getloansurl as the urlfunc.
    Override that behaviour by sending getgrouploansurl (this function)
    as the urlfunc to the constructor of MambuLoans (note the final 's')
    and voila! you get the Loans just for a certain group.

    If idgroup is set, you'll get a response adequate for a
    MambuLoans object.

    If not set, you'll get a Jar Jar Binks object, or something quite
    strange and useless as JarJar. A MambuError must likely since I
    haven't needed it for anything but for loans of one and just
    one group.

    See mambugroup module and pydoc for further information.

    Currently implemented filter parameters:
    * accountState

    See Mambu official developer documentation for further details, and
    info on parameters that may be implemented here in the future.
    """
    getparams = []
    if kwargs:
        try:
            if kwargs["fullDetails"] == True:
                getparams.append("fullDetails=true")
            else:
                getparams.append("fullDetails=false")
        except Exception as ex:
            pass
        try:
            getparams.append("accountState=%s" % kwargs["accountState"])
        except Exception as ex:
            pass
    groupidparam = "/" + idgroup
    url = getmambuurl(*args,**kwargs) + "groups" + groupidparam  + "/loans" + ( "" if len(getparams) == 0 else "?" + "&".join(getparams) )
    return url