def checkSystemVersion(s, versions=None):
    """
    Check if the current version is different from the previously recorded
    version.  If it is, or if there is no previously recorded version,
    create a version matching the current config.
    """

    if versions is None:
        versions = getSystemVersions()

    currentVersionMap = dict([(v.package, v) for v in versions])
    mostRecentSystemVersion = s.findFirst(SystemVersion,
                                          sort=SystemVersion.creation.descending)
    mostRecentVersionMap = dict([(v.package, v.asVersion()) for v in
                                 s.query(SoftwareVersion,
                                         (SoftwareVersion.systemVersion ==
                                          mostRecentSystemVersion))])

    if mostRecentVersionMap != currentVersionMap:
        currentSystemVersion = SystemVersion(store=s, creation=Time())
        for v in currentVersionMap.itervalues():
            makeSoftwareVersion(s, v, currentSystemVersion)