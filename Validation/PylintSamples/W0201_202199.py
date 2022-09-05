def clone(self):
        """
        Get a shallow clone of this object.
        The clone only shares the WSDL.  All other attributes are
        unique to the cloned object including options.
        @return: A shallow clone.
        @rtype: L{Client}
        """
        class Uninitialized(Client):
            def __init__(self):
                pass
        clone = Uninitialized()
        clone.options = Options()
        cp = Unskin(clone.options)
        mp = Unskin(self.options)
        cp.update(deepcopy(mp))
        clone.wsdl = self.wsdl
        clone.factory = self.factory
        clone.service = ServiceSelector(clone, self.wsdl.services)
        clone.sd = self.sd
        clone.messages = dict(tx=None, rx=None)
        return clone