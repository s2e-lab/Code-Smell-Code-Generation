def getView(self, lv):
        """Determine the detector view starting with a G4LogicalVolume"""
        view = None
        if str(lv.GetName())[-1] == 'X':
            return 'X'
        elif str(lv.GetName())[-1] == 'Y':
            return 'Y'

        self.log.error('Cannot determine view for %s', lv.GetName())
        raise 'Cannot determine view for %s' % lv.GetName()
        return view