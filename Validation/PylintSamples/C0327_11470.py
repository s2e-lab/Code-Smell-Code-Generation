def refreshContents( self ):
        """
        Refreshes the contents tab with the latest selection from the browser.
        """
        item = self.uiContentsTREE.currentItem()
        if not isinstance(item, XdkEntryItem):
           return
        
        item.load()
        url = item.url()
        if url:
            self.gotoUrl(url)