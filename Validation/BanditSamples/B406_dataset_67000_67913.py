def from_xml(self, doc):
        """Load this domain based on an XML document"""
        import xml.sax
        handler = DomainDumpParser(self)
        xml.sax.parse(doc, handler)
        return handler