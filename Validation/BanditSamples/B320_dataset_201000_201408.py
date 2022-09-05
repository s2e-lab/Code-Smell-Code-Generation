def parse_doctype(cls, file, encoding=None):
        '''Get the doctype from the document.

        Returns:
            str, None
        '''
        if encoding:
            lxml_encoding = to_lxml_encoding(encoding) or 'latin1'
        else:
            lxml_encoding = encoding

        try:
            parser = lxml.etree.XMLParser(encoding=lxml_encoding, recover=True)
            tree = lxml.etree.parse(
                io.BytesIO(wpull.util.peek_file(file)), parser=parser
            )
            if tree.getroot() is not None:
                return tree.docinfo.doctype
        except lxml.etree.LxmlError:
            pass