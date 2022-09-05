def parse(self, input_text, syncmap):
        """
        Read from SMIL file.

        Limitations:
        1. parses only ``<par>`` elements, in order
        2. timings must have ``hh:mm:ss.mmm`` or ``ss.mmm`` format (autodetected)
        3. both ``clipBegin`` and ``clipEnd`` attributes of ``<audio>`` must be populated
        """
        from lxml import etree
        smil_ns = "{http://www.w3.org/ns/SMIL}"
        root = etree.fromstring(gf.safe_bytes(input_text))
        for par in root.iter(smil_ns + "par"):
            for child in par:
                if child.tag == (smil_ns + "text"):
                    identifier = gf.safe_unicode(gf.split_url(child.get("src"))[1])
                elif child.tag == (smil_ns + "audio"):
                    begin_text = child.get("clipBegin")
                    if ":" in begin_text:
                        begin = gf.time_from_hhmmssmmm(begin_text)
                    else:
                        begin = gf.time_from_ssmmm(begin_text)
                    end_text = child.get("clipEnd")
                    if ":" in end_text:
                        end = gf.time_from_hhmmssmmm(end_text)
                    else:
                        end = gf.time_from_ssmmm(end_text)
            # TODO read text from additional text_file?
            self._add_fragment(
                syncmap=syncmap,
                identifier=identifier,
                lines=[u""],
                begin=begin,
                end=end
            )