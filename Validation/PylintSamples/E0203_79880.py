def register(self):
        """Register spec codec"""
        # Assume utf8 encoding
        utf8 = encodings.search_function('utf8')

        class StreamReader(utf_8.StreamReader):
            """Used by cPython to deal with a spec file"""
            def __init__(sr, stream, *args, **kwargs):
                codecs.StreamReader.__init__(sr, stream, *args, **kwargs)
                data = self.dealwith(sr.stream.readline)
                sr.stream = StringIO(data)

        def decode(text, *args, **kwargs):
            """Used by pypy and pylint to deal with a spec file"""
            return_tuple = kwargs.get("return_tuple", True)

            if six.PY3:
                if hasattr(text, 'tobytes'):
                    text = text.tobytes().decode('utf8')
                else:
                    text = text.decode('utf8')

            buffered = StringIO(text)

            # Determine if we need to have imports for this string
            # It may be a fragment of the file
            has_spec = regexes['encoding_matcher'].search(buffered.readline())
            no_imports = not has_spec
            buffered.seek(0)

            # Translate the text
            if six.PY2:
                utf8 = encodings.search_function('utf8') # Assume utf8 encoding
                reader = utf8.streamreader(buffered)
            else:
                reader = buffered

            data = self.dealwith(reader.readline, no_imports=no_imports)

            # If nothing was changed, then we want to use the original file/line
            # Also have to replace indentation of original line with indentation of new line
            # To take into account nested describes
            if text and not regexes['only_whitespace'].match(text):
                if regexes['whitespace'].sub('', text) == regexes['whitespace'].sub('', data):
                    bad_indentation = regexes['leading_whitespace'].search(text).groups()[0]
                    good_indentation = regexes['leading_whitespace'].search(data).groups()[0]
                    data = '%s%s' % (good_indentation, text[len(bad_indentation):])

            # If text is empty and data isn't, then we should return text
            if len(text) == 0 and len(data) == 1:
                if return_tuple:
                    return "", 0
                else:
                    return ""

            # Return translated version and it's length
            if return_tuple:
                return data, len(data)
            else:
                return data

        incrementaldecoder = utf8.incrementaldecoder
        if six.PY3:
            def incremental_decode(decoder, *args, **kwargs):
                """Wrapper for decode from IncrementalDecoder"""
                kwargs["return_tuple"] = False
                return decode(*args, **kwargs)
            incrementaldecoder = type("IncrementalDecoder", (utf8.incrementaldecoder, ), {"decode": incremental_decode})

        def search_function(s):
            """Determine if a file is of spec encoding and return special CodecInfo if it is"""
            if s != 'spec': return None
            return codecs.CodecInfo(
                  name='spec'
                , encode=utf8.encode
                , decode=decode
                , streamreader=StreamReader
                , streamwriter=utf8.streamwriter
                , incrementalencoder=utf8.incrementalencoder
                , incrementaldecoder=incrementaldecoder
                )

        # Do the register
        codecs.register(search_function)