def _read(self, fp, fpname):
        """A direct copy of the py2.4 version of the super class's _read method
        to assure it uses ordered dicts. Had to change one line to make it work.

        Future versions have this fixed, but in fact its quite embarrassing for the
        guys not to have done it right in the first place !

        Removed big comments to make it more compact.

        Made sure it ignores initial whitespace as git uses tabs"""
        cursect = None                            # None, or a dictionary
        optname = None
        lineno = 0
        is_multi_line = False
        e = None                                  # None, or an exception

        def string_decode(v):
            if v[-1] == '\\':
                v = v[:-1]
            # end cut trailing escapes to prevent decode error

            if PY3:
                return v.encode(defenc).decode('unicode_escape')
            else:
                return v.decode('string_escape')
            # end
        # end

        while True:
            # we assume to read binary !
            line = fp.readline().decode(defenc)
            if not line:
                break
            lineno = lineno + 1
            # comment or blank line?
            if line.strip() == '' or self.re_comment.match(line):
                continue
            if line.split(None, 1)[0].lower() == 'rem' and line[0] in "rR":
                # no leading whitespace
                continue

            # is it a section header?
            mo = self.SECTCRE.match(line.strip())
            if not is_multi_line and mo:
                sectname = mo.group('header').strip()
                if sectname in self._sections:
                    cursect = self._sections[sectname]
                elif sectname == cp.DEFAULTSECT:
                    cursect = self._defaults
                else:
                    cursect = self._dict((('__name__', sectname),))
                    self._sections[sectname] = cursect
                    self._proxies[sectname] = None
                # So sections can't start with a continuation line
                optname = None
            # no section header in the file?
            elif cursect is None:
                raise cp.MissingSectionHeaderError(fpname, lineno, line)
            # an option line?
            elif not is_multi_line:
                mo = self.OPTCRE.match(line)
                if mo:
                    # We might just have handled the last line, which could contain a quotation we want to remove
                    optname, vi, optval = mo.group('option', 'vi', 'value')
                    if vi in ('=', ':') and ';' in optval and not optval.strip().startswith('"'):
                        pos = optval.find(';')
                        if pos != -1 and optval[pos - 1].isspace():
                            optval = optval[:pos]
                    optval = optval.strip()
                    if optval == '""':
                        optval = ''
                    # end handle empty string
                    optname = self.optionxform(optname.rstrip())
                    if len(optval) > 1 and optval[0] == '"' and optval[-1] != '"':
                        is_multi_line = True
                        optval = string_decode(optval[1:])
                    # end handle multi-line
                    cursect[optname] = optval
                else:
                    # check if it's an option with no value - it's just ignored by git
                    if not self.OPTVALUEONLY.match(line):
                        if not e:
                            e = cp.ParsingError(fpname)
                        e.append(lineno, repr(line))
                    continue
            else:
                line = line.rstrip()
                if line.endswith('"'):
                    is_multi_line = False
                    line = line[:-1]
                # end handle quotations
                cursect[optname] += string_decode(line)
            # END parse section or option
        # END while reading

        # if any parsing errors occurred, raise an exception
        if e:
            raise e