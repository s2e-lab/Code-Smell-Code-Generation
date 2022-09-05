def _parse_index(self, v_name):
        """Parse Fortran vector indices into a tuple of Python indices."""
        i_start = i_end = i_stride = None

        # Start index
        self._update_tokens()
        try:
            i_start = int(self.token)
            self._update_tokens()
        except ValueError:
            if self.token in (',', ')'):
                raise ValueError('{0} index cannot be empty.'.format(v_name))
            elif not self.token == ':':
                raise

        # End index
        if self.token == ':':
            self._update_tokens()
            try:
                i_end = 1 + int(self.token)
                self._update_tokens()
            except ValueError:
                if self.token == ':':
                    raise ValueError('{0} end index cannot be implicit '
                                     'when using stride.'.format(v_name))
                elif self.token not in (',', ')'):
                    raise
        elif self.token in (',', ')'):
            # Replace index with single-index range
            if i_start:
                i_end = 1 + i_start

        # Stride index
        if self.token == ':':
            self._update_tokens()
            try:
                i_stride = int(self.token)
            except ValueError:
                if self.token == ')':
                    raise ValueError('{0} stride index cannot be '
                                     'implicit.'.format(v_name))
                else:
                    raise

            if i_stride == 0:
                raise ValueError('{0} stride index cannot be zero.'
                                 ''.format(v_name))

            self._update_tokens()

        if self.token not in (',', ')'):
            raise ValueError('{0} index did not terminate '
                             'correctly.'.format(v_name))

        idx_triplet = (i_start, i_end, i_stride)
        return idx_triplet