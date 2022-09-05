def get_filename(self):
        ''' Return the source filename of the current Stim. '''
        if self.filename is None or not os.path.exists(self.filename):
            tf = tempfile.mktemp() + self._default_file_extension
            self.save(tf)
            yield tf
            os.remove(tf)
        else:
            yield self.filename