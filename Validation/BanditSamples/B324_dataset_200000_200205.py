def hash(self):
        """Return md5 hash for current dataset."""
        if self._hash is None:
            m = hashlib.new('md5')
            if self._preprocessor is None:
                # generate hash from numpy array
                m.update(numpy_buffer(self._X_train))
                m.update(numpy_buffer(self._y_train))
                if self._X_test is not None:
                    m.update(numpy_buffer(self._X_test))
                if self._y_test is not None:
                    m.update(numpy_buffer(self._y_test))
            elif callable(self._preprocessor):
                # generate hash from user defined object (source code)
                m.update(inspect.getsource(self._preprocessor).encode('utf-8'))

            self._hash = m.hexdigest()

        return self._hash