def eval_cached(self, statement, *args):
        """
        Evaluate a statement and cache the result before returning.

        Statements are evaluated inside the Trimesh object, and

        Parameters
        -----------
        statement : str
          Statement of valid python code
        *args : list
          Available inside statement as args[0], etc

        Returns
        -----------
        result : result of running eval on statement with args

        Examples
        -----------
        r = mesh.eval_cached('np.dot(self.vertices, args[0])', [0,0,1])
        """

        statement = str(statement)
        key = 'eval_cached_' + statement
        key += '_'.join(str(i) for i in args)

        if key in self._cache:
            return self._cache[key]

        result = eval(statement)
        self._cache[key] = result
        return result