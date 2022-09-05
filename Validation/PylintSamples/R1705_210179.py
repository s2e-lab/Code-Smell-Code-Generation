def obfn_fvar(self):
        """Variable to be evaluated in computing regularisation term,
        depending on 'fEvalX' option value.
        """

        if self.opt['fEvalX']:
            return self.X
        else:
            return self.cnst_c() - self.cnst_B(self.Y)