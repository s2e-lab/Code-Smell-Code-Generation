def Verify(self):
        """
        Verify block using the verification script.

        Returns:
            bool: True if valid. False otherwise.
        """
        if not self.Hash.ToBytes() == GetGenesis().Hash.ToBytes():
            return False

        bc = GetBlockchain()

        if not bc.ContainsBlock(self.Index):
            return False

        if self.Index > 0:
            prev_header = GetBlockchain().GetHeader(self.PrevHash.ToBytes())

            if prev_header is None:
                return False

            if prev_header.Index + 1 != self.Index:
                return False

            if prev_header.Timestamp >= self.Timestamp:
                return False

        # this should be done to actually verify the block
        if not Helper.VerifyScripts(self):
            return False

        return True