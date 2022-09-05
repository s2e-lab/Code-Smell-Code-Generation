def set_bytes_at_rva(self, rva, data):
        """Overwrite, with the given string, the bytes at the file offset corresponding to the given RVA.
        
        Return True if successful, False otherwise. It can fail if the
        offset is outside the file's boundaries.
        """
        
        offset = self.get_physical_by_rva(rva)
        if not offset:
            raise False
        
        return self.set_bytes_at_offset(offset, data)