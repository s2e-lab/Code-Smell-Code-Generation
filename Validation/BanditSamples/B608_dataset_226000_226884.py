def __remove_index(self, ids):
        """remove affected ids from the index"""
        if not ids:
            return

        ids = ",".join((str(id) for id in ids))
        self.execute("DELETE FROM fact_index where id in (%s)" % ids)