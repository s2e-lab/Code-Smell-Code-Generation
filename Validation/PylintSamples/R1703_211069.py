def _orientation_ok_to_bridge_contigs(self, start_hit, end_hit):
        '''Returns True iff the orientation of the hits means that the query contig of both hits can bridge the reference contigs of the hits'''
        assert start_hit.qry_name == end_hit.qry_name
        if start_hit.ref_name == end_hit.ref_name:
            return False

        if (
            (self._is_at_ref_end(start_hit) and start_hit.on_same_strand())
            or (self._is_at_ref_start(start_hit) and not start_hit.on_same_strand())
        ):
            start_hit_ok = True
        else:
            start_hit_ok = False

        if (
            (self._is_at_ref_start(end_hit) and end_hit.on_same_strand())
            or (self._is_at_ref_end(end_hit) and not end_hit.on_same_strand())
        ):
            end_hit_ok = True
        else:
            end_hit_ok = False

        return start_hit_ok and end_hit_ok