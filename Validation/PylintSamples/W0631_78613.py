def release(self, lane, status, target=None, meta=None, svcs=None):
        """Set release information on a build"""

        if target not in (None, 'current', 'future'):
            raise ValueError("\nError: Target must be None, 'current', or 'future'\n")

        svcs, meta, lane = self._prep_for_release(lane, svcs=svcs, meta=meta)
        when = time.time()

        # loathe non-functional dictionaries in python
        rel_data = meta.copy()
        rel_data.update({
            "_time": when,
            "status": status,
            "services": list(svcs.keys()),
        })
        rel_lane = self.obj.get('lanes', {}).get(lane, dict(log=[],status=status))
        rel_lane['status'] = status
        rel_lane['log'] = [rel_data] + rel_lane.get('log', [])

        self.rcs.patch('build', self.name, {
            "lanes": {
                lane: rel_lane,
            }
        })

        if target:
            for svc in svcs:
                rel_data = {target: self.name}

            # if target is specified, then also update svc.release
            #    {current/previous/future}
            if target == "current":
                mysvc = svcs[svc]
                curver = mysvc.get('release', {}).get('current', '')
                prev = []
                if curver:
                    prev = mysvc.get('release', {}).get('previous', [])
                    if not prev or prev[0] != curver:
                        prev = [curver] + prev
                    while len(prev) > 5: # magic values FTW
                        prev.pop() # only keep history of 5 previous
                rel_data['previous'] = prev

            self.rcs.patch('service', svc, {
                "release": rel_data,
                "statuses": {status: when},
                "status": status
            })