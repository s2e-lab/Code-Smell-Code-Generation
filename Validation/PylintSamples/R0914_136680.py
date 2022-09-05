def discretize(self, contact_id=0, accuracy=0.004, dt=0.001):
        """
        Sample this motion track into discretized motion events.

        Args:
            contact_id: contact point id
            accuracy: motion minimum difference in space
            dt: sample time difference
        """

        if not self.event_points:
            return []

        events = []
        action_dt = accuracy / self.speed
        dt = dt or action_dt

        ep0 = self.event_points[0]
        for _ in range(int(ep0[0] / dt)):
            events.append(['s', dt])
        events.append(['d', ep0[1], contact_id])
        for i, ep in enumerate(self.event_points[1:]):
            prev_ts = self.event_points[i][0]
            curr_ts = ep[0]
            p0 = self.event_points[i][1]
            p1 = ep[1]
            if p0 == p1:
                # hold
                for _ in range(int((curr_ts - prev_ts) / dt)):
                    events.append(['s', dt])
            else:
                # move
                dpoints = track_sampling([p0, p1], accuracy)
                for p in dpoints:
                    events.append(['m', p, contact_id])
                    for _ in range(int(action_dt / dt)):
                        events.append(['s', dt])

        events.append(['u', contact_id])
        return events