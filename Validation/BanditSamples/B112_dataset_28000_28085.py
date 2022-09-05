async def _close(self):
        """
        Removes any present internal state from the client.
        """

        # Remove the core NATS Streaming subscriptions.
        try:
            if self._hb_inbox_sid is not None:
                await self._nc.unsubscribe(self._hb_inbox_sid)
                self._hb_inbox = None
                self._hb_inbox_sid = None
            if self._ack_subject_sid is not None:
                await self._nc.unsubscribe(self._ack_subject_sid)
                self._ack_subject = None
                self._ack_subject_sid = None
        except:
            # FIXME: async error in case these fail?
            pass

        # Remove all the related subscriptions
        for _, sub in self._sub_map.items():
            if sub._msgs_task is not None:
                sub._msgs_task.cancel()
            try:
                await self._nc.unsubscribe(sub.sid)
            except:
                continue
        self._sub_map = {}