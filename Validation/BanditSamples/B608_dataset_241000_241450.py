def save(self, session):
        """Saves a requests.Session object for the next heartbeat process.
        """

        if not HAS_SQL:  # pragma: nocover
            return
        try:
            conn, c = self.connect()
            c.execute('DELETE FROM {0}'.format(self.table_name))
            values = {
                'value': sqlite3.Binary(pickle.dumps(session, protocol=2)),
            }
            c.execute('INSERT INTO {0} VALUES (:value)'.format(self.table_name), values)
            conn.commit()
            conn.close()
        except:  # pragma: nocover
            log.traceback(logging.DEBUG)