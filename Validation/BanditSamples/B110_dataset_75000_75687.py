def regenerate(self):
        """Regenerate the session id.

        This function creates a new session id and stores all information
        associated with the current id in that new id. It then destroys the
        old session id. This is useful for preventing session fixation attacks
        and should be done whenever someone uses a login to obtain additional
        authorizaiton.
        """

        oldhash = self.session_hash
        self.new_session_id()
        try:
            self.rdb.rename(oldhash,self.session_hash)
            self.rdb.expire(self.session_hash,self.ttl)
        except:
            pass