def reconnect(self):
		"""Reconnect to the remote server."""
		self.lock.acquire()
		if self.use_ssl:
			self.client = http.client.HTTPSConnection(self.host, self.port, context=self.ssl_context)
		else:
			self.client = http.client.HTTPConnection(self.host, self.port)
		self.lock.release()