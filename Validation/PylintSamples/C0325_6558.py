def Servers(self,cached=True):
		"""Returns list of server objects, populates if necessary.

		>>> clc.v2.Servers(["NY1BTDIPHYP0101","NY1BTDIWEB0101"]).Servers()
		[<clc.APIv2.server.Server object at 0x1065b0d50>, <clc.APIv2.server.Server object at 0x1065b0e50>]
		>>> print _[0]
		NY1BTDIPHYP0101

		"""

		if not hasattr(self,'_servers') or not cached:
			self._servers = []
			for server in self.servers_lst:
				self._servers.append(Server(id=server,alias=self.alias,session=self.session))

		return(self._servers)