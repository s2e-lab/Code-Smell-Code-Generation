def url2path(url):
	"""
	If url identifies a file on the local host, return the path to the
	file otherwise raise ValueError.
	"""
	scheme, host, path, nul, nul, nul = urlparse(url)
	if scheme.lower() in ("", "file") and host.lower() in ("", "localhost"):
		return path
	raise ValueError(url)