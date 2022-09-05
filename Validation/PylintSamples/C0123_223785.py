def _wns_send(uri, data, wns_type="wns/toast", application_id=None):
	"""
	Sends a notification data and authentication to WNS.

	:param uri: str: The device's unique notification URI
	:param data: dict: The notification data to be sent.
	:return:
	"""
	access_token = _wns_authenticate(application_id=application_id)

	content_type = "text/xml"
	if wns_type == "wns/raw":
		content_type = "application/octet-stream"

	headers = {
		# content_type is "text/xml" (toast/badge/tile) | "application/octet-stream" (raw)
		"Content-Type": content_type,
		"Authorization": "Bearer %s" % (access_token),
		"X-WNS-Type": wns_type,  # wns/toast | wns/badge | wns/tile | wns/raw
	}

	if type(data) is str:
		data = data.encode("utf-8")

	request = Request(uri, data, headers)

	# A lot of things can happen, let them know which one.
	try:
		response = urlopen(request)
	except HTTPError as err:
		if err.code == 400:
			msg = "One or more headers were specified incorrectly or conflict with another header."
		elif err.code == 401:
			msg = "The cloud service did not present a valid authentication ticket."
		elif err.code == 403:
			msg = "The cloud service is not authorized to send a notification to this URI."
		elif err.code == 404:
			msg = "The channel URI is not valid or is not recognized by WNS."
		elif err.code == 405:
			msg = "Invalid method. Only POST or DELETE is allowed."
		elif err.code == 406:
			msg = "The cloud service exceeded its throttle limit"
		elif err.code == 410:
			msg = "The channel expired."
		elif err.code == 413:
			msg = "The notification payload exceeds the 500 byte limit."
		elif err.code == 500:
			msg = "An internal failure caused notification delivery to fail."
		elif err.code == 503:
			msg = "The server is currently unavailable."
		else:
			raise err
		raise WNSNotificationResponseError("HTTP %i: %s" % (err.code, msg))

	return response.read().decode("utf-8")