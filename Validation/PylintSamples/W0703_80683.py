def request(self, *args, **kwargs):
		"""Overrided method. Returns jsonrpc response
		or fetches exception? returns appropriate data to client
		and response mail to administrator.
		"""
		try:
			import settings

			with open(os.path.join(settings.BASE_DIR, "keys.json")) as f:
				keys = json.load(f)
				privkey = keys["privkey"]

			message = json.dumps(kwargs)
			signature = Bip32Keys.sign_message(message, privkey)
			result = super().request(method_name=kwargs["method_name"],
												message=message, signature=signature)
			return result
		except ConnectionRefusedError:
			return {"error":500, 
					"reason": "Service connection error."}
		except Exception as e:
			return {"error":500, "reason": str(e)}