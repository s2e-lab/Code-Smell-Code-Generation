def decrypt_report(self, device_id, root, data, **kwargs):
        """Decrypt a buffer of report data on behalf of a device.

        Args:
            device_id (int): The id of the device that we should encrypt for
            root (int): The root key type that should be used to generate the report
            data (bytearray): The data that we should decrypt
            **kwargs: There are additional specific keyword args that are required
                depending on the root key used.  Typically, you must specify
                - report_id (int): The report id
                - sent_timestamp (int): The sent timestamp of the report

                These two bits of information are used to construct the per report
                signing and encryption key from the specific root key type.

        Returns:
            dict: The decrypted data and any associated metadata about the data.
                The data itself must always be a bytearray stored under the 'data'
                key, however additional keys may be present depending on the encryption method
                used.

        Raises:
            NotFoundError: If the auth provider is not able to decrypt the data.
        """

        report_key = self._verify_derive_key(device_id, root, **kwargs)

        try:
            from Crypto.Cipher import AES
            import Crypto.Util.Counter
        except ImportError:
            raise NotFoundError

        ctr = Crypto.Util.Counter.new(128)

        # We use AES-128 for encryption
        encryptor = AES.new(bytes(report_key[:16]), AES.MODE_CTR, counter=ctr)

        decrypted = encryptor.decrypt(bytes(data))
        return {'data': decrypted}