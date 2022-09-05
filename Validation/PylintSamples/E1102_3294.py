def _build_ocsp_response(self, ocsp_request: OCSPRequest) -> OCSPResponse:
        """
        Create and return an OCSP response from an OCSP request.
        """
        # Get the certificate serial
        tbs_request = ocsp_request['tbs_request']
        request_list = tbs_request['request_list']
        if len(request_list) != 1:
            logger.warning('Received OCSP request with multiple sub requests')
            raise NotImplemented('Combined requests not yet supported')
        single_request = request_list[0]  # TODO: Support more than one request
        req_cert = single_request['req_cert']
        serial = req_cert['serial_number'].native

        # Check certificate status
        try:
            certificate_status, revocation_date = self._validate(serial)
        except Exception as e:
            logger.exception('Could not determine certificate status: %s', e)
            return self._fail(ResponseStatus.internal_error)

        # Retrieve certificate
        try:
            subject_cert_contents = self._cert_retrieve(serial)
        except Exception as e:
            logger.exception('Could not retrieve certificate with serial %s: %s', serial, e)
            return self._fail(ResponseStatus.internal_error)

        # Parse certificate
        try:
            subject_cert = asymmetric.load_certificate(subject_cert_contents.encode('utf8'))
        except Exception as e:
            logger.exception('Returned certificate with serial %s is invalid: %s', serial, e)
            return self._fail(ResponseStatus.internal_error)

        # Build the response
        builder = OCSPResponseBuilder(**{
            'response_status': ResponseStatus.successful.value,
            'certificate': subject_cert,
            'certificate_status': certificate_status.value,
            'revocation_date': revocation_date,
        })

        # Parse extensions
        for extension in tbs_request['request_extensions']:
            extn_id = extension['extn_id'].native
            critical = extension['critical'].native
            value = extension['extn_value'].parsed

            # This variable tracks whether any unknown extensions were encountered
            unknown = False

            # Handle nonce extension
            if extn_id == 'nonce':
                builder.nonce = value.native

            # That's all we know
            else:
                unknown = True

            # If an unknown critical extension is encountered (which should not
            # usually happen, according to RFC 6960 4.1.2), we should throw our
            # hands up in despair and run.
            if unknown is True and critical is True:
                logger.warning('Could not parse unknown critical extension: %r',
                        dict(extension.native))
                return self._fail(ResponseStatus.internal_error)

            # If it's an unknown non-critical extension, we can safely ignore it.
            elif unknown is True:
                logger.info('Ignored unknown non-critical extension: %r', dict(extension.native))

        # Set certificate issuer
        builder.certificate_issuer = self._issuer_cert

        # Set next update date
        builder.next_update = datetime.now(timezone.utc) + timedelta(days=self._next_update_days)

        return builder.build(self._responder_key, self._responder_cert)