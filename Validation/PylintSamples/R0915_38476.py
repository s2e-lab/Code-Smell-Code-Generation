def construct(self, request, service=None, http_args=None, **kwargs):
        """
        Constructs a client assertion and signs it with a key.
        The request is modified as a side effect.

        :param request: The request
        :param service: A :py:class:`oidcservice.service.Service` instance
        :param http_args: HTTP arguments
        :param kwargs: Extra arguments
        :return: Constructed HTTP arguments, in this case none
        """

        if 'client_assertion' in kwargs:
            request["client_assertion"] = kwargs['client_assertion']
            if 'client_assertion_type' in kwargs:
                request[
                    'client_assertion_type'] = kwargs['client_assertion_type']
            else:
                request["client_assertion_type"] = JWT_BEARER
        elif 'client_assertion' in request:
            if 'client_assertion_type' not in request:
                request["client_assertion_type"] = JWT_BEARER
        else:
            algorithm = None
            _context = service.service_context
            # audience for the signed JWT depends on which endpoint
            # we're talking to.
            if kwargs['authn_endpoint'] in ['token_endpoint']:
                try:
                    algorithm = _context.behaviour[
                        'token_endpoint_auth_signing_alg']
                except (KeyError, AttributeError):
                    pass
                audience = _context.provider_info['token_endpoint']
            else:
                audience = _context.provider_info['issuer']

            if not algorithm:
                algorithm = self.choose_algorithm(**kwargs)

            ktype = alg2keytype(algorithm)
            try:
                if 'kid' in kwargs:
                    signing_key = [self.get_key_by_kid(kwargs["kid"], algorithm,
                                                       _context)]
                elif ktype in _context.kid["sig"]:
                    try:
                        signing_key = [self.get_key_by_kid(
                            _context.kid["sig"][ktype], algorithm, _context)]
                    except KeyError:
                        signing_key = self.get_signing_key(algorithm, _context)
                else:
                    signing_key = self.get_signing_key(algorithm, _context)
            except NoMatchingKey as err:
                logger.error("%s" % sanitize(err))
                raise

            try:
                _args = {'lifetime': kwargs['lifetime']}
            except KeyError:
                _args = {}

            # construct the signed JWT with the assertions and add
            # it as value to the 'client_assertion' claim of the request
            request["client_assertion"] = assertion_jwt(
                _context.client_id, signing_key, audience,
                algorithm, **_args)

            request["client_assertion_type"] = JWT_BEARER

        try:
            del request["client_secret"]
        except KeyError:
            pass

        # If client_id is not required to be present, remove it.
        if not request.c_param["client_id"][VREQUIRED]:
            try:
                del request["client_id"]
            except KeyError:
                pass

        return {}