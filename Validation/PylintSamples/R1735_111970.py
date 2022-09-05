def publish(self, topic=None, msg=None, modname=None,
                pre_fire_hook=None, **kw):
        """
        Send a message over the publishing zeromq socket.

            >>> import fedmsg
            >>> fedmsg.publish(topic='testing', modname='test', msg={
            ...     'test': "Hello World",
            ... })

        The above snippet will send the message ``'{test: "Hello World"}'``
        over the ``<topic_prefix>.dev.test.testing`` topic. The fully qualified
        topic of a message is constructed out of the following pieces:

            <:ref:`conf-topic-prefix`>.<:ref:`conf-environment`>.<``modname``>.<``topic``>

        This function (and other API functions) do a little bit more
        heavy lifting than they let on.  If the "zeromq context" is not yet
        initialized, :func:`fedmsg.init` is called to construct it and
        store it as :data:`fedmsg.__local.__context` before anything else is
        done.

        **An example from Fedora Tagger -- SQLAlchemy encoding**

        Here's an example from
        `fedora-tagger <https://github.com/fedora-infra/fedora-tagger>`_ that
        sends the information about a new tag over
        ``org.fedoraproject.{dev,stg,prod}.fedoratagger.tag.update``::

          >>> import fedmsg
          >>> fedmsg.publish(topic='tag.update', msg={
          ...     'user': user,
          ...     'tag': tag,
          ... })

        Note that the `tag` and `user` objects are SQLAlchemy objects defined
        by tagger.  They both have ``.__json__()`` methods which
        :func:`fedmsg.publish` uses to encode both objects as stringified
        JSON for you.  Under the hood, specifically, ``.publish`` uses
        :mod:`fedmsg.encoding` to do this.

        ``fedmsg`` has also guessed the module name (``modname``) of it's
        caller and inserted it into the topic for you.  The code from which
        we stole the above snippet lives in
        ``fedoratagger.controllers.root``.  ``fedmsg`` figured that out and
        stripped it down to just ``fedoratagger`` for the final topic of
        ``org.fedoraproject.{dev,stg,prod}.fedoratagger.tag.update``.

        **Shell Usage**

        You could also use the ``fedmsg-logger`` from a shell script like so::

            $ echo "Hello, world." | fedmsg-logger --topic testing
            $ echo '{"foo": "bar"}' | fedmsg-logger --json-input

        :param topic: The message topic suffix. This suffix is joined to the
            configured topic prefix (e.g. ``org.fedoraproject``), environment
            (e.g. ``prod``, ``dev``, etc.), and modname.
        :type topic: unicode
        :param msg: A message to publish. This message will be JSON-encoded
            prior to being sent, so the object must be composed of JSON-
            serializable data types. Please note that if this is already a
            string JSON serialization will be applied to that string.
        :type msg: dict
        :param modname: The module name that is publishing the message. If this
            is omitted, ``fedmsg`` will try to guess the name of the module
            that called it and use that to produce an intelligent topic.
            Specifying ``modname`` explicitly overrides this behavior.
        :type modname: unicode
        :param pre_fire_hook: A callable that will be called with a single
            argument -- the dict of the constructed message -- just before it
            is handed off to ZeroMQ for publication.
        :type pre_fire_hook: function
        """

        topic = topic or 'unspecified'
        msg = msg or dict()

        # If no modname is supplied, then guess it from the call stack.
        modname = modname or guess_calling_module(default="fedmsg")
        topic = '.'.join([modname, topic])

        if topic[:len(self.c['topic_prefix'])] != self.c['topic_prefix']:
            topic = '.'.join([
                self.c['topic_prefix'],
                self.c['environment'],
                topic,
            ])

        if isinstance(topic, six.text_type):
            topic = to_bytes(topic, encoding='utf8', nonstring="passthru")

        year = datetime.datetime.now().year

        self._i += 1
        msg = dict(
            topic=topic.decode('utf-8'),
            msg=msg,
            timestamp=int(time.time()),
            msg_id=str(year) + '-' + str(uuid.uuid4()),
            i=self._i,
            username=getpass.getuser(),
        )

        # Find my message-signing cert if I need one.
        if self.c.get('sign_messages', False):
            if not self.c.get("crypto_backend") == "gpg":
                if 'cert_prefix' in self.c:
                    cert_index = "%s.%s" % (self.c['cert_prefix'],
                                            self.hostname)
                else:
                    cert_index = self.c['name']
                    if cert_index == 'relay_inbound':
                        cert_index = "shell.%s" % self.hostname

                self.c['certname'] = self.c['certnames'][cert_index]
            else:
                if 'gpg_signing_key' not in self.c:
                    self.c['gpg_signing_key'] = self.c['gpg_keys'][self.hostname]

        if self.c.get('sign_messages', False):
            msg = fedmsg.crypto.sign(msg, **self.c)

        store = self.c.get('persistent_store', None)
        if store:
            # Add the seq_id field
            msg = store.add(msg)

        if pre_fire_hook:
            pre_fire_hook(msg)

        # We handle zeromq publishing ourselves.  But, if that is disabled,
        # defer to the moksha' hub's twisted reactor to send messages (if
        # available).
        if self.c.get('zmq_enabled', True):
            self.publisher.send_multipart(
                [topic, fedmsg.encoding.dumps(msg).encode('utf-8')],
                flags=zmq.NOBLOCK,
            )
        else:
            # Perhaps we're using STOMP or AMQP?  Let moksha handle it.
            import moksha.hub
            # First, a quick sanity check.
            if not getattr(moksha.hub, '_hub', None):
                raise AttributeError("Unable to publish non-zeromq msg "
                                     "without moksha-hub initialization.")
            # Let moksha.hub do our work.
            moksha.hub._hub.send_message(
                topic=topic,
                message=fedmsg.encoding.dumps(msg).encode('utf-8'),
                jsonify=False,
            )