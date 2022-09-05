def serve(service_brokers: Union[List[ServiceBroker], ServiceBroker],
          credentials: Union[List[BrokerCredentials], BrokerCredentials, None],
          logger: logging.Logger = logging.root,
          port=5000,
          debug=False):
    """
    Starts flask with the given brokers.
    You can provide a list or just one ServiceBroker

    :param service_brokers: ServicesBroker for services to provide
    :param credentials: Username and password that will be required to communicate with service broker
    :param logger: Used for api logs. This will not influence Flasks logging behavior
    :param port: Port
    :param debug: Enables debugging in flask app
    """

    from gevent.pywsgi import WSGIServer
    from flask import Flask
    app = Flask(__name__)
    app.debug = debug

    blueprint = get_blueprint(service_brokers, credentials, logger)

    logger.debug("Register openbrokerapi blueprint")
    app.register_blueprint(blueprint)

    logger.info("Start Flask on 0.0.0.0:%s" % port)
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()