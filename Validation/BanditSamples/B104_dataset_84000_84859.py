def start(path=None, host=None, port=None, color=None, cors=None, detach=False, nolog=False):
    """start web server"""
    if detach:
        sys.argv.append('--no-log')
        idx = sys.argv.index('-d')
        del sys.argv[idx]
        cmd = sys.executable + ' ' + ' '.join([sys.argv[0], 'start'] + sys.argv[1:])
        if os.name == 'nt':
            cmd = 'start /B %s' % cmd
        else:
            cmd = '%s &' % cmd
        os.system(cmd)
    else:    
        if path:
            path = os.path.abspath(path)
        app.config['PATH_HTML']= first_value(path, app.config.get('PATH_HTML',None), os.getcwd())
        app.config['HOST'] = first_value(host, app.config.get('HOST',None), '0.0.0.0')
        app.config['PORT'] = int(first_value(port, app.config.get('PORT',None), 5001))
        app.logger.setLevel(logging.DEBUG)
        app.config['historylog'] = HistoryHandler()
        app.logger.addHandler(app.config['historylog'])
        if not nolog:
            app.logger.addHandler(StreamHandler())
        if cors: CORS(app)
        app.run(host = app.config['HOST'],
                port = app.config['PORT'],
                threaded = True)