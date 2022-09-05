def create_bitcoind_connection( rpc_username, rpc_password, server, port, use_https, timeout ):
    """
    Creates an RPC client to a bitcoind instance.
    It will have ".opts" defined as a member, which will be a dict that stores the above connection options.
    """
    
    from .bitcoin_blockchain import AuthServiceProxy

    global do_wrap_socket, create_ssl_authproxy
        
    log.debug("[%s] Connect to bitcoind at %s://%s@%s:%s, timeout=%s" % (os.getpid(), 'https' if use_https else 'http', rpc_username, server, port, timeout) )
    
    protocol = 'https' if use_https else 'http'
    if not server or len(server) < 1:
        raise Exception('Invalid bitcoind host address.')
    if not port or not is_valid_int(port):
        raise Exception('Invalid bitcoind port number.')
    
    authproxy_config_uri = '%s://%s:%s@%s:%s' % (protocol, rpc_username, rpc_password, server, port)
    
    if use_https:
        # TODO: ship with a cert
        if do_wrap_socket:
           # ssl._create_unverified_context and ssl.create_default_context are not supported.
           # wrap the socket directly 
           connection = BitcoindConnection( server, int(port), timeout=timeout )
           ret = AuthServiceProxy(authproxy_config_uri, connection=connection)
           
        elif create_ssl_authproxy:
           # ssl has _create_unverified_context, so we're good to go 
           ret = AuthServiceProxy(authproxy_config_uri, timeout=timeout)
        
        else:
           # have to set up an unverified context ourselves 
           ssl_ctx = ssl.create_default_context()
           ssl_ctx.check_hostname = False
           ssl_ctx.verify_mode = ssl.CERT_NONE
           connection = httplib.HTTPSConnection( server, int(port), context=ssl_ctx, timeout=timeout )
           ret = AuthServiceProxy(authproxy_config_uri, connection=connection)
          
    else:
        ret = AuthServiceProxy(authproxy_config_uri)

    # remember the options 
    bitcoind_opts = {
       "bitcoind_user": rpc_username,
       "bitcoind_passwd": rpc_password,
       "bitcoind_server": server,
       "bitcoind_port": port,
       "bitcoind_use_https": use_https,
       "bitcoind_timeout": timeout
    }
    
    setattr( ret, "opts", bitcoind_opts )
    return ret