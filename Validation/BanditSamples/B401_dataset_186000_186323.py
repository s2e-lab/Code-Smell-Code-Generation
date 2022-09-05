def check_zk(self):
        '''
        Will attempt to telnet to each zookeeper that is used by SolrClient and issue 'mntr' command. Response is parsed to check to see if the 
        zookeeper node is a leader or a follower and returned as a dict. 

        If the telnet collection fails or the proper response is not parsed, the zk node will be listed as 'down' in the dict. Desired values are
        either follower or leader. 
        '''
        import telnetlib
        temp = self.zk_hosts.split('/')
        zks = temp[0].split(',')
        status = {}
        for zk in zks:
            self.logger.debug("Checking {}".format(zk))
            host, port = zk.split(':')
            try:
                t = telnetlib.Telnet(host, port=int(port))
                t.write('mntr'.encode('ascii'))
                r = t.read_all()
                for out in r.decode('utf-8').split('\n'):
                    if out:
                        param, val = out.split('\t')
                        if param == 'zk_server_state':
                            status[zk] = val
            except Exception as e:
                self.logger.error("Unable to reach ZK: {}".format(zk))
                self.logger.exception(e)
                status[zk] = 'down'
        #assert len(zks) == len(status)
        return status