#!/usr/bin/env python

import logging
import re
import threading
from types import MethodType
from telnetlib import Telnet
from distutils.version import StrictVersion
import time

import memcache
from uhashring.monkey import patch_memcache
from elasticache_pyclient.repeat_timer import RepeatTimer

patch_memcache()

logger = logging.getLogger('elasticache')


class ElasticacheInvalidTelentReplyError(Exception):
    """
    receive configuration from endpoint is invalide
    """
    pass


class Cluster():
    """
    get cluster configuration, keep version and node list
    """

    def __init__(self, endpoint, timeout):
        host, port = endpoint.split(':')
        self.host = host
        self.port = port
        self.timeout = timeout
        logger.debug('cluster: %s %s %d', host, port, timeout)
        self.version = 0
        self.lock = threading.Lock()
        self.update()

    def update(self):
        tn = Telnet(self.host, self.port, self.timeout)
        tn.write(b'version\n')
        ret = tn.read_until(b'\r\n', self.timeout).decode('utf-8')
        logger.debug('version: %s', ret)
        version_list = ret.split(' ')
        if len(version_list) != 2 or version_list[0] != 'VERSION':
            raise ElasticacheInvalidTelentReplyError(ret)
        version = version_list[1][0:-2]
        if StrictVersion(version) >= StrictVersion('1.4.14'):
            get_cluster = b'config get cluster\n'
        else:
            get_cluster = b'get AmazonElastiCache:cluster\n'
        tn.write(get_cluster)
        ret = tn.read_until(b'END\r\n', self.timeout).decode('utf-8')
        logger.debug('config: %s', ret)
        tn.close()
        p = re.compile(r'\r?\n')
        conf = p.split(ret)
        if len(conf) != 6 or conf[4][0:3] != 'END':
            raise ElasticacheInvalidTelentReplyError(ret)
        version = int(conf[1])
        servers = []
        nodes_str = conf[2].split(' ')
        for node_str in nodes_str:
            node_list = node_str.split('|')
            if len(node_list) != 3:
                raise ElasticacheInvalidTelentReplyError(ret)
            servers.append(node_list[0] + ':' + node_list[2])
        with self.lock:
            if version > self.version:
                self.servers = servers
                self.version = version
                self.timestamp = time.time()
                logger.debug('cluster update: %s', self)

    def __str__(self):
        return '%d %s %f' % (self.version, self.servers, self.timestamp)


class WrapperClient(threading.local):

    def __init__(self, cluster, *args, **kwargs):
        self.cluster = cluster
        with self.cluster.lock:
            self.client = memcache.Client(
                cluster.servers, *args, **kwargs)
            self.timestamp = time.time()
            assert(self.timestamp > self.cluster.timestamp)
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name):
        # it should always be true
        # because MemcacheClient has checked it
        assert(hasattr(memcache.Client, name))

        def wrapper(self, *args, **kwargs):
            with self.cluster.lock:
                if self.cluster.timestamp > self.timestamp:
                    logger.info('cluster changed: %s', self.cluster)
                    self.client.disconnect_all()
                    self.client = memcache.Client(
                        self.cluster.servers, *self.args, **self.kwargs)
                    self.timestamp = time.time()
                    assert(self.timestamp > self.cluster.timestamp)
            func = getattr(self.client, name)
            return func(*args, **kwargs)

        wrapper.__name__ = name
        method_func = MethodType(wrapper, self)
        setattr(self, name, method_func)
        return method_func


class MemcacheClient(object):
    """
    Implement autodiscovery for elasticache memcache cluster
    """

    def __init__(
            self, endpoint, ad_timeout=10, ad_interval=60, *args, **kwargs):
        """
        Create a new Client object, and launch a timer for auto discovery

        :param endpoint: String
        something like: test.lwgyhw.cfg.usw2.cache.amazonaws.com:11211

        :param ad_timeout: Int
        socket connection timeout during auto discovery, in the unit of second

        :param ad_interval: Int
        auto discovery interval, the unit is second, in the unit of second

        All other parameters will be passed to python-memcached
        """
        self.endpoint = endpoint
        self.ad_timeout = ad_timeout
        cluster = Cluster(endpoint, ad_timeout)
        self.cluster = cluster
        self.wc = WrapperClient(cluster, *args, **kwargs)
        self.lock = threading.Lock()
        self.timer = RepeatTimer('autodiscovery', ad_interval, self._update)
        self.timer.start()

    def __getattr__(self, name):
        if not hasattr(memcache.Client, name):
            msg = 'no attribute %s' % name
            raise AttributeError(msg)
        with self.lock:
            func = getattr(self.wc, name)

            def wrapper(self, *args, **kwargs):
                return func(*args, **kwargs)

            wrapper.__name__ = name
            method_func = MethodType(wrapper, self)
            setattr(self, name, method_func)
            return method_func

    def _update(self):
        self.cluster.update()

    def stop_timer(self):
        """
        Every MemcacheClient will start a timer for auto discovery,
        if do not use MemcacheClient object anymore,
        please call this funciton to stop the timer,
        or the timer will run forever
        """
        self.timer.stop_timer()
        self.timer.join()

    def cluster_size(self):
        with self.cluster.lock:
            return len(self.cluster.servers)
