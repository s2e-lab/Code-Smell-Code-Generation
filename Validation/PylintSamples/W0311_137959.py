def _do_flood(self, in_port, msg):
        """the process when the snooper received a message of the
        outside for processing. """
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        self._do_packet_out(datapath, msg.data, in_port, actions)