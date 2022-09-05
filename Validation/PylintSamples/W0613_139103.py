def handle(self, connection_id, message_content):
        """
        If an AuthorizationViolation is recieved, the connection has decided
        that this validator is no longer permitted to be connected.
        Remove the connection preemptively.
        """
        LOGGER.warning("Received AuthorizationViolation from %s",
                       connection_id)
        # Close the connection
        endpoint = self._network.connection_id_to_endpoint(connection_id)
        self._network.remove_connection(connection_id)
        self._gossip.remove_temp_endpoint(endpoint)
        return HandlerResult(HandlerStatus.DROP)