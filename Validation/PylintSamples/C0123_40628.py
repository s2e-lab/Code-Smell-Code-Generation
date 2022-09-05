def add_state_execution_output_to_scoped_data(self, dictionary, state):
        """Add a state execution output to the scoped data

        :param dictionary: The dictionary that is added to the scoped data
        :param state: The state that finished execution and provide the dictionary
        """
        for output_name, value in dictionary.items():
            for output_data_port_key, data_port in list(state.output_data_ports.items()):
                if output_name == data_port.name:
                    if not isinstance(value, data_port.data_type):
                        if (not ((type(value) is float or type(value) is int) and
                                     (data_port.data_type is float or data_port.data_type is int)) and
                                not (isinstance(value, type(None)))):
                            logger.error("The data type of output port {0} should be of type {1}, but is of type {2}".
                                         format(output_name, data_port.data_type, type(value)))
                    self.scoped_data[str(output_data_port_key) + state.state_id] = \
                        ScopedData(data_port.name, value, type(value), state.state_id, OutputDataPort, parent=self)