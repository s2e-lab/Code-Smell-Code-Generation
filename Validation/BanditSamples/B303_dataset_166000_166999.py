def update_hash(self, layers: Iterable):
        """
        Calculation of `hash_id` of Layer. Which is determined by the properties of itself, and the `hash_id`s of input layers
        """
        if self.graph_type == LayerType.input.value:
            return
        hasher = hashlib.md5()
        hasher.update(LayerType(self.graph_type).name.encode('ascii'))
        hasher.update(str(self.size).encode('ascii'))
        for i in self.input:
            if layers[i].hash_id is None:
                raise ValueError('Hash id of layer {}: {} not generated!'.format(i, layers[i]))
            hasher.update(layers[i].hash_id.encode('ascii'))
        self.hash_id = hasher.hexdigest()