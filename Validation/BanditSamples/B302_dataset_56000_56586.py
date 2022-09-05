def loads(astring):
        """Decompress and deserialize string into Python object via marshal."""
        try:
            return marshal.loads(zlib.decompress(astring))
        except zlib.error as e:
            raise SerializerError(
                'Cannot decompress object ("{}")'.format(str(e))
            )
        except Exception as e:
            # marshal module does not provide a proper Exception model
            raise SerializerError(
                'Cannot restore object ("{}")'.format(str(e))
            )