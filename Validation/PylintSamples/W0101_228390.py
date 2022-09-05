def _encoder(self, obj):
        """ Encode a toc element leaf-node """
        return {'__class__': obj.__class__.__name__,
                'ident': obj.ident,
                'group': obj.group,
                'name': obj.name,
                'ctype': obj.ctype,
                'pytype': obj.pytype,
                'access': obj.access}
        raise TypeError(repr(obj) + ' is not JSON serializable')