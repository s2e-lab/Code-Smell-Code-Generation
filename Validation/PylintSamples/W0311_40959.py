def contribute_to_class(self, cls, name):
        '''
        Sets up the signal processor. Since self.model is not available
        in the constructor, we perform this operation here.
        '''
        super(BungiesearchManager, self).contribute_to_class(cls, name)

        from . import Bungiesearch
        from .signals import get_signal_processor
        settings = Bungiesearch.BUNGIE
        if 'SIGNALS' in settings:
            self.signal_processor = get_signal_processor()
            self.signal_processor.setup(self.model)