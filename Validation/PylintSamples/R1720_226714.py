def callback(self, output, inputs=[], state=[], events=[]): # pylint: disable=dangerous-default-value
        'Invoke callback, adjusting variable names as needed'

        if isinstance(output, (list, tuple)):
            fixed_outputs = [self._fix_callback_item(x) for x in output]
            # Temporary check; can be removed once the library has been extended
            raise NotImplementedError("django-plotly-dash cannot handle multiple callback outputs at present")
        else:
            fixed_outputs = self._fix_callback_item(output)

        return super(WrappedDash, self).callback(fixed_outputs,
                                                 [self._fix_callback_item(x) for x in inputs],
                                                 [self._fix_callback_item(x) for x in state])