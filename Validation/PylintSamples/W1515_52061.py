def validate(self,value):
        """Validate the parameter"""
        if self.validator is not None:
            try:
                valid = self.validator(value)
            except Exception as e:
                import pdb; pdb.set_trace()
            if isinstance(valid, tuple) and len(valid) == 2:
                valid, errormsg = valid
            elif isinstance(valid, bool):
                errormsg = "Invalid value"
            else:
                raise TypeError("Custom validator must return a boolean or a (bool, errormsg) tuple.")
            if valid:
                self.error = None
            else:
                self.error = errormsg
            return valid
        else:
            self.error = None #reset error
            return True