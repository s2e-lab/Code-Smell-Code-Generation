def create_environment(self, **kwargs):
        """
        Return a new Jinja environment.
        
        Derived classes may override method to pass additional parameters or to change the template
        loader type.
        """
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_path),
            **kwargs
        )