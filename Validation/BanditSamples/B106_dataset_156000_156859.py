def module_def(self):
        """
        This is a module definition so parse content till end.
        """
        if self.module_level == MAX_NESTING_LEVELS:
            self.error("Module nested too deep")
        self.module_level += 1
        module = ModuleDefinition()
        self.parse(module, end_token="}")
        self.module_level -= 1
        self.current_module.pop()
        return module