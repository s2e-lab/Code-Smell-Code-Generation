def security_pkg(self, pkg):
        """Warning message for some special reasons
        """
        print("")
        self.template(78)
        print("| {0}{1}*** WARNING ***{2}").format(
            " " * 27, self.meta.color["RED"], self.meta.color["ENDC"])
        self.template(78)
        print("| Before proceed with the package '{0}' will you must read\n"
              "| the README file. You can use the command "
              "'slpkg -n {1}'").format(pkg, pkg)
        self.template(78)
        print("")