def _parse_xml(self):
        """Extracts the XML settings into class instances that can operate on
        the settings to perform the testing functions.
        """
        import xml.etree.ElementTree as ET
        from os import path
        #This dict has the keys of XML tags that are required in order for the
        #CI server to run the repo. When each one is parsed, we change its value
        #to True and then check that they are all true at the end.
        required = {"testing": False, "wiki": False}
        #Make sure the file exists and then import it as XML and read the values out.
        if path.isfile(self.filepath):
            tree = ET.parse(self.filepath)
            vms("Parsing XML tree from {}.".format(self.filepath), 2)
            root = tree.getroot()
            if root.tag != "cirepo":
                raise ValueError("The root tag in a continuous integration settings XML "
                                 "file should be a <cirepo> tag.")

            self._parse_repo(root)
            for child in root:
                if child.tag == "cron":
                    if self.server is not None:
                        self.server.cron.settings[self.name] = CronSettings(child)
                if child.tag == "testing":
                    self.testing = TestingSettings(child)
                if child.tag == "static":
                    self.static = StaticSettings(child)
                if child.tag == "wiki":
                    self.wiki["user"] = get_attrib(child, "user", "wiki")
                    self.wiki["password"] = get_attrib(child, "password", "wiki")
                    self.wiki["basepage"] = get_attrib(child, "basepage", "wiki")
                if child.tag in required:
                    required[child.tag] = True

            if not all(required.values()):
                tags = ', '.join(["<{}>".format(t) for t in required])
                raise ValueError("{} are required tags in the repo's XML settings file.".format(tags))