def create_dat_file(self):
        """
        Create and write empty data file in the data directory
        """
        output = "## {}\n".format(self.name)
        try:
            kwargs_items = self.kwargs.iteritems()
        except AttributeError:
            kwargs_items = self.kwargs.items()
        for key, val in kwargs_items:
            if val is "l":
                output += "#l {}=\n".format(str(key))
            elif val is "f" or True:
                output += "#f {}=\n".format(str(key))
        comment = "## " + "\t".join(["col{" + str(i) + ":d}"
                                     for i in range(self.argnum)])
        comment += "\n"
        rangeargnum = range(self.argnum)
        output += comment.format(*rangeargnum)
        if os.path.isfile(self.location_dat):
            files = glob.glob(self.location_dat + "*")
            count = 2
            while (
                    (self.location_dat + str(count) in files)
                  ) and (count <= 10):
                count += 1
            os.rename(self.location_dat, self.location_dat + str(count))
        dat_file = open(self.location_dat, "wb")
        dat_file.write(output)
        dat_file.close()