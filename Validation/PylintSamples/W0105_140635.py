def _setup_advanced_theme(self, theme_name, output_dir, advanced_name):
        """
        Setup all the files required to enable an advanced theme.

        Copies all the files over and creates the required directories
        if they do not exist.

        :param theme_name: theme to copy the files over from
        :param output_dir: output directory to place the files in
        """
        """Directories"""
        output_theme_dir = os.path.join(output_dir, advanced_name)
        output_images_dir = os.path.join(output_theme_dir, advanced_name)
        input_theme_dir = os.path.join(
            utils.get_themes_directory(theme_name, self.png_support), theme_name)
        input_images_dir = os.path.join(input_theme_dir, theme_name)
        advanced_pkg_dir = os.path.join(utils.get_file_directory(), "advanced")
        """Directory creation"""
        for directory in [output_dir, output_theme_dir]:
            utils.create_directory(directory)
        """Theme TCL file"""
        file_name = theme_name + ".tcl"
        theme_input = os.path.join(input_theme_dir, file_name)
        theme_output = os.path.join(output_theme_dir, "{}.tcl".format(advanced_name))
        with open(theme_input, "r") as fi, open(theme_output, "w") as fo:
            for line in fi:
                # Setup new theme
                line = line.replace(theme_name, advanced_name)
                # Setup new image format
                line = line.replace("gif89", "png")
                line = line.replace("gif", "png")
                # Write processed line
                fo.write(line)
        """pkgIndex.tcl file"""
        theme_pkg_input = os.path.join(advanced_pkg_dir, "pkgIndex.tcl")
        theme_pkg_output = os.path.join(output_theme_dir, "pkgIndex.tcl")
        with open(theme_pkg_input, "r") as fi, open(theme_pkg_output, "w") as fo:
            for line in fi:
                fo.write(line.replace("advanced", advanced_name))
        """pkgIndex_package.tcl -> pkgIndex.tcl"""
        theme_pkg_input = os.path.join(advanced_pkg_dir, "pkgIndex_package.tcl")
        theme_pkg_output = os.path.join(output_dir, "pkgIndex.tcl")
        with open(theme_pkg_input, "r") as fi, open(theme_pkg_output, "w") as fo:
            for line in fi:
                fo.write(line.replace("advanced", advanced_name))
        """Images"""
        if os.path.exists(output_images_dir):
            rmtree(output_images_dir)
        copytree(input_images_dir, output_images_dir)