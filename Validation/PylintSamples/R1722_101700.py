def build_main_image_cmd(self, srcdir, force):
        """
        Build the main image to be used for launching containers
        """
        check_permissions()

        basetag = self.conf.basetag
        basedir = self.conf.basedir
        maintag = self.conf.maintag

        if not self.image_exists(tag=basetag):
            if not force:
                exit("Base image with tag {0} does not exist".format(basetag))
            else:
                echo("FORCE given. Forcefully building the base image.")
                self.build_base_image_cmd(force)

        if self.image_exists(tag=maintag):
            self.remove_image(tag=maintag)

        build_command = "/build/make-install-gluster.sh"
        container = self.create_container(image=basetag, command=build_command,
                                          volumes=["/build", "/src"])

        self.start(container, binds={basedir: "/build", srcdir: "/src"})
        echo('Building main image')
        while self.inspect_container(container)["State"]["Running"]:
            time.sleep(5)

        if not self.inspect_container(container)["State"]["ExitCode"] == 0:
            echo("Build failed")
            echo("Dumping logs")
            echo(self.logs(container))
            exit()

        # The docker remote api expects the repository and tag to be seperate
        # items for commit
        repo = maintag.split(':')[0]
        tag = maintag.split(':')[1]
        image = self.commit(container['Id'], repository=repo, tag=tag)
        echo("Built main image {0} (Id: {1})".format(maintag, image['Id']))