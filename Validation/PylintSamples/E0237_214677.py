def fetch_git_package(self, config):
        """Make a remote git repository available for local use.

        Args:
            config (dict): git config dictionary

        """
        # only loading git here when needed to avoid load errors on systems
        # without git installed
        from git import Repo

        ref = self.determine_git_ref(config)
        dir_name = self.sanitize_git_path(uri=config['uri'], ref=ref)
        cached_dir_path = os.path.join(self.package_cache_dir, dir_name)

        # We can skip cloning the repo if it's already been cached
        if not os.path.isdir(cached_dir_path):
            logger.debug("Remote repo %s does not appear to have been "
                         "previously downloaded - starting clone to %s",
                         config['uri'],
                         cached_dir_path)
            tmp_dir = tempfile.mkdtemp(prefix='stacker')
            try:
                tmp_repo_path = os.path.join(tmp_dir, dir_name)
                with Repo.clone_from(config['uri'], tmp_repo_path) as repo:
                    repo.head.reference = ref
                    repo.head.reset(index=True, working_tree=True)
                shutil.move(tmp_repo_path, self.package_cache_dir)
            finally:
                shutil.rmtree(tmp_dir)
        else:
            logger.debug("Remote repo %s appears to have been previously "
                         "cloned to %s -- bypassing download",
                         config['uri'],
                         cached_dir_path)

        # Update sys.path & merge in remote configs (if necessary)
        self.update_paths_and_config(config=config,
                                     pkg_dir_name=dir_name)