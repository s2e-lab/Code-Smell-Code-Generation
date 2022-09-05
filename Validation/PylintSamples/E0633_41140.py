def create_folder_structure(self):
        """Creates a folder structure based on the project and batch name.

        Project - Batch-name - Raw-data-dir

        The info_df JSON-file will be stored in the Project folder.
        The summary-files will be saved in the Batch-name folder.
        The raw data (including exported cycles and ica-data) will be saved to
        the Raw-data-dir.

        """
        self.info_file, directories = create_folder_structure(self.project,
                                                              self.name)
        self.project_dir, self.batch_dir, self.raw_dir = directories
        logger.debug("create folders:" + str(directories))