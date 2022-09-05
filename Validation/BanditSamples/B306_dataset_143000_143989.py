def from_tree(cls, repo, *treeish, **kwargs):
        """Merge the given treeish revisions into a new index which is returned.
        The original index will remain unaltered

        :param repo:
            The repository treeish are located in.

        :param treeish:
            One, two or three Tree Objects, Commits or 40 byte hexshas. The result
            changes according to the amount of trees.
            If 1 Tree is given, it will just be read into a new index
            If 2 Trees are given, they will be merged into a new index using a
            two way merge algorithm. Tree 1 is the 'current' tree, tree 2 is the 'other'
            one. It behaves like a fast-forward.
            If 3 Trees are given, a 3-way merge will be performed with the first tree
            being the common ancestor of tree 2 and tree 3. Tree 2 is the 'current' tree,
            tree 3 is the 'other' one

        :param kwargs:
            Additional arguments passed to git-read-tree

        :return:
            New IndexFile instance. It will point to a temporary index location which
            does not exist anymore. If you intend to write such a merged Index, supply
            an alternate file_path to its 'write' method.

        :note:
            In the three-way merge case, --aggressive will be specified to automatically
            resolve more cases in a commonly correct manner. Specify trivial=True as kwarg
            to override that.

            As the underlying git-read-tree command takes into account the current index,
            it will be temporarily moved out of the way to assure there are no unsuspected
            interferences."""
        if len(treeish) == 0 or len(treeish) > 3:
            raise ValueError("Please specify between 1 and 3 treeish, got %i" % len(treeish))

        arg_list = []
        # ignore that working tree and index possibly are out of date
        if len(treeish) > 1:
            # drop unmerged entries when reading our index and merging
            arg_list.append("--reset")
            # handle non-trivial cases the way a real merge does
            arg_list.append("--aggressive")
        # END merge handling

        # tmp file created in git home directory to be sure renaming
        # works - /tmp/ dirs could be on another device
        tmp_index = tempfile.mktemp('', '', repo.git_dir)
        arg_list.append("--index-output=%s" % tmp_index)
        arg_list.extend(treeish)

        # move current index out of the way - otherwise the merge may fail
        # as it considers existing entries. moving it essentially clears the index.
        # Unfortunately there is no 'soft' way to do it.
        # The TemporaryFileSwap assure the original file get put back
        index_handler = TemporaryFileSwap(join_path_native(repo.git_dir, 'index'))
        try:
            repo.git.read_tree(*arg_list, **kwargs)
            index = cls(repo, tmp_index)
            index.entries       # force it to read the file as we will delete the temp-file
            del(index_handler)  # release as soon as possible
        finally:
            if osp.exists(tmp_index):
                os.remove(tmp_index)
        # END index merge handling

        return index