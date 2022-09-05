def describe(cls, sha1):
        """Returns a human-readable representation of the given SHA1."""

        # For now we invoke git-describe(1), but eventually we will be
        # able to do this via pygit2, since libgit2 already provides
        # an API for this:
        #   https://github.com/libgit2/pygit2/pull/459#issuecomment-68866929
        #   https://github.com/libgit2/libgit2/pull/2592
        cmd = [
            'git', 'describe',
            '--all',       # look for tags and branches
            '--long',      # remotes/github/master-0-g2b6d591
            # '--contains',
            # '--abbrev',
            sha1
        ]
        # cls.logger.debug(" ".join(cmd))
        out = None
        try:
            out = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            if e.output.find('No tags can describe') != -1:
                return ''
            raise

        out = out.strip()
        out = re.sub(r'^(heads|tags|remotes)/', '', out)
        # We already have the abbreviated SHA1 from abbreviate_sha1()
        out = re.sub(r'-g[0-9a-f]{7,}$', '', out)
        # cls.logger.debug(out)
        return out