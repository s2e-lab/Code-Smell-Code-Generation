def rebase(upstream, branch=None):
    """Rebase branch onto upstream

    If branch is empty, use current branch

    """
    rebase_branch = branch and branch or current_branch()
    with git_continuer(run, 'rebase --continue', no_edit=True):
        stdout = run('rebase %s %s' % (upstream, rebase_branch))
        return 'Applying' in stdout