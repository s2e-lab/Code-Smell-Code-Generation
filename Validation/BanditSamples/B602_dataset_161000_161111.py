def symlink_to(orig, dest):
    """Create a symlink. Used for model shortcut links.

    orig (unicode / Path): The origin path.
    dest (unicode / Path): The destination path of the symlink.
    """
    if is_windows:
        import subprocess

        subprocess.check_call(
            ["mklink", "/d", path2str(orig), path2str(dest)], shell=True
        )
    else:
        orig.symlink_to(dest)