def install(shell=None, prog_name=None, env_name=None, path=None, append=None, extra_env=None):
    """Install the completion

    Parameters
    ----------
    shell : Shell
        The shell type targeted. It will be guessed with get_auto_shell() if the value is None (Default value = None)
    prog_name : str
        The program name on the command line. It will be automatically computed if the value is None
        (Default value = None)
    env_name : str
        The environment variable name used to control the completion. It will be automatically computed if the value is
        None (Default value = None)
    path : str
        The installation path of the code to be evaluated by the shell. The standard installation path is used if the
        value is None (Default value = None)
    append : bool
        Whether to append the content to the file or to override it. The default behavior depends on the shell type
        (Default value = None)
    extra_env : dict
        A set of environment variables and their values to be added to the generated code (Default value = None)
    """
    prog_name = prog_name or click.get_current_context().find_root().info_name
    shell = shell or get_auto_shell()
    if append is None and path is not None:
        append = True
    if append is not None:
        mode = 'a' if append else 'w'
    else:
        mode = None

    if shell == 'fish':
        path = path or os.path.expanduser('~') + '/.config/fish/completions/%s.fish' % prog_name
        mode = mode or 'w'
    elif shell == 'bash':
        path = path or os.path.expanduser('~') + '/.bash_completion'
        mode = mode or 'a'
    elif shell == 'zsh':
        ohmyzsh = os.path.expanduser('~') + '/.oh-my-zsh'
        if os.path.exists(ohmyzsh):
            path = path or ohmyzsh + '/completions/_%s' % prog_name
            mode = mode or 'w'
        else:
            path = path or os.path.expanduser('~') + '/.zshrc'
            mode = mode or 'a'
    elif shell == 'powershell':
        subprocess.check_call(['powershell', 'Set-ExecutionPolicy Unrestricted -Scope CurrentUser'])
        path = path or subprocess.check_output(['powershell', '-NoProfile', 'echo $profile']).strip() if install else ''
        mode = mode or 'a'
    else:
        raise click.ClickException('%s is not supported.' % shell)

    if append is not None:
        mode = 'a' if append else 'w'
    else:
        mode = mode
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)
    f = open(path, mode)
    f.write(get_code(shell, prog_name, env_name, extra_env))
    f.write("\n")
    f.close()
    return shell, path