def cmd(send, msg, args):
    """Runs eix with the given arguments.

    Syntax: {command} <package>

    """
    if not msg:
        result = subprocess.run(['eix', '-c'], env={'EIX_LIMIT': '0', 'HOME': os.environ['HOME']}, stdout=subprocess.PIPE, universal_newlines=True)
        if result.returncode:
            send("eix what?")
            return
        send(choice(result.stdout.splitlines()))
        return
    args = ['eix', '-c'] + msg.split()
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if result.returncode:
        send("%s isn't important enough for Gentoo." % msg)
    else:
        send(result.stdout.splitlines()[0].strip())