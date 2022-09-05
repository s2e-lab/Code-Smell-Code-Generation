def scp(args):
    """
    Transfer files to or from EC2 instance.

    Use "--" to separate scp args from aegea args:

        aegea scp -- -r local_dir instance_name:~/remote_dir
    """
    if args.scp_args[0] == "--":
        del args.scp_args[0]
    user_or_hostname_chars = string.ascii_letters + string.digits
    for i, arg in enumerate(args.scp_args):
        if arg[0] in user_or_hostname_chars and ":" in arg:
            hostname, colon, path = arg.partition(":")
            username, at, hostname = hostname.rpartition("@")
            hostname = resolve_instance_public_dns(hostname)
            if not (username or at):
                try:
                    username, at = get_linux_username(), "@"
                except Exception:
                    logger.info("Unable to determine IAM username, using local username")
            args.scp_args[i] = username + at + hostname + colon + path
    os.execvp("scp", ["scp"] + args.scp_args)