def main():
    """
    Entry point for the Windows loopback tool.
    """

    parser = argparse.ArgumentParser(description='%(prog)s add/remove Windows loopback adapters')
    parser.add_argument('-a', "--add", nargs=3, action=parse_add_loopback(), help="add a Windows loopback adapter")
    parser.add_argument("-r", "--remove", action="store", help="remove a Windows loopback adapter")
    try:
        args = parser.parse_args()
    except argparse.ArgumentTypeError as e:
        raise SystemExit(e)

    # devcon is required to install/remove Windows loopback adapters
    devcon_path = shutil.which("devcon")
    if not devcon_path:
        raise SystemExit("Could not find devcon.exe")

    from win32com.shell import shell
    if not shell.IsUserAnAdmin():
        raise SystemExit("You must run this script as an administrator")

    try:
        if args.add:
            add_loopback(devcon_path, args.add[0], args.add[1], args.add[2])
        if args.remove:
            remove_loopback(devcon_path, args.remove)
    except SystemExit as e:
        print(e)
        os.system("pause")