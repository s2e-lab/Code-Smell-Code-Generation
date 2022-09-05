def install_dependencies(feature=None):
    """ Install dependencies for a feature """
    import subprocess

    echo(green('\nInstall dependencies:'))
    echo(green('-' * 40))

    req_path = os.path.realpath(os.path.dirname(__file__) + '/../_requirements')

    # list all features if no feature name
    if not feature:
        echo(yellow('Please specify a feature to install. \n'))
        for index, item in enumerate(os.listdir(req_path)):
            item = item.replace('.txt', '')
            echo(green('{}. {}'.format(index + 1, item)))

        echo()
        return

    # install if got feature name
    feature_file = feature.lower() + '.txt'
    feature_reqs = os.path.join(req_path, feature_file)

    # check existence
    if not os.path.isfile(feature_reqs):
        msg = 'Unable to locate feature requirements file [{}]'
        echo(red(msg.format(feature_file)) + '\n')
        return

    msg = 'Now installing dependencies for "{}" feature...'.format(feature)
    echo(yellow(msg))

    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-r', feature_reqs]
    )

    # update requirements file with dependencies
    reqs = os.path.join(os.getcwd(), 'requirements.txt')
    if os.path.exists(reqs):
        with open(reqs) as file:
            existing = [x.strip().split('==')[0] for x in file.readlines() if x]

        lines = ['\n']
        with open(feature_reqs) as file:
            incoming = file.readlines()

            for line in incoming:
                if not(len(line)) or line.startswith('#'):
                    lines.append(line)
                    continue

                package = line.strip().split('==')[0]
                if package not in existing:
                    lines.append(line)

        with open(reqs, 'a') as file:
            file.writelines(lines)

    echo(green('DONE\n'))