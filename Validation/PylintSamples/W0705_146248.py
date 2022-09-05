def run_graphviz(program, code, options=[], format='png'):
    """ Runs graphviz programs and returns image data

        Copied from https://github.com/tkf/ipython-hierarchymagic/blob/master/hierarchymagic.py
    """
    import os
    from subprocess import Popen, PIPE

    dot_args = [program] + options + ['-T', format]

    if os.name == 'nt':
        # Avoid opening shell window.
        # * https://github.com/tkf/ipython-hierarchymagic/issues/1
        # * http://stackoverflow.com/a/2935727/727827
        p = Popen(dot_args, stdout=PIPE, stdin=PIPE, stderr=PIPE, creationflags=0x08000000)
    else:
        p = Popen(dot_args, stdout=PIPE, stdin=PIPE, stderr=PIPE)
        wentwrong = False

    try:
        # Graphviz may close standard input when an error occurs,
        # resulting in a broken pipe on communicate()
        stdout, stderr = p.communicate(code.encode('utf-8'))
    except (OSError, IOError) as err:
        if err.errno != EPIPE:
            raise
        wentwrong = True
    except IOError as err:
        if err.errno != EINVAL:
            raise
        wentwrong = True

    if wentwrong:
    # in this case, read the standard output and standard error streams
    # directly, to get the error message(s)
        stdout, stderr = p.stdout.read(), p.stderr.read()
        p.wait()

    if p.returncode != 0:
        raise RuntimeError('dot exited with error:\n[stderr]\n{0}'.format(stderr.decode('utf-8')))

    return stdout