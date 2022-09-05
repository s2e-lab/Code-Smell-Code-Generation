def check_cmake_exists(cmake_command):
    """
    Check whether CMake is installed. If not, print
    informative error message and quits.
    """
    from subprocess import Popen, PIPE

    p = Popen(
        '{0} --version'.format(cmake_command),
        shell=True,
        stdin=PIPE,
        stdout=PIPE)
    if not ('cmake version' in p.communicate()[0].decode('UTF-8')):
        sys.stderr.write('   This code is built using CMake\n\n')
        sys.stderr.write('   CMake is not found\n')
        sys.stderr.write('   get CMake at http://www.cmake.org/\n')
        sys.stderr.write('   on many clusters CMake is installed\n')
        sys.stderr.write('   but you have to load it first:\n')
        sys.stderr.write('   $ module load cmake\n')
        sys.exit(1)