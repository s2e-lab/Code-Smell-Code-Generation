def checkconfig():
    '''
    returns the output of lxc-checkconfig
    '''
    cmd = ['lxc-checkconfig']
    return subprocess.check_output(cmd).replace('[1;32m', '').replace('[1;33m', '').replace('[0;39m', '').replace('[1;32m', '').replace(' ', '').split('\n')