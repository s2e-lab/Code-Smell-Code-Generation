import os

def get_output(s):
    return os.popen(s).read()
from subprocess import check_output

def get_output(s):
    return check_output(s.split()).decode('utf-8') 
def get_output(s):
    import subprocess
    return subprocess.check_output(s.split()).decode('ascii')
get_output = lambda c, o=__import__("os"): o.popen(c).read()
from os import popen

def get_output(s):
    return popen(s).read()
from subprocess import check_output

def get_output(s):
    return check_output(s, shell=True).decode('utf-8')
import os
get_output=lambda Q:''.join(os.popen(Q).readlines())
import subprocess
get_output=lambda s:__import__('os').popen(s).read()
from os import popen 
get_output=lambda s:popen(s).read()
import subprocess
def get_output(s):
    res=subprocess.run(s.split(' '),stdout=subprocess.PIPE)
    return res.stdout.decode()
