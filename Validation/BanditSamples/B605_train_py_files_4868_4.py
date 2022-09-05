from os import popen

def get_output(s):
    return popen(s).read()
