import subprocess
def get_output(s):
    res=subprocess.run(s.split(' '),stdout=subprocess.PIPE)
    return res.stdout.decode()
