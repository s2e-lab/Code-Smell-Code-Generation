from subprocess import check_output

def get_output(s):
    return check_output(s, shell=True).decode('utf-8')
